# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import datetime
import logging
import math
import random
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist
try:
    import deepspeed
except ImportError:
    deepspeed = None

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.losses import DPOLoss
from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)
from cosyvoice.utils.file_utils import read_lists


def _epoch_batch_total(shard_counts, batch_size, epoch, shuffle, partition,
                       rank, world_size, num_workers):
    """Mirror DataList shard assignment and return this rank's batch count."""
    indexes = list(range(len(shard_counts)))
    if partition:
        if shuffle:
            random.Random(epoch).shuffle(indexes)
        if len(indexes) < world_size:
            indexes = (indexes * math.ceil(world_size / len(indexes)))[:world_size]
        indexes = indexes[rank::world_size]
    workers = max(1, num_workers)
    if len(indexes) < workers:
        indexes = (indexes * math.ceil(workers / len(indexes)))[:workers]
    total = 0
    for worker_id in range(workers):
        usable = sum(shard_counts[index] for index in indexes[worker_id::workers])
        total += math.ceil(usable / batch_size)
    return total


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--ref_model', required=False, help='ref model used in dpo')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--dreamon_model_dir', help='override DreamOn speech training base-model path')
    parser.add_argument('--cosyvoice_model_dir', help='override DreamOn speech training codec-model path')
    parser.add_argument('--freeze_dreamon', choices=['true', 'false'], default=None,
                        help='override YAML: true trains projections; false also trains DreamOn')
    parser.add_argument('--dreamon_lr', type=float, help='override DreamOn backbone learning rate')
    parser.add_argument('--use_lora', choices=['true', 'false'], default=None,
                        help='train DreamOn LoRA plus projections; requires --freeze_dreamon true')
    parser.add_argument('--lora_rank', type=int)
    parser.add_argument('--lora_alpha', type=float)
    parser.add_argument('--lora_dropout', type=float)
    parser.add_argument('--lora_lr', type=float)
    parser.add_argument('--lora_target_modules', nargs='+',
                        help='linear layer names, default q_proj k_proj v_proj o_proj')
    parser.add_argument('--batch_size', type=int,
                        help='override per-GPU DreamOn batch size')
    parser.add_argument('--weights_only', action='store_true',
                        help='load DreamOn experiment weights but reset epoch/step for a new training stage')
    parser.add_argument('--onnx_path', required=False, help='onnx path, which is required for online feature extraction')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--math_sdpa', action='store_true', help='use math SDPA for numerical diagnostics')
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    if deepspeed is not None:
        parser = deepspeed.add_config_arguments(parser)
    else:
        parser.add_argument('--deepspeed_config', help='ignored with torch_ddp; keeps recipe CLI compatibility')
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if args.onnx_path is not None:
        os.environ['onnx_path'] = args.onnx_path
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # gan train has some special initialization logic
    gan = True if args.model == 'hifigan' else False

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    if args.qwen_pretrain_path is not None:
        override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path
    for key in ('dreamon_model_dir', 'cosyvoice_model_dir', 'batch_size'):
        if getattr(args, key) is not None:
            override_dict[key] = getattr(args, key)
    if args.freeze_dreamon is not None:
        override_dict['freeze_dreamon'] = args.freeze_dreamon == 'true'
    if args.dreamon_lr is not None:
        override_dict['dreamon_lr'] = args.dreamon_lr
    if args.use_lora is not None:
        override_dict['use_lora'] = args.use_lora == 'true'
    for key in ('lora_rank', 'lora_alpha', 'lora_dropout', 'lora_lr', 'lora_target_modules'):
        if getattr(args, key) is not None:
            override_dict[key] = getattr(args, key)
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))
    if args.weights_only and (not args.checkpoint or configs.get('model_family') != 'dreamon_speech_adapter'):
        raise ValueError('--weights_only requires --checkpoint and the DreamOn training configuration.')
    if configs.get('model_family') == 'dreamon_speech_adapter':
        if args.model != 'llm' or args.dpo or args.train_engine != 'torch_ddp':
            raise ValueError('DreamOn speech training uses --model llm --train_engine torch_ddp without --dpo.')
        configs['train_conf']['freeze_dreamon'] = configs['freeze_dreamon']
        configs['train_conf']['dreamon_lr'] = configs['dreamon_lr']
        for key in ('use_lora', 'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_lr', 'lora_target_modules'):
            configs['train_conf'][key] = configs[key]
        if args.use_amp and not torch.cuda.is_bf16_supported():
            raise ValueError('This training configuration uses BF16 AMP. Use FP32 weights without --use_amp on this GPU.')
        if configs['batch_size'] < 1:
            raise ValueError('--batch_size must be positive.')

    # Init env for ddp
    init_distributed(args)

    dreamon_shard_counts = None
    if configs.get('model_family') == 'dreamon_speech_adapter':
        from cosyvoice.dataset.dreamon_processor import count_usable_samples_by_shard
        train_shards = read_lists(args.train_data)
        cv_shards = read_lists(args.cv_data)
        counts_holder = [None, None]
        if dist.get_rank() == 0:
            logging.info('Counting usable DreamOn samples for exact epoch progress')
            counts_holder[0] = count_usable_samples_by_shard(
                train_shards, configs['get_tokenizer'], configs['max_sequence_tokens'])
            counts_holder[1] = count_usable_samples_by_shard(
                cv_shards, configs['get_tokenizer'], configs['max_sequence_tokens'])
        dist.broadcast_object_list(counts_holder, src=0)
        dreamon_shard_counts = tuple(counts_holder)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, args.dpo)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint
    if args.dpo is True:
        configs[args.model].forward = configs[args.model].forward_dpo
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            if hasattr(model, 'load_training_checkpoint'):
                state_dict = model.load_training_checkpoint(args.checkpoint)
                logging.warning('Loaded DreamOn experiment checkpoint (%s); optimizer state is not included.',
                                state_dict.get('checkpoint_scope', 'projection_weights_only'))
            else:
                state_dict = torch.load(args.checkpoint, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
            if not args.weights_only and 'step' in state_dict:
                start_step = state_dict['step']
            if not args.weights_only and 'epoch' in state_dict:
                start_epoch = state_dict['epoch']
            del state_dict
        else:
            if hasattr(model, 'load_training_checkpoint'):
                raise FileNotFoundError(args.checkpoint)
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)
    scheduler.set_step(start_step)
    if scheduler_d is not None:
        scheduler_d.set_step(start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    # DPO related
    if args.dpo is True:
        ref_model = deepcopy(configs[args.model])
        state_dict = torch.load(args.ref_model, map_location='cpu')
        ref_model.load_state_dict(state_dict, strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        # NOTE maybe it is not needed to wrap ref_model as ddp because its parameter is not updated
        ref_model = wrap_cuda_model(args, ref_model)
    else:
        ref_model, dpo_loss = None, None

    # Get executor
    executor = Executor(gan=gan, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.step = start_step

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    print('start step {} start epoch {}'.format(start_step, start_epoch))

    # Start training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        if dreamon_shard_counts is not None:
            train_counts, cv_counts = dreamon_shard_counts
            info_dict['train_total_batches'] = _epoch_batch_total(
                train_counts, configs['batch_size'], epoch, True, True,
                dist.get_rank(), dist.get_world_size(), args.num_workers)
            info_dict['cv_total_batches'] = _epoch_batch_total(
                cv_counts, configs['batch_size'], epoch, False, False,
                dist.get_rank(), dist.get_world_size(), args.num_workers)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        if gan is True:
            executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                        writer, info_dict, scaler, group_join)
        else:
            executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=ref_model)
        dist.destroy_process_group(group_join)


if __name__ == '__main__':
    main()
