"""Route B full training: joint speech/edit NLL, no duration predictor."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from cosyvoice.llm.dreamon_route_b import build_route_b_model, EditGenerationError

ROOT = Path(__file__).resolve().parents[2]
BASE = ("loss","token_ce","action_ce","acc","action_acc","mask_fraction","canvas_ratio")
COUNTS = ("token_correct","token_count","masked_count","token_ce_sum","action_ce_sum")+tuple(
    name+"_"+suffix for name in ("fill","expand","delete") for suffix in ("correct","count","predicted"))
METRICS = BASE + COUNTS


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def batch(record):
    return dict(utts=[record["utt"]],text_tokenizer="dreamon",
                text_token=torch.tensor([record["text_ids"]],dtype=torch.long),
                text_token_len=torch.tensor([len(record["text_ids"])]),
                speech_token=torch.tensor([record["speech"]],dtype=torch.long),
                speech_token_len=torch.tensor([len(record["speech"])]))


def summarize(sums, utterances):
    raw = dict(zip(METRICS, sums))
    report = {k:raw[k]/max(1,utterances) for k in BASE}
    report.update({k:raw[k] for k in COUNTS})
    # No-speech (all-control) examples must not contribute a fabricated zero CE.
    report.update(utterances=int(utterances),
                  token_ce=raw["token_ce_sum"]/max(1,raw["token_count"]),
                  action_ce=raw["action_ce_sum"]/max(1,raw["masked_count"]),
                  acc=raw["token_correct"]/max(1,raw["token_count"]))
    report["token_accuracy"] = report["acc"]
    for name in ("fill","expand","delete"):
        report[name+"_recall"] = raw[name+"_correct"]/raw[name+"_count"] if raw[name+"_count"] else None
        report[name+"_precision"] = raw[name+"_correct"]/raw[name+"_predicted"] if raw[name+"_predicted"] else None
    return report


@torch.inference_mode()
def evaluate(model, records, local, rank, world, ratio, edits=True):
    model.eval()
    model.validation_mask_ratio = ratio
    model.validation_edits = edits
    sums = torch.zeros(len(METRICS)+1,device=local,dtype=torch.float64)
    try:
        for index in range(rank,len(records),world):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                result = model(batch(records[index]),local)
            sums[:-1] += torch.stack([result[k].double() for k in METRICS])
            sums[-1] += 1
        dist.all_reduce(sums)
        return summarize(sums[:-1].tolist(),float(sums[-1])) | dict(mask_ratio=ratio,edits=edits)
    finally:
        model.validation_edits = True


@torch.inference_mode()
def probe(model, dev, directory, rank, world, max_steps=800):
    model.eval()
    candidates = [r for r in dev if 40 <= len(r["speech"]) <= 120][:2]
    # Include real initial-canvas mismatch; target length is never passed to generation.
    tasks = [(r,n,temp) for r in candidates for n in (16,64,128) for temp in (0.,.8)]
    directory.mkdir(parents=True,exist_ok=True)
    records = []
    for index,(r,n,temp) in enumerate(tasks):
        if index % world != rank:
            continue
        started = time.monotonic()
        try:
            tokens,report = model.bridge.generate_dynamic(
                r["text"],initial_masks=n,maximum=350,max_steps=max_steps,
                temperature=temp,action_temperature=temp,seed=1986)
            torch.save(tokens,directory/(r["utt"]+"_"+str(n)+"_t"+str(temp)+".pt"))
        except EditGenerationError as exc:
            report = exc.report
        report.update(utt=r["utt"],text=r["text"],elapsed_seconds=time.monotonic()-started,
                      reference_tokens_for_evaluation_only=len(r["speech"]))
        write_json(directory/(r["utt"]+"_"+str(n)+"_t"+str(temp)+".json"),report)
        records.append({k:v for k,v in report.items() if k!="steps"})
    gathered = [None]*world
    dist.all_gather_object(gathered,records)
    if rank==0:
        rows = [r for group in gathered for r in group]
        result = dict(attempts=len(rows),completed=sum(r["status"]=="complete" for r in rows),results=rows,
                      note="Token/length diagnostics; completion does not establish intelligible audio.")
        write_json(directory/"summary.json",result)
        print("GENERATION_PROBE",json.dumps({k:v for k,v in result.items() if k!="results"}),flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--data-cache",type=Path,required=True)
    p.add_argument("--dreamon-model-dir",type=Path,default=ROOT/"DreamOn-v0-7B")
    p.add_argument("--cosyvoice-model-dir",type=Path,default=ROOT/"CosyVoice2-0.5B")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--max-steps",type=int,default=0,help="0: complete all epochs; positive: bounded diagnostic.")
    p.add_argument("--eval-interval",type=int,default=2000)
    p.add_argument("--probe-max-steps",type=int,default=800)
    p.add_argument("--lora-rank",type=int,default=16)
    p.add_argument("--lora-alpha",type=float,default=32.)
    p.add_argument("--lora-lr",type=float,default=1e-4)
    p.add_argument("--projection-lr",type=float,default=1e-4)
    p.add_argument("--max-sequence-tokens",type=int,default=2048)
    p.add_argument("--seed",type=int,default=1986)
    args = p.parse_args()
    if args.epochs<1 or args.max_steps<0 or args.eval_interval<1 or args.probe_max_steps<1:
        raise ValueError("Invalid epoch/step settings.")
    return args


def main():
    args = arguments()
    os.chdir(ROOT)
    rank,local,world = (int(os.environ[k]) for k in ("RANK","LOCAL_RANK","WORLD_SIZE"))
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    dist.init_process_group("nccl",timeout=datetime.timedelta(minutes=45))
    writer = None
    step = 0
    started = time.monotonic()
    train_seconds = 0.
    best_ce = float("inf")
    try:
        if rank==0:
            args.model_dir.mkdir(parents=True,exist_ok=False)
            write_json(args.model_dir/"status.json",dict(status="loading_model",step=0))
        dist.barrier()
        cache = json.loads(args.data_cache.read_text())
        for split in ("train","dev"):
            info = cache["summary"][split]
            if hashlib.sha256(Path(info["source_list"]).read_bytes()).hexdigest()!=info["list_sha256"]:
                raise ValueError("Dataset list changed; reprepare cache.")
        records,dev = cache["train"],cache["dev"]
        if not records or not dev or {r["utt"] for r in records}&{r["utt"] for r in dev}:
            raise ValueError("Empty or overlapping data splits.")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = build_route_b_model(args.dreamon_model_dir,args.cosyvoice_model_dir,
                                   max_sequence_tokens=args.max_sequence_tokens,seed=args.seed,
                                   lora_rank=args.lora_rank,lora_alpha=args.lora_alpha,lora_lr=args.lora_lr)
        model.cuda(local)
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(),lr=args.projection_lr,weight_decay=.01)
        ddp = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],
                    find_unused_parameters=False,gradient_as_bucket_view=True)
        params = [p for p in model.parameters() if p.requires_grad]
        sampler = DistributedSampler(records,num_replicas=world,rank=rank,seed=args.seed,shuffle=True)
        cv = [dev[i] for i in sorted(random.Random(args.seed).sample(range(len(dev)),min(80,len(dev))))]
        if rank==0:
            writer = SummaryWriter(str(args.model_dir/"tensorboard"))
            manifest = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
            counts = model.trainable_parameter_counts()
            manifest.update(world_size=world,gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES"),
                trainable_parameters=counts,total_trainable_parameters=sum(counts.values()),
                train_samples=len(records),dev_samples=len(dev),steps_per_epoch=len(sampler),
                batch_size_per_gpu=1,effective_batch_size=world,accum_grad=1,
                duration_predictor=False,initialization="original_pretrained_DreamOn_CosyVoice_fresh_LoRA_projections_edit_head",
                checkpoint_scope="full_DreamOn_backbone_LoRA_projections_edit_head; original_frozen_CosyVoice_required",
                objective="per_utterance_joint_action_and_conditional_speech_NLL",
                speech_conditioning="target_text_only",early_stopping=False,
                checkpoint_schedule="step100_step1000_and_every_completed_epoch; no deletions",
                validation="fixed80_edit_full/edit_half/noedit_full; full_dev_edit_full_at_epoch_end",
                cache_sha256=hashlib.sha256(args.data_cache.read_bytes()).hexdigest(),
                source_data=cache["summary"],implementation_revision="b0.1")
            write_json(args.model_dir/"manifest.json",manifest)
            source_dir=args.model_dir/"source"
            hashes={}
            for f in ["cosyvoice/llm/dreamon_route_b.py","cosyvoice/llm/dreamon_speech.py",
                      "cosyvoice/llm/dreamon_training.py","cosyvoice/llm/dreamon_lora.py",
                      "cosyvoice/bin/train_route_b.py","examples/dreamon/run_route_b.sh",
                      "tests/test_dreamon_route_b.py","dreamon_generate.py"]:
                dest=source_dir/f
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/f,dest)
                hashes[f]=hashlib.sha256(dest.read_bytes()).hexdigest()
            write_json(args.model_dir/"source_hashes.json",hashes)
            print("MODEL_READY",json.dumps(manifest),flush=True)
        dist.barrier()
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            for bidx,index in enumerate(sampler,1):
                tick=time.monotonic()
                ddp.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    losses=ddp(batch(records[index]),local)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("Nonfinite loss")
                losses["loss"].backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
                step+=1
                train_seconds+=time.monotonic()-tick
                if step<=3 or step%10==0:
                    values=torch.stack([losses[k].detach().double() for k in METRICS]+[norm.detach().double()])
                    dist.all_reduce(values)
                    if rank==0:
                        result=summarize(values[:-1].tolist(),world)
                        result["grad_norm"]=float(values[-1])/world
                        record=dict(status="training",epoch=epoch,batch=bidx,total_batches=len(sampler),step=step,
                                    epoch_eta_seconds=(len(sampler)-bidx)*train_seconds/step,
                                    elapsed_seconds=time.monotonic()-started,**result)
                        write_json(args.model_dir/"status.json",record)
                        with (args.model_dir/"metrics.jsonl").open("a") as out:
                            out.write(json.dumps(record)+"\n")
                        for k,v in result.items():
                            if v is not None:writer.add_scalar("TRAIN/"+k,v,step)
                        for group in optimizer.param_groups:
                            writer.add_scalar("LR/"+group["name"],group["lr"],step)
                        print("TRAIN Epoch",epoch,"Batch",str(bidx)+"/"+str(len(sampler)),
                              json.dumps({k:record[k] for k in BASE+("grad_norm","epoch_eta_seconds")}),flush=True)
                epoch_end=bidx==len(sampler)
                limit=bool(args.max_steps and step>=args.max_steps)
                save_now=step in (100,1000) or epoch_end or limit
                validate=save_now or step%args.eval_interval==0
                if validate:
                    if rank==0:
                        write_json(args.model_dir/"status.json",dict(status="validating",epoch=epoch,step=step,full_dev=epoch_end))
                    cv_full=evaluate(model,cv,local,rank,world,1.)
                    cv_half=evaluate(model,cv,local,rank,world,.5)
                    cv_noedit=evaluate(model,cv,local,rank,world,1.,edits=False)
                    full_dev=evaluate(model,dev,local,rank,world,1.) if epoch_end else None
                    name=f"epoch_{epoch}_whole" if epoch_end else f"step_{step:06d}"
                    if rank==0:
                        checkpoint=None
                        if save_now:
                            if shutil.disk_usage(args.model_dir).free<35*1024**3:
                                raise RuntimeError("Insufficient checkpoint disk space.")
                            payload=model.training_checkpoint(dict(epoch=epoch,step=step))
                            payload["optimizer_steps_completed"]=step
                            temp=args.model_dir/(name+".pt.tmp")
                            torch.save(payload,temp)
                            temp.replace(args.model_dir/(name+".pt"))
                            del payload
                            opt=args.model_dir/(name+"_optimizer.pt.tmp")
                            torch.save(dict(optimizer=optimizer.state_dict(),epoch=epoch,step=step),opt)
                            opt.replace(args.model_dir/(name+"_optimizer.pt"))
                            checkpoint=str(args.model_dir/(name+".pt"))
                        result=dict(step=step,epoch=epoch,checkpoint=checkpoint,
                                    CV_full=cv_full,CV_half=cv_half,CV_noedit_full=cv_noedit,full_dev=full_dev)
                        write_json(args.model_dir/(name+".json"),result)
                        for group,metrics in [("CV_full",cv_full),("CV_half",cv_half),
                                               ("CV_noedit_full",cv_noedit),("DEV_full",full_dev)]:
                            if metrics is not None:
                                for k,v in metrics.items():
                                    if v is not None:writer.add_scalar(group+"/"+k,v,step)
                        if epoch_end and cv_noedit["token_ce"]<best_ce:
                            best_ce=cv_noedit["token_ce"]
                            write_json(args.model_dir/"best_checkpoint.json",dict(checkpoint=checkpoint,
                                criterion="fixed80_noedit_fullmask_token_CE; not an audio-quality ranking",
                                value=best_ce,epoch=epoch))
                        writer.flush()
                        print("VALIDATION",json.dumps(result),flush=True)
                    dist.barrier()
                    if step>=1000 or epoch_end or limit:
                        if rank==0:
                            write_json(args.model_dir/"status.json",dict(status="generation_probe",epoch=epoch,step=step))
                        probe(model,dev,args.model_dir/"probes"/name,rank,world,args.probe_max_steps)
                    dist.barrier()
                if limit:break
            if args.max_steps and step>=args.max_steps:break
        if rank==0:
            write_json(args.model_dir/"status.json",dict(status="complete",step=step,
                fully_completed_epochs=epoch+int(epoch_end),
                stopped_at_step_cap=bool(args.max_steps and step>=args.max_steps),
                checkpoint=str(args.model_dir/(name+".pt")),elapsed_seconds=time.monotonic()-started))
    except Exception:
        if args.model_dir.is_dir():
            failure=dict(status="failed",step=step,error=traceback.format_exc())
            write_json(args.model_dir/f"failure_rank_{rank}.json",failure)
            if rank==0:write_json(args.model_dir/"status.json",failure)
        raise
    finally:
        if writer:writer.close()
        dist.destroy_process_group()


if __name__=="__main__":
    main()
