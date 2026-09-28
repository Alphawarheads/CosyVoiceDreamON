# DreamOn × CosyVoice2：固定长度适配训练

数据划分、训练、批量生成和音频评测完整命令见 [TRAIN_EVAL.md](TRAIN_EVAL.md)。

训练复用 `cosyvoice/bin/train.py` 和原来的 Executor、DDP、优化器、验证、日志。
`DreamOnSpeechTrainer.forward(batch, device)` 返回 `loss`、`acc`、`mask_fraction`。
默认只训练 `896→3584` 和 `3584→896` 两层投影；`freeze_dreamon: false` 时，
DreamOn 主干也参与训练。新增 `--use_lora true --freeze_dreamon true` 可训练内部 LoRA。
这些模式都冻结 CosyVoice 语音 embedding／输出头／任务
embedding，Flow、HiFT 继续使用原权重。原始模型目录不会被覆盖。

目标是固定长度的语音 MASK 重建；动态扩张／删除、自动预测输出长度、DPO 和
DeepSpeed 训练尚不包含在这阶段。成功保存检查点不代表已经获得可懂语音。

## 环境

在 Linux CUDA 环境运行训练，使用已有的项目依赖。运行环境目标为
torch/torchaudio 2.5.1、transformers 4.51.3，另需 accelerate、HyperPyYAML、
tensorboard；读取 Parquet 需要 pyarrow。DDP 训练不要求安装 DeepSpeed。
原根目录 requirements.txt 固定了 torch 2.3.1，安装时避免把 DreamOn 环境降级。
完整音频推理还需要 `third_party/Matcha-TTS` 的实际源码，空目录不够。
当前开发终端已能调用 Python，但缺少 torch；训练适配测试和真实 7B 训练尚未完成运行验证。
新增评测清单测试与 Python 语法编译已通过，不能据此认为 GPU 推理或训练已经通过。

所有命令从项目根目录执行。基座默认位于 `DreamOn-v0-7B/`、`CosyVoice2-0.5B/`。
所有模型加载都使用本地文件。训练的缓存路径由启动脚本设置。

## 第一步：先用已有权重试跑

```bash
python dreamon_cosyvoice_test.py --check-only
python dreamon_cosyvoice_test.py --speech-tokens 100
```

配置在 `configs/dreamon_cosyvoice.yaml`，先保持 `adapter_checkpoint: null`。
输出位于 `outputs/dreamon_cosyvoice/` 下的新时间戳目录，包括 `baseline.wav`、
`dreamon_stitched.wav`、token 文件和 `diagnostics.json`。已有权重会自动加载，
两个投影层此时是随机初始化；这一步用于检查接口与数值，不保证可懂语音。
冻结开关只影响训练，不影响这一步推理。Downloads 中的代码补全诊断脚本不是语音入口。

## 数据集下载

本地 `examples/libritts/cosyvoice2/run.sh` 使用 **LibriTTS**。下面是可下载的公开
实验数据；它们不是 CosyVoice2 已发布权重全部预训练语料的打包下载。

| 数据集 | 用途与下载 |
| --- | --- |
| [LibriTTS 官方页](https://www.openslr.org/60/) | 英语，多说话人。先下载 [train-clean-100（7.7G，CN 镜像）](https://openslr.magicdatatech.com/resources/60/train-clean-100.tar.gz) 和 [dev-clean（1.2G，CN 镜像）](https://openslr.magicdatatech.com/resources/60/dev-clean.tar.gz)。官方页提供其他镜像、划分和校验值。 |
| [LibriTTS-R 官方页](https://www.openslr.org/141/) | LibriTTS 的音质恢复版本，可作为后续英语实验数据；文件名使用下划线，如 `train_clean_100.tar.gz`。 |
| [AISHELL-3 官方页](https://www.openslr.org/93/) | 中文，多说话人，约 85 小时；[数据与文本（19G，CN 镜像）](https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz)。需要先将其标注转换成下面的 `wav.scp/text/utt2spk`，不能直接使用 LibriTTS 的目录解析脚本。 |

LibriTTS 的 `train-clean-100` 是沿用源数据划分的名称，不表示该 TTS 子集恰好有 100 小时。
先取少量干净训练样本验证，再扩大数据。训练和验证集应独立；完整下载包需要自行下载、
校验及解压，本次修改没有下载这些大文件。

例如 LibriTTS 已解压到 `/data/LibriTTS`，从项目根目录准备元数据：

```bash
mkdir -p data/train data/dev
python examples/libritts/cosyvoice/local/prepare_data.py --src_dir /data/LibriTTS/train-clean-100 --des_dir data/train
python examples/libritts/cosyvoice/local/prepare_data.py --src_dir /data/LibriTTS/dev-clean --des_dir data/dev
```

## 准备数据

直接复用已经包含 `utt`、`text`、`speech_token` 的 CosyVoice Parquet 分片及
`data.list` 即可。后缀为 `.tar` 的原版 Parquet 分片也支持。
必须使用 **speech_tokenizer_v2.onnx** 提取 token，不能混用 v1/v3 或 DreamOn 文本 ID。
原版 CosyVoice2 run.sh 的离线提取路径已修正为 v2。

从原版数据目录生成分片的示例（事先准备 wav.scp、text、utt2spk）：

```bash
python tools/extract_speech_token.py --dir data/train --onnx_path CosyVoice2-0.5B/speech_tokenizer_v2.onnx
mkdir -p data/train/parquet
python tools/make_parquet_list.py --src_dir data/train --des_dir data/train/parquet --num_processes 1
```

对验证集目录做同样处理，并保持训练／验证划分独立。这阶段不需要提取说话人
embedding 或 mel，数据管线直接读取离线语音 token，不加载 Parquet 中的音频字节。
原提取器对超过 30 秒的录音会返回空 token，应先将长录音切成有对应文本的片段。

也可以提供 JSONL，每行一条记录。以下只是格式示意，语音 ID 必须换成真实提取结果：

```json
{"utt":"utt_001","text":"你好。","speech_token":[12,35,8],"speech_tokenizer":"cosyvoice2"}
```

`train.data.list` 和 `dev.data.list` 每行放一个 JSONL 或 Parquet 文件路径，推荐绝对路径。
不要把 JSONL 记录直接放进 data.list。超过总上下文限制的样本会被过滤，并记录数量；
文本 token 数 + 语音 token 数 + 3 必须不大于配置中的 `max_sequence_tokens`。
如果全部样本被过滤，训练会报错。

## 训练

在 `configs/dreamon_cosyvoice_train.yaml` 修改一个布尔值即可；不要给布尔值加引号：

```yaml
freeze_dreamon: true   # false = DreamOn 主干 + 两个投影层一起训练
dreamon_lr: 0.00001    # 只在解冻时使用；投影层 LR 仍取 optim_conf.lr，默认 0.0001
```

也可以通过命令行覆盖配置。先冻结训练两个投影层：

```bash
bash examples/dreamon/run_train.sh \
  --freeze_dreamon true \
  --model_dir exp/dreamon_frozen \
  --tensorboard_dir tensorboard/dreamon_frozen \
  --train_data data/train/parquet/data.list \
  --cv_data data/dev/parquet/data.list
```

解冻 DreamOn，与两个投影层一起训练（单独的实验目录）：

```bash
bash examples/dreamon/run_train.sh \
  --freeze_dreamon false --dreamon_lr 0.00001 \
  --model_dir exp/dreamon_unfrozen \
  --tensorboard_dir tensorboard/dreamon_unfrozen \
  --train_data data/train/parquet/data.list \
  --cv_data data/dev/parquet/data.list
```

两条命令都从现有基座权重初始化。若要把第一种训练结果作为第二阶段起点，在第二条命令追加：

```bash
--checkpoint exp/dreamon_frozen/epoch_0_whole.pt --weights_only
```

`--weights_only` 重置 epoch/step，适合切换阶段或开展独立实验。加载 checkpoint 不会
覆盖当前 YAML/CLI 的冻结设置。切换模式需要重新启动训练，不能在已创建的 DDP／优化器
上直接翻转 `requires_grad`。

解冻模式是 **DreamOn 主干全量训练**，包含其原生输入 embedding，不是 LoRA。
可训练权重保持 FP32，前向默认 BF16 AMP，Adam 状态也是 FP32，因此显存、CPU 内存和
checkpoint 体积都会显著增加。当前 DDP 每张卡保留完整模型和优化器状态，增加卡数不会
将它们分片；不能把 7B 推理所需显存当作全量训练所需显存。梯度检查点默认开启。

脚本最终调用原来的 `python -m torch.distributed.run ... -m cosyvoice.bin.train`。
配置为 `configs/dreamon_cosyvoice_train.yaml`，默认单 GPU、batch_size=1、10 个 epoch。
增加 GPU 数量：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash examples/dreamon/run_train.sh \
  --train_data data/train/parquet/data.list --cv_data data/dev/parquet/data.list
```

不同权重位置可以追加：

```bash
--dreamon_model_dir /home/lize/DreamOn/DreamOn-v0-7B \
--cosyvoice_model_dir /path/to/CosyVoice2-0.5B
```

初次训练不要把原版 `llm.pt` 传给 `--checkpoint`；配置中的工厂已经分别加载
DreamOn 主干和 CosyVoice 语音模块。`--checkpoint` 只接受本实验的适配检查点。
若遇到 SDPA 非有限数值，可追加 `--math_sdpa` 诊断。

训练会随机选取同一条录音的前段作为可见参考，剩余部分作为预测目标；完整文本
仍作为条件。目标随机遮盖，包含全 MASK 样本，损失只覆盖被遮盖的位置。
padding 按长度移除，不进入注意力，也不计入损失。
验证时按 utterance ID 固定参考切分及 MASK，因此验证指标可比较。
指标是遮盖 token 的交叉熵与准确率，不是语音听感或 ASR 指标。

## 保存、加载与推理

默认输出 `exp/dreamon_speech/init.pt`、`epoch_0_whole.pt` 等。
指定 `--model_dir` 时保存到指定目录。冻结原始 DreamOn 的模式只保存两层投影及元数据；
解冻模式还保存完整的 DreamOn 主干。加载已训练的主干后即使再次冻结，后续 checkpoint
也继续保存该主干，避免遗漏训练结果。冻结的 CosyVoice 语音模块依然从同一个基座加载。
v3 checkpoint 支持这两种范围，旧版 v1/v2 投影 checkpoint 仍能加载。

```bash
python dreamon_cosyvoice_test.py \
  --adapter-checkpoint exp/dreamon_frozen/epoch_0_whole.pt
```

解冻模式只需将路径换成 `exp/dreamon_unfrozen/epoch_0_whole.pt`，推理脚本自动加载
checkpoint 中的投影与 DreamOn 主干。它仍会生成原始 CosyVoice 对照；诊断文件的
`loaded_finetuned_backbone` 记录是否加载了微调主干。推理按所选 `--dtype` 转换权重。
输出长度仍由 `--speech-tokens` 指定；训练数据中的实际语音长度提供训练画布长度。

继续训练可以追加 `--checkpoint exp/dreamon_speech/epoch_0_whole.pt`。
与原版 DDP `.pt` 一样，这会加载模型和 epoch/step，但**不恢复优化器或 RNG 状态**，
不是逐步完全一致的断点续训。新实验可加 `--weights_only`，同时更换输出目录。
推理只为投影模式另存 `adapter_loaded.pt`，包含完整主干时记录源 checkpoint 路径，
不重复复制大型 checkpoint。不要用原版 average_model.py 平均此格式。

## 不加载大模型的验证

```bash
python -m unittest discover -s tests -p 'test_dreamon*.py' -v
```

测试使用小型 CPU 模型验证 MASK 损失、padding、参数更新、验证确定性、token
合法性以及训练检查点加载后生成结果的一致性。安装 HyperPyYAML/pyarrow 后，
还会检查训练配置的构建和 Parquet 列读取。
冻结开关测试覆盖两种模式的实际参数更新、分组学习率、完整主干 checkpoint 的加载、
再次冻结后的保存，以及错误 checkpoint 在写入参数前被拒绝。


## LoRA 训练与加载

完整的 100/all 训练、生成、eval 命令见根目录 [cmd.txt](../../cmd.txt) 第七节。

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash examples/dreamon/run_train.sh \
  --freeze_dreamon true --use_lora true \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --lora_lr 0.0001 \
  --checkpoint exp/dreamon_frozen_train_100/epoch_3_whole.pt --weights_only \
  --train_data /home/lize/AudioData/libritts/prepared/lists/train_100.data.list \
  --cv_data /home/lize/AudioData/libritts/prepared/lists/dev_all.data.list \
  --model_dir exp/dreamon_lora_r16_train_100_from_frozen_epoch3 \
  --tensorboard_dir exp/dreamon_lora_r16_train_100_from_frozen_epoch3/tensorboard
```

使用原生 PyTorch 的标准 LoRA：`W x + (alpha / rank) B A x`，不要求 PEFT。
默认覆盖所有层的 `q_proj/k_proj/v_proj/o_proj`，可用 `--lora_target_modules` 覆盖。
A 使用 Kaiming 初始化，B 为零，初始 LoRA 不改变原模型输出。LoRA 参数及投影参数
保持 FP32，原 DreamOn 主干保留 BF16，前向复用现有 BF16 AMP 和非重入梯度检查点。
LoRA LR 独立于 `dreamon_lr`，后者仍只用于全量主干训练。

LoRA 模式需要 `freeze_dreamon=true`，同时全量解冻会报错。默认 `use_lora=false`
保留此前冻结／全量训练行为。已有 projection checkpoint（版本 1/2/3）可以初始化 LoRA；
LoRA checkpoint（版本 4）可以继续训练或直接用于所有调用 `load_adapter_checkpoint`
的生成入口。继续 LoRA 训练时保持 rank、alpha、dropout、target_modules 一致。
载入权重不会替你更改 CLI 训练模式：想训练 LoRA 时始终传 `--use_lora true`。

版本 4 checkpoint 保存**完整主干 + LoRA A/B + 两个投影层**，并携带 LoRA 结构参数。
生成时自动安装 LoRA，再严格检查并加载所有张量；不需要另存或手动合并 LoRA。
这些 `.pt` 仍依赖原目录的 tokenizer、模型结构以及 CosyVoice 组件，不是独立模型目录。
小体积 LoRA-only 导出、量化和分片训练暂未包含。不要用旧版本项目代码读取版本 4。
旧版全量解冻 checkpoint 不能直接加载到已安装 LoRA 的主干，需先加载再启用 LoRA；
当前命令行建议直接从已有冻结投影 checkpoint 开始。

单卡显存通常显著低于全量微调，但仍包含完整主干和激活。DDP 不分片参数；增加卡数
不能保证解决单卡 OOM。建议先观察第一个 batch 的日志和显存，再继续完整实验。

本次 CPU 回归验证使用临时 Python 3.14 / PyTorch 2.10 环境：39 项通过，1 项因
PyArrow 未安装跳过；覆盖梯度、BF16 冻结权重、梯度检查点、旧 checkpoint 初始化、
完整 LoRA checkpoint 加载与生成一致性、CLI/YAML。服务器 PyTorch 2.5.1 的真实
7B GPU 训练、显存占用和音频效果尚未实测。

```bash
python -m unittest discover -s tests -p 'test_dreamon_*.py' -v
```
