# 数据、训练、生成与评测

以下项目命令从 CosyVoice-main 根目录执行，使用配置好依赖的 Linux/CUDA 环境。
训练、保存和加载的代码已接通；当前开发终端的 Python 缺少 torch，GPU 流程尚未实际验证。
评测元数据测试和 Python 语法编译已通过；训练测试因缺少 torch 无法导入。

## 与原版 CosyVoice2 的关系

训练复用原来的 `cosyvoice/bin/train.py`、Executor、DDP、验证和保存流程，数据准备也
复用原脚本。但 DreamOn 使用专用配置和 MASK 重建目标，不能套用 Qwen 自回归 loss。
训练时 eval 是验证集 loss；生成后的 eval 是 ASR 内容错误率、说话人相似度与试听。
不要直接运行原版 run.sh 所有 stage，也不要用原版 average_model.py 平均实验 checkpoint。

## 1. 数据划分和准备

从 [LibriTTS 官方页](https://www.openslr.org/60/) 下载：

| 用途 | 起步 | 扩大实验 |
| --- | --- | --- |
| 训练 | train-clean-100 | 加入 train-clean-360、train-other-500 |
| 验证与选择 checkpoint | dev-clean | 加入 dev-other |
| 最终测试 | test-clean | 同时报 test-other |

原版示例就是使用上述三个训练划分、两个 dev 划分。不能用 test 训练或反复挑选模型。
开发时先在 dev 上生成；配置固定后再报告 test。中文可以使用
[AISHELL-3](https://www.openslr.org/93/)，但必须先转换标注，不能套用 LibriTTS 目录解析。

假定解压到 `/data/LibriTTS`，准备三个起步划分：

```bash
for split in train-clean-100 dev-clean test-clean; do
  mkdir -p "data/$split/parquet"
  python examples/libritts/cosyvoice/local/prepare_data.py \
    --src_dir "/data/LibriTTS/$split" --des_dir "data/$split"
  python tools/extract_speech_token.py \
    --dir "data/$split" --onnx_path CosyVoice2-0.5B/speech_tokenizer_v2.onnx
  python tools/make_parquet_list.py \
    --src_dir "data/$split" --des_dir "data/$split/parquet" --num_processes 1
done
```

DreamOn 训练只需要文本和离线语音 token，不需要提取 speaker embedding 或训练 Flow/HiFT。
语音 token 必须来自 v2，文本在训练时使用 DreamOn tokenizer 编码。
原提取器对超过 30 秒的音频给出空 token，训练管线会过滤，先检查提取输出。

扩大数据时分别处理其他划分，再合并列表：

```bash
cat data/train-clean-100/parquet/data.list data/train-clean-360/parquet/data.list data/train-other-500/parquet/data.list > data/train.data.list
cat data/dev-clean/parquet/data.list data/dev-other/parquet/data.list > data/dev.data.list
```

## 2. 现有权重试跑与训练

```bash
python dreamon_cosyvoice_test.py --check-only
python dreamon_cosyvoice_test.py --speech-tokens 100
```

输出在 `outputs/dreamon_cosyvoice/` 的新目录里，包括原版 `baseline.wav`、拼接版
`dreamon_stitched.wav` 和诊断记录。无训练 checkpoint 时投影随机，只是接口试跑。
使用自己的输入时加 `--text`、`--prompt-text`、`--prompt-wav`；参考文本必须对应录音。

冻结 DreamOn，训练两个投影层：

```bash
bash examples/dreamon/run_train.sh \
  --freeze_dreamon true \
  --train_data data/train-clean-100/parquet/data.list \
  --cv_data data/dev-clean/parquet/data.list \
  --model_dir exp/dreamon_frozen \
  --tensorboard_dir tensorboard/dreamon_frozen
```

联合训练主干与投影层，改为 `--freeze_dreamon false` 并更换实验目录。
接着第一阶段权重训练时加 `--checkpoint exp/dreamon_frozen/epoch_0_whole.pt --weights_only`。
`--weights_only` 重置 epoch/step；两种加载方式都不恢复优化器状态。解冻是全量主干训练，
不是 LoRA；DDP 不分片模型和优化器，显存需求显著增加。

## 3. 训练中的 eval 和单条生成

每个 epoch 结束自动调用原版 Executor.cv，无需另开 eval 进程。
结果写入 TensorBoard 和 checkpoint 旁的 `.yaml`，例如 `epoch_0_whole.yaml` 的 `loss_dict`。

```bash
tensorboard --logdir tensorboard/dreamon_frozen --port 6006
```

看 `CV/loss` 和 `CV/acc`，先选验证表现好的 checkpoint，再做 dev 音频评测。
这里的 acc 是遮盖位置的语音 token 准确率，不是 ASR 准确率。DreamOn 与原版 Qwen
训练目标不同，不能直接比较两者 loss 数值。

```bash
python dreamon_cosyvoice_test.py \
  --adapter-checkpoint exp/dreamon_frozen/epoch_0_whole.pt --speech-tokens 100
```

epoch 0 只是文件名示例，不代表它最好。解冻 checkpoint 用同一个参数，会加载其主干。
`speech-tokens / 25` 约等于输出秒数；当前没有学习好的自动长度预测器。

## 4. 批量生成

开发阶段从 dev-clean 制作 20 条诊断清单：

```bash
python tools/prepare_dreamon_eval.py \
  --data-dir data/dev-clean --output data/eval/dev20.lst --limit 20
```

脚本选取同一说话人的不同参考句和目标句，按说话人轮流取样；参考 2–8 秒，目标 1–20 秒。
这是自建诊断集，不是论文官方测试清单。配置固定后，把输入改为 `data/test-clean`，
输出改为 `data/eval/test100.lst`，limit 改为 100，即可做最终小规模测试。

也支持 [SEED-TTS 官方测试集](https://github.com/BytedanceSpeech/seed-tts-eval) 的
`en/meta.lst`、`zh/meta.lst`、`zh/hardcase.lst`。清单格式是：

```text
utt_id|参考文本|参考音频路径|要合成的文本|真实目标音频路径（可省略）
```

音频相对路径相对于清单目录。两种模型使用同一份清单的原始文本，不额外规范化文本。
默认不读取最后一列的真实目标音频。

```bash
python dreamon_generate.py --backend cosyvoice \
  --meta data/eval/dev20.lst --output-dir outputs/dev20_baseline

python dreamon_generate.py --backend dreamon \
  --meta data/eval/dev20.lst --output-dir outputs/dev20_dreamon \
  --adapter-checkpoint exp/dreamon_frozen/epoch_0_whole.pt \
  --length-mode prompt-rate
```

每种模型只加载一次。输出包含 `wavs/utt_id.wav`、`tokens/`、`meta.lst`、逐句日志和
`run.json`。输出目录必须是新目录；生成失败会保留记录，评分时拒绝缺失音频。
加 `--check-only` 只检查清单和配置，不检查 GPU/模型依赖。

| DreamOn 长度模式 | 含义 |
| --- | --- |
| prompt-rate（默认） | 根据参考 token 数及参考／目标文字长度估计，不使用真实目标录音；只是启发式 |
| fixed | 使用 `--speech-tokens` 指定固定长度，仅用于简单调试 |
| oracle | 从真实目标录音提取 token 数，只用长度、不输入其 token 内容；仅用于诊断 |

oracle 额外使用真实时长，必须单独标注，不能当作正常零样本结果。
默认最大 750 个输出 token，估算截断会记入日志；超过总上下文会报错。
原版基线仍按 EOS 停止，不使用 DreamOn 长度模式。

## 5. 音频 WER/CER 和 SIM

CosyVoice2 论文用英语 Whisper-large-v3、中文 Paraformer 评估内容，用 ERes2Net
评估说话人相似度。这里接入 SEED 官方评分代码：ASR 采用上述模型；SIM 使用
SEED 的 WavLM，因此结果标记为 `SIM_WavLM`，不能直接与论文 ERes2Net 分数比较。
[论文评测说明](https://arxiv.org/html/2412.10117v1)

建议在独立评测环境准备上游依赖，避免影响训练环境。在服务器执行：

```bash
git clone https://github.com/BytedanceSpeech/seed-tts-eval.git /data/seed-tts-eval
python -m pip install -r /data/seed-tts-eval/requirements.txt
```

上游 run_wer.py 依赖 `jiwer.compute_measures`，需使用提供该 API 的兼容版本。
ASR 首次执行会下载模型，离线时需预先准备缓存。SIM 需要从上游 README 的 model link
下载 `wavlm_large_finetune.pth`。本次修改未安装评测依赖或下载这些权重。

回到项目根目录，保持使用评测环境的 Python，对两种模型分别计算英语 WER：

```bash
CUDA_VISIBLE_DEVICES=0 python dreamon_eval.py \
  --seed-eval-dir /data/seed-tts-eval \
  --meta outputs/dev20_baseline/meta.lst --wav-dir outputs/dev20_baseline/wavs \
  --metric wer --language en --output-dir outputs/dev20_baseline/wer

CUDA_VISIBLE_DEVICES=0 python dreamon_eval.py \
  --seed-eval-dir /data/seed-tts-eval \
  --meta outputs/dev20_dreamon/meta.lst --wav-dir outputs/dev20_dreamon/wavs \
  --metric wer --language en --output-dir outputs/dev20_dreamon/wer
```

中文改 `--language zh`，结果标记为 CER。评分使用上游逐句均值，保留所有高错误率样本；
不应误称为按全语料词数加权的 WER。缺失生成文件或评分条数不足会报错。

说话人相似度（原版也运行一次，替换路径即可）：

```bash
CUDA_VISIBLE_DEVICES=0 python dreamon_eval.py \
  --seed-eval-dir /data/seed-tts-eval \
  --meta outputs/dev20_dreamon/meta.lst --wav-dir outputs/dev20_dreamon/wavs \
  --metric sim --speaker-checkpoint /data/wavlm_large_finetune.pth \
  --output-dir outputs/dev20_dreamon/sim
```

脚本直接调用上游 Python，避免其集群 shell 中 sudo、GPU 环境变量和目录假设。
结果保存在 `summary.json` 与逐句 `raw_scores.txt`。上游评测使用单张可见 GPU。
比较时记录同一清单、checkpoint、长度模式和完整样本数，WER/CER 越低越好、SIM 越高越好，
并试听漏读、重复、噪声、异常停顿。当前未实现 NMOS/MOS，也没有实测分数。

## 无模型依赖的元数据测试

```bash
python -m unittest discover -s tests -p test_dreamon_eval.py -v
```

这只验证清单和评分完整性，不代表模型生成或 ASR/SIM 已通过运行验证。
