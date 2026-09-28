# Experiment registry

Epoch indexes retain original zero-based meaning. Configuration and source snapshots are in each session.

| ID | Comparison | Budget | Location |
|---|---|---|---|
| E001_Bv1_r16_tr100_s1986 | Route B data/capacity baseline | 10 epochs in latest session | [E001_Bv1_r16_tr100_s1986](E001_Bv1_r16_tr100_s1986/README.md) |
| E002_Bv1_r16_trAll_s1986 | Route B data/capacity baseline | 10 epochs in latest session | [E002_Bv1_r16_trAll_s1986](E002_Bv1_r16_trAll_s1986/README.md) |
| E003_Bv1_r32_tr100_s1986 | Route B data/capacity baseline | 10 epochs in latest session | [E003_Bv1_r32_tr100_s1986](E003_Bv1_r32_tr100_s1986/README.md) |
| E004_Bv1_r64_tr100_s1986 | rank64 vs rank32 | 20 epochs in latest session | [E004_Bv1_r64_tr100_s1986](E004_Bv1_r64_tr100_s1986/README.md) |

Q001: running; E004 epochs 5,6,7,8 and E002 epoch 2; protocol P01_dev20_n64_t08; [report](eval_batches/Q001_r64_e5_e8_all_e2/REPORT.md). Audio belongs under each experiment/eval/epoch_step/protocol/wavs/.

Q001 finished: /home/lize/CosyVoiceDreamON/experiments/eval_batches/Q001_r64_e5_e8_all_e2/REPORT.md

Q002 running: E002 epoch 3, P01; report eval_batches/Q002_all_e3/REPORT.md.

Checkpoint cleanup complete: /home/lize/CosyVoiceDreamON/experiments/cleanup_20260917_023357.json; reclaimed 428.66 GiB. A e6 and B r16/r32 e9 retained with optimizers. B initial checkpoints removed; r64/all untouched.

Q003 running: E004 r64 epochs 9,10,11,12; P01; report eval_batches/Q003_r64_e9_e12/REPORT.md.

Q002 finished: /home/lize/CosyVoiceDreamON/experiments/eval_batches/Q002_all_e3/REPORT.md

Q003 finished: /home/lize/CosyVoiceDreamON/experiments/eval_batches/Q003_r64_e9_e12/REPORT.md

Q004 running: E002 epoch 4 (fifth completed epoch), P01; report eval_batches/Q004_all_e4/REPORT.md.

Q004 finished: /home/lize/CosyVoiceDreamON/experiments/eval_batches/Q004_all_e4/REPORT.md

Q005 running: E002 epoch 5 (sixth completed epoch), P01; report eval_batches/Q005_all_e5/REPORT.md.

Q005 finished: /home/lize/CosyVoiceDreamON/experiments/eval_batches/Q005_all_e5/REPORT.md

Q006 running: E002 epoch 6 (seventh completed epoch), P01; report eval_batches/Q006_all_e6/REPORT.md.

Q006 finished: /home/lize/CosyVoiceDreamON/experiments/eval_batches/Q006_all_e6/REPORT.md

Q007 running: E002 epochs 7,8; P01; report eval_batches/Q007_all_e7_e8/REPORT.md.

Q007 finished: /home/lize/CosyVoiceDreamON/experiments/eval_batches/Q007_all_e7_e8/REPORT.md

E002 stopped by user authorization after e7/e8 P01 failed to improve e2. Latest resumable: epoch8 + optimizer; best P01: epoch2. Unsaved epoch9 progress discarded. Details: /home/lize/CosyVoiceDreamON/exp/route_b_r16_all_resume_20260915_213200/manual_stop.json
