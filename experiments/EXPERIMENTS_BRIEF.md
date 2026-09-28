# Experiment brief registry
E005 all e2 inference with DELETE sampling relative weights 0.75 and 0.5.
E006 all e2 oracle fixed-length forced-FILL diagnostic using cached target lengths.
E007 all e2/e8 teacher-forced text dependence diagnostic with correct, mismatched, and empty text.
E008 C100 epochs 4/5/6 evaluated on P01 with generated audio and WER.
E009 C-all epoch 0 evaluated on P01 with generated audio and WER.

C-all-e1: Route C r16 all-data checkpoint, P01 20-sentence generation and WER; /home/lize/CosyVoiceDreamON/experiments/eval_batches/C_all_epoch1_P01

C-all-e2: Route C r16 all-data checkpoint, P01 20-sentence generation and WER; /home/lize/CosyVoiceDreamON/experiments/eval_batches/C_all_epoch2_P01

C-all-e3: Route C r16 all-data checkpoint, P01 20-sentence generation and WER; /home/lize/CosyVoiceDreamON/experiments/eval_batches/C_all_epoch3_P01

C-all-e4: Route C r16 all-data checkpoint, P01 20-sentence generation and WER; /home/lize/CosyVoiceDreamON/experiments/eval_batches/C_all_epoch4_P01

C-all-e5: Route C r16 all-data checkpoint, P01 20-sentence generation and WER; /home/lize/CosyVoiceDreamON/experiments/eval_batches/C_all_epoch5_P01
E008 B all e2: predicted versus real speech tokens through CosyVoice tts() and direct token2wav(), with matched references and seeds; compare PCM and WER.
E009 B r32 all 8GPU: fresh DreamOn/CosyVoice weights plus new LoRA, 20 epochs on cached LibriTTS 100+360+500; exp/route_b_r32_trainall_8gpu_20260924.
