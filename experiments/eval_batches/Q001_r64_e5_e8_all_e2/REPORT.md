# Route B checkpoint audio evaluation

Saved checkpoints evaluated while training continues. GPU mapping is in jobs.json; ASR used CPU.

| Group | Completed | WER incl. failed outputs | Near-silent RMS<1e-4 | Mean adjacent repeat |
|---|---:|---:|---:|---:|
| E004_epoch5 | 20/20 | 124.08% | 0 | 5.66% |
| E004_epoch6 | 20/20 | 89.91% | 0 | 3.30% |
| E004_epoch7 | 19/20 | 84.63% | 0 | 4.17% |
| E004_epoch8 | 18/20 | 87.84% | 0 | 5.27% |
| E002_epoch2 | 20/20 | 39.91% | 0 | 3.84% |

Original recordings diagnostic WER: 0.92%.

20 fixed dev-clean sentences, one seed, n64 dynamic generation. CPU Whisper small.en diagnostic, not official full-test WER/MOS. Missing generation counts as deletions and coverage is reported separately.

Detailed ASR transcripts, token lengths, edit counts and per-utterance audio statistics: results.json.
No subjective listening or speaker-similarity score is claimed.