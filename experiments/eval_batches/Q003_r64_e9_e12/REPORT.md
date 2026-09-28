# Route B checkpoint audio evaluation

Saved checkpoints evaluated while training continues. GPU mapping is in jobs.json; ASR used CPU.

| Group | Completed | WER incl. failed outputs | Near-silent RMS<1e-4 | Mean adjacent repeat |
|---|---:|---:|---:|---:|
| E004_epoch9 | 20/20 | 88.30% | 0 | 6.15% |
| E004_epoch10 | 17/20 | 88.99% | 0 | 4.46% |
| E004_epoch11 | 20/20 | 115.83% | 0 | 3.70% |
| E004_epoch12 | 20/20 | 81.88% | 0 | 6.22% |

Original recordings diagnostic WER: 0.92%.

20 fixed dev-clean sentences, one seed, n64 dynamic generation. CPU Whisper small.en diagnostic, not official full-test WER/MOS. Missing generation counts as deletions and coverage is reported separately.

Detailed ASR transcripts, token lengths, edit counts and per-utterance audio statistics: results.json.
No subjective listening or speaker-similarity score is claimed.