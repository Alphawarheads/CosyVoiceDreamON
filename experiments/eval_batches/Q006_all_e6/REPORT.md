# Route B checkpoint audio evaluation

Saved checkpoints evaluated while training continues. GPU mapping is in jobs.json; ASR used CPU.

| Group | Completed | WER incl. failed outputs | Near-silent RMS<1e-4 | Mean adjacent repeat |
|---|---:|---:|---:|---:|
| E002_epoch6 | 19/20 | 52.98% | 0 | 7.68% |

Original recordings diagnostic WER: 0.92%.

20 fixed dev-clean sentences, one seed, n64 dynamic generation. CPU Whisper small.en diagnostic, not official full-test WER/MOS. Missing generation counts as deletions and coverage is reported separately.

Detailed ASR transcripts, token lengths, edit counts and per-utterance audio statistics: results.json.
No subjective listening or speaker-similarity score is claimed.