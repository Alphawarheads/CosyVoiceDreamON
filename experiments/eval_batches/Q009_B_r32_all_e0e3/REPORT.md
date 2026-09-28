# Route B checkpoint audio evaluation

Route B r32 all-data checkpoints evaluated with the fixed P01 protocol. GPU mapping is in jobs.json; ASR used CPU.

| Group | Completed | WER incl. failed outputs | Near-silent RMS<1e-4 | Mean adjacent repeat |
|---|---:|---:|---:|---:|
| E009_epoch0 | 20/20 | 67.66% | 0 | 5.82% |
| E009_epoch1 | 20/20 | 63.76% | 0 | 5.38% |
| E009_epoch2 | 19/20 | 60.78% | 0 | 5.22% |
| E009_epoch3 | 20/20 | 55.28% | 0 | 4.52% |

Original recordings diagnostic WER: 0.92%.

20 fixed dev-clean sentences, one seed, n64 dynamic generation. CPU Whisper small.en diagnostic, not official full-test WER/MOS. Missing generation counts as deletions and coverage is reported separately.

Detailed ASR transcripts, token lengths, edit counts and per-utterance audio statistics: results.json.
No subjective listening or speaker-similarity score is claimed.