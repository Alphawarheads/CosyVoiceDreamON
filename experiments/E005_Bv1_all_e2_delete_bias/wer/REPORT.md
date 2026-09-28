# Route B checkpoint audio evaluation

Saved checkpoints evaluated while training continues. GPU mapping is in jobs.json; ASR used CPU.

| Group | Completed | WER incl. failed outputs | Near-silent RMS<1e-4 | Mean adjacent repeat |
|---|---:|---:|---:|---:|
| e2_delete_0p75 | 20/20 | 41.06% | 0 | 3.72% |
| e2_delete_0p5 | 20/20 | 45.64% | 0 | 4.08% |

Original recordings diagnostic WER: 0.92%.

20 fixed dev-clean sentences, one seed, n64 dynamic generation with DELETE action reweighting. CPU Whisper small.en diagnostic, not official full-test WER/MOS. Missing generation counts as deletions and coverage is reported separately.

Detailed ASR transcripts, token lengths, edit counts and per-utterance audio statistics: results.json.
No subjective listening or speaker-similarity score is claimed.