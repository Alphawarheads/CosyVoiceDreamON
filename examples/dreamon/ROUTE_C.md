# Route C: reference-conditioned dynamic speech tokens

Input order matches the CosyVoice prompt convention using the DreamOn tokenizer:
`BOS, Ref text, Target text, TASK, Ref speech tokens, Target canvas, EOS`.
The DreamOn backbone retains bidirectional attention and its shifted hidden-state
alignment. Only target-canvas positions receive FILL / EXPAND / DELETE supervision.
The reference is never masked, edited, decoded as target output, or included in loss.
There is no duration predictor; initial MASK count is a generation starting point.

`cosyvoice/llm/dreamon_route_c.py` is an independent copy of the B implementation,
with reference conditioning added. B source and checkpoints remain unchanged.
C uses checkpoint format 7 / route_c; B uses 6 / route_b. Cross-loading is rejected.
Both speech and action heads see the reference through the backbone hidden states.
LoRA and projections/edit head training, factorized loss, corruption, and sampling
are retained from B to isolate the effect of reference conditioning.

## Data

`tools/prepare_dreamon_route_c.py` reads the existing B token cache (no extraction).
It pairs each target with a different utterance/transcript from the same speaker,
within its own split. Pairings are deterministic and fixed across epochs. Full
reference utterances, up to 250 speech tokens, are selected without cropping text
or speech. The shared 2048-token budget includes BOTH texts, reference speech,
target speech, three special tokens, plus 64 slots reserved during preparation.
Targets with no compatible reference are explicitly counted and excluded. No
cross-split target, reference, or speaker overlap is allowed. The reference pool
retains source text/token identity for validation. Pairing does not alter B caches.

Prepared train100: 33,224 targets (1 excluded from 33,225); dev: 10,340 targets.
Comparisons against B should use matched retained targets and the same evaluation
references; a future experiment can vary the fixed reference selection separately.

## Run and checkpoints

See ROUTE_C_COMMANDS.txt or the appended Route C section in root cmd.txt.
Fresh r32 training example: 10 full epochs, no automatic early stopping.
The optimizer is not resumed from B. Frozen pretrained DreamOn + CosyVoice speech
modules are loaded, with fresh LoRA, projections, and edit head.
Full checkpoints include the DreamOn backbone and trainable modules, requiring
substantial disk space. Original frozen CosyVoice components remain external.
The C training manifest includes reference pairing settings and source hashes.
Generation automatically dispatches to C from checkpoint metadata and obtains
reference text/audio from the existing evaluation meta format.

## Validation and scope

24 A/B/C CPU tests passed, including reference influence, target-only losses,
reference preservation under edits, gradients, context limits, split isolation,
and checkpoint roundtrips. A real-weight r32 GPU smoke test performs one update
and two edit steps; its report is outputs/route_c_implementation_checks/real_weight_smoke.json.
This is implementation validation, not evidence of speech quality improvement.
No long-running C training job is started by this implementation task.
