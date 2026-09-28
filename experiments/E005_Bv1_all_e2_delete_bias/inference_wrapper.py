import os,sys,json,inspect,textwrap,runpy
from pathlib import Path
ROOT=Path("/home/lize/CosyVoiceDreamON")
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
import cosyvoice.llm.dreamon_route_b as rb
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.38,0)
factor=float(os.environ.get("DELETE_FACTOR","1"))
mode=os.environ.get("DIAGNOSTIC_MODE","delete")
assert 0<factor<=1 and mode in ("delete","oracle")
source=textwrap.dedent(inspect.getsource(rb.DynamicSpeechLM.generate_dynamic))
# Position ranking uses unmodified probabilities; modify only sampling distribution.
needle="            before = canvas.shape[1]"
# dedented source has 8 spaces inside loop
needle="        act = (int(torch.multinomial(ap[j], 1, generator=rng)) if action_temperature\n               else int(a[j].argmax()))"
assert needle in source
replacement="""        sample_probs = ap[j].clone()
        sample_probs[DELETE] *= DELETE_FACTOR
        sample_probs /= sample_probs.sum()
        act = (int(torch.multinomial(sample_probs, 1, generator=rng)) if action_temperature
               else int(sample_probs.argmax()))"""
source=source.replace(needle,replacement)
source=source.replace('action_probability=float(ap[j,act])','action_probability=float(sample_probs[act])')
source=source.replace('self.eval()','self.eval()\n    if DIAGNOSTIC_MODE == "oracle":\n        initial_masks = ORACLE_LENGTHS[text]',1)
source=source.replace('        before = canvas.shape[1]','        if DIAGNOSTIC_MODE == "oracle":\n            act = FILL\n        before = canvas.shape[1]',1)
source=source.replace('target_audio_used=False,','target_audio_used=(DIAGNOSTIC_MODE == "oracle"),\n                  target_length_used=(DIAGNOSTIC_MODE == "oracle"), diagnostic_mode=DIAGNOSTIC_MODE,\n                  delete_probability_factor=DELETE_FACTOR,',1)
scope=dict(vars(rb),DELETE_FACTOR=factor,DIAGNOSTIC_MODE=mode,ORACLE_LENGTHS=json.loads((OUT/"oracle_lengths.json").read_text()))
exec(compile(source,str(OUT/"isolated_generate_dynamic.py"),"exec"),scope)
rb.DynamicSpeechLM.generate_dynamic=scope["generate_dynamic"]
out=Path(sys.argv[sys.argv.index("--output-dir")+1])
try:
 runpy.run_path(str(ROOT/"dreamon_generate.py"),run_name="__main__")
finally:
 f=out/"run.json"
 if f.exists():
  d=json.loads(f.read_text());d.update(diagnostic_mode=mode,delete_probability_factor=factor,target_length_used=mode=="oracle")
  if mode=="oracle":
   d["length_mode"]="oracle_fixed_length_diagnostic"
   d["note"]="Oracle target token count only; no target token values supplied; action-entropy ordering retained and FILL forced."
  f.write_text(json.dumps(d,indent=2))
