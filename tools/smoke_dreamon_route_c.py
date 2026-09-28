#!/usr/bin/env python3
"""One real-weight update and two edit steps. Does not start a training run."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from cosyvoice.llm.dreamon_route_c import build_route_c_model, EditGenerationError
from cosyvoice.bin.train_route_c import batch

p=argparse.ArgumentParser()
p.add_argument("--cache",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
start=time.monotonic()
cache=json.loads(a.cache.read_text())
# Nontrivial target, with an independently recorded same-speaker reference.
r=next(r for r in cache["train"] if 100 <= len(r["speech"]) <= 150)
m=build_route_c_model(root/"DreamOn-v0-7B",root/"CosyVoice2-0.5B",lora_rank=32,lora_alpha=64).cuda()
opt=torch.optim.AdamW(m.optimizer_parameter_groups(),lr=1e-4)
m.train()
with torch.autocast("cuda",dtype=torch.bfloat16):
    loss=m(batch(r),0)
assert all(torch.isfinite(v) for v in loss.values())
loss["loss"].backward()
params=[p for p in m.parameters() if p.requires_grad]
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
norm=torch.nn.utils.clip_grad_norm_(params,1.)
opt.step();opt.zero_grad(set_to_none=True)
m.eval()
with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
    ids=m.bridge.encode_text(r["text"],r["ref_text"])
    canvas=torch.full((1,16),-1,device="cuda",dtype=torch.long)
    ref=torch.tensor([r["ref_speech"]],device="cuda",dtype=torch.long)
    s,act=m.bridge.edit_forward(ids,canvas,ref)
    s2,act2=m.bridge.edit_forward(ids,canvas,ref.roll(1,dims=1))
    delta=float((s.float()-s2.float()).abs().mean())
    assert delta>0
    try:
        tokens,gen=m.bridge.generate_dynamic(r["text"],prompt_text=r["ref_text"],
            prompt_speech_tokens=ref,initial_masks=16,maximum=32,max_steps=2,
            temperature=0,action_temperature=.8)
    except EditGenerationError as exc:
        gen=exc.report
    assert gen.get("failure_reason") in (None,"edit_step_budget_exhausted","empty_output","canvas_cap_reached")
report=dict(status="passed",test="one_update_two_edit_steps_not_audio_quality",utt=r["utt"],ref_utt=r["ref_utt"],
            target_tokens=len(r["speech"]),reference_tokens=len(r["ref_speech"]),
            loss=float(loss["loss"]),grad_norm=float(norm),reference_change_logit_mean_abs=delta,
            trainable_parameters=sum(p.numel() for p in params),
            max_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            generation=gen,elapsed_seconds=time.monotonic()-start)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="generation"},indent=2))
