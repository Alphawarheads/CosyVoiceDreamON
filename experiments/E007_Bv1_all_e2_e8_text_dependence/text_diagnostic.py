import os,sys,json,random,hashlib,traceback
from pathlib import Path
ROOT=Path("/home/lize/CosyVoiceDreamON");OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from cosyvoice.llm.dreamon_route_b import build_route_b_model
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.6,0)
def save(name,obj):
 p=OUT/name;t=p.with_suffix(".tmp");t.write_text(json.dumps(obj,indent=2));t.replace(p)
rows=json.loads((OUT/"diagnostic_rows.json").read_text())
try:
 results=[]
 for epoch in [2,8]:
  model=build_route_b_model(ROOT/"DreamOn-v0-7B",ROOT/"CosyVoice2-0.5B")
  model.bridge.load_adapter_checkpoint(ROOT/f"exp/route_b_r16_all_resume_20260915_213200/epoch_{epoch}_whole.pt")
  model.cuda().eval()
  for i,r in enumerate(rows):
   speech=torch.tensor([r["speech"]],device="cuda")
   original=torch.tensor([r["text_ids"]],device="cuda",dtype=torch.long)
   wrong=torch.tensor([rows[(i+1)%len(rows)]["text_ids"]],device="cuda",dtype=torch.long)
   # Null text keeps BOS/task/EOS; empty text_ids are valid in forward_features.
   empty=original[:,:0]
   for ratio in [0.5,1.0]:
    seed=int.from_bytes(hashlib.sha256((r["utt"]+str(ratio)).encode()).digest()[:8],"little")
    positions=random.Random(seed).sample(range(speech.numel()),max(1,round(speech.numel()*ratio)))
    mask=torch.zeros_like(speech,dtype=torch.bool);mask[0,positions]=True
    canvas=speech.clone();canvas[mask]=-1
    for condition,ids in [("correct",original),("mismatched",wrong),("empty",empty)]:
     with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
      s,a=model.bridge.edit_forward(ids,canvas)
     logits=s[mask].float();labels=speech[mask]
     result=dict(epoch=epoch,utt=r["utt"],mask_ratio=ratio,condition=condition,
        count=int(labels.numel()),ce_sum=float(F.cross_entropy(logits,labels,reduction="sum")),
        correct=int(logits.argmax(-1).eq(labels).sum()),fill_probability=float(a[mask].float().softmax(-1)[:,0].mean()))
     results.append(result)
   save("text_status.json",dict(status="running",epoch=epoch,completed_utterances=i+1))
  del model
  torch.cuda.empty_cache()
 summary={}
 for epoch in [2,8]:
  for ratio in [.5,1.]:
   for condition in ["correct","mismatched","empty"]:
    part=[r for r in results if r["epoch"]==epoch and r["mask_ratio"]==ratio and r["condition"]==condition]
    n=sum(r["count"] for r in part)
    summary[f"e{epoch}_mask{ratio}_{condition}"]=dict(token_ce=sum(r["ce_sum"] for r in part)/n,accuracy=sum(r["correct"] for r in part)/n)
 save("results.json",dict(status="complete",summary=summary,rows=results,note="Oracle fixed-length teacher-forced masked-token diagnostic, not generated WER. Wrong/empty text changes text length and position; not a pure causal ablation."))
 save("text_status.json",dict(status="complete"))
except Exception:
 save("text_status.json",dict(status="failed",error=traceback.format_exc()));raise
