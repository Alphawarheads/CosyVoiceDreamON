import os,json,time,subprocess,traceback
from pathlib import Path
ROOT=Path("/home/lize/CosyVoiceDreamON"); OUT=Path(__file__).resolve().parent; PY="/home/lize/miniconda3/envs/cosyvoice_dreamon/bin/python"
def wr(n,o): (OUT/n).write_text(json.dumps(o,indent=2))
jobs=json.loads((OUT/"jobs.json").read_text()); pending=list(jobs); active={}; done=[]
try:
 while pending or active:
  for gpu in [4,5,6,7]:
   if pending and gpu not in active:
    j=pending.pop(0); e=os.environ.copy(); e.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",DELETE_FACTOR=str(j.get("factor",1)),DIAGNOSTIC_MODE=j.get("mode","delete")); e["LD_LIBRARY_PATH"]="/usr/local/cuda-12.1/targets/x86_64-linux/lib:"+e.get("LD_LIBRARY_PATH","")
    log=(OUT/(j["name"]+".log")).open("w"); p=subprocess.Popen([PY,"-u",str(OUT/"inference_wrapper.py")]+j["args"],cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT); j.update(pid=p.pid,gpu=gpu); active[gpu]=(j,p,log)
  wr("generation_status.json",{"status":"running","pending":[j["name"] for j in pending],"active":{str(g):j["name"] for g,(j,p,l) in active.items()},"completed":done}); wr("jobs.json",jobs)
  for g,(j,p,l) in list(active.items()):
   c=p.poll()
   if c is not None: l.close(); j["exit_code"]=c; done.append(j["name"]); del active[g]
  time.sleep(5)
 wr("generation_status.json",{"status":"complete","completed":done})
except Exception:
 wr("generation_status.json",{"status":"failed","error":traceback.format_exc()}); raise
