import os,sys,json,time,subprocess,traceback
from pathlib import Path
ROOT=Path("/home/lize/CosyVoiceDreamON")
OUT=Path(__file__).resolve().parent
PY="/home/lize/miniconda3/envs/cosyvoice_dreamon/bin/python"
def write(name,obj):
 p=OUT/name;t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(obj,indent=2));t.replace(p)
def launch(job,gpu):
 env=os.environ.copy()
 env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
 env["LD_LIBRARY_PATH"]="/usr/local/cuda-12.1/targets/x86_64-linux/lib:"+env.get("LD_LIBRARY_PATH","")
 wrapper="import torch,runpy; torch.cuda.set_per_process_memory_fraction(0.38,0); runpy.run_path('dreamon_generate.py',run_name='__main__')"
 log=(OUT/(job["name"]+".log")).open("w")
 p=subprocess.Popen([PY,"-u","-c",wrapper]+job["args"],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 job.update(pid=p.pid,gpu=gpu)
 return p,log
try:
 jobs=json.loads((OUT/"jobs.json").read_text())
 pending=list(jobs); active={}; completed=[]
 while pending or active:
  free=[]
  text=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],text=True)
  for line in text.splitlines():
   gpu,mem=(int(x.strip()) for x in line.split(","))
   if gpu in (4,5,6,7) and mem>=20*1024 and gpu not in active:free.append(gpu)
  while pending and free:
   gpu=free.pop(0);job=pending.pop(0);p,log=launch(job,gpu);active[gpu]=(job,p,log)
  write("generation_status.json",dict(status="running" if active or pending else "complete",
       pending=[j["name"] for j in pending],active={str(g):j["name"] for g,(j,p,l) in active.items()},completed=completed))
  (OUT/"jobs.json").write_text(json.dumps(jobs,indent=2))
  for gpu,(job,p,log) in list(active.items()):
   code=p.poll()
   if code is None:continue
   log.close();job["exit_code"]=code
   state=json.loads((OUT/job["name"]/"run.json").read_text()) if (OUT/job["name"]/"run.json").exists() else {}
   if code not in (0,2) or state.get("status") not in ("complete","complete_with_failures"):
    raise RuntimeError(job["name"]+" failed; inspect log")
   import shutil
   shutil.copy2(OUT/("metadata_"+job["name"])/"manifest.json",Path(job["output"])/"manifest.json")
   completed.append(job["name"]);del active[gpu]
  time.sleep(5)
 write("generation_status.json",dict(status="complete",completed=completed))
except Exception:
 write("generation_status.json",dict(status="failed",error=traceback.format_exc()))
 for job,p,log in active.values():
  if p.poll() is None:p.terminate()
 raise
