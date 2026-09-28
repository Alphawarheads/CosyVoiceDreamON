from pathlib import Path
import subprocess,sys,json,traceback,time
OUT=Path(__file__).resolve().parent
def write(obj):
 p=OUT/"eval_status.json";t=p.with_suffix(".tmp");t.write_text(json.dumps(obj,indent=2));t.replace(p)
try:
 children={}
 for script in ("run_generation.py","score.py"):
  log=(OUT/(script+".log")).open("w")
  p=subprocess.Popen([sys.executable,"-u",str(OUT/script)],cwd="/home/lize/CosyVoiceDreamON",stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  children[script]=(p,log)
 write(dict(status="running",pids={k:v[0].pid for k,v in children.items()}))
 while any(p.poll() is None for p,l in children.values()):
  bad={k:p.returncode for k,(p,l) in children.items() if p.poll() not in (None,0)}
  if bad:raise RuntimeError("child failed "+repr(bad))
  time.sleep(10)
 result=json.loads((OUT/"results.json").read_text())
 assert result["status"]=="complete"
 write(dict(status="complete",results=str(OUT/"results.json"),report=str(OUT/"REPORT.md"),audio_root=str(OUT)))
except Exception:
 write(dict(status="failed",error=traceback.format_exc()))
 for p,l in globals().get("children",{}).values():
  if p.poll() is None:p.terminate()
 raise
finally:
 for p,l in globals().get("children",{}).values():l.close()

index=OUT.parents[1]/"INDEX.md"
with index.open("a") as f:
 f.write("\nQ004 finished: "+str(OUT/"REPORT.md")+"\n")
