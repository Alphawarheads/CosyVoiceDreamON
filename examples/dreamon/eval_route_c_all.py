import json,sys,subprocess,shutil
from pathlib import Path
ROOT=Path("/home/lize/CosyVoiceDreamON")
PY="/home/lize/miniconda3/envs/cosyvoice_dreamon/bin/python"
epoch=int(sys.argv[1])
ckpt=ROOT/f"exp/route_c_r16_trainall_20260921/epoch_{epoch}_whole.pt"
assert ckpt.is_file(), f"Checkpoint not ready: {ckpt}"
source=ROOT/"experiments/eval_batches/Q008_route_c_c100_e4e6_all_e0"
out=ROOT/f"experiments/eval_batches/C_all_epoch{epoch}_P01"
assert not (out/"launch.json").exists(), "Evaluation already launched"
out.mkdir(exist_ok=True)
for fn in ["eval.meta.lst","original_asr.json","run_generation.py","score.py"]:
 shutil.copy2(source/fn,out/fn)
job=next(j for j in json.loads((source/"jobs.json").read_text()) if j["name"].lower().startswith("call"))
oldname=job["name"]
manifest={"protocol":"P01_dev20_n64_t08", "reference_conditioning":True}
job={k:v for k,v in job.items() if k not in ("pid","gpu","exit_code")}
job["name"]=f"Call_epoch{epoch}"
job["output"]=str(out/job["name"])
args=job["args"]
for key,val in [("--meta",str(out/"eval.meta.lst")),("--adapter-checkpoint",str(ckpt)),("--output-dir",job["output"])]:
 args[args.index(key)+1]=val
(out/"jobs.json").write_text(json.dumps([job],indent=2))
meta=out/("metadata_"+job["name"]);meta.mkdir()
manifest.update(status="queued",checkpoint=str(ckpt),epoch=epoch,output=job["output"],route="C")
(meta/"manifest.json").write_text(json.dumps(manifest,indent=2))
pids={}
for fn in ["run_generation.py","score.py"]:
 with (out/(fn+".log")).open("w") as log:
  p=subprocess.Popen([PY,"-u",str(out/fn)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 pids[fn]=p.pid
(out/"launch.json").write_text(json.dumps(pids))
registry=ROOT/"experiments/EXPERIMENTS_BRIEF.md"
with registry.open("a") as f:
 f.write(f"\nC-all-e{epoch}: Route C r16 all-data checkpoint, P01 20-sentence generation and WER; {out}\n")
print(json.dumps(dict(output=str(out),pids=pids)))
