
import os,sys,json,time,traceback,hashlib
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
ROOT=Path("/home/lize/CosyVoiceDreamON")
OUT=Path(__file__).resolve().parent
def write(name,obj):
 p=OUT/name;t=p.with_suffix(p.suffix+".tmp")
 t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def counts(ref,hyp):
 a,b=ref.split(),hyp.split()
 prev=[(j,0,0,j) for j in range(len(b)+1)]
 for i,x in enumerate(a,1):
  row=[(i,0,i,0)]
  for j,y in enumerate(b,1):
   if x==y:row.append(prev[j-1])
   else:
    c,s,d,k=prev[j-1];sub=(c+1,s+1,d,k)
    c,s,d,k=prev[j];delete=(c+1,s,d+1,k)
    c,s,d,k=row[-1];insert=(c+1,s,d,k+1)
    row.append(min((sub,delete,insert),key=lambda z:z[0]))
  prev=row
 c,s,d,k=prev[-1]
 return dict(wer=c/max(1,len(a)),substitutions=s,deletions=d,insertions=k,reference_words=len(a))
assert counts("one two","one three")["wer"]==.5
assert counts("one two","")["deletions"]==2
assert counts("one","one two")["insertions"]==1
try:
 import torch,numpy as np,soundfile as sf,whisper
 from whisper.normalizers import EnglishTextNormalizer
 torch.set_num_threads(4)
 norm=EnglishTextNormalizer()
 model=whisper.load_model("small.en",device="cpu",download_root="/home/lize/.cache/whisper")
 meta=[line.split("|") for line in (OUT/"eval.meta.lst").read_text().splitlines() if line.strip()]
 def score(path,text):
  actual=model.transcribe(str(path),language="en",temperature=0.,fp16=False,condition_on_previous_text=False)["text"].strip()
  return dict(transcript=actual,**counts(norm(text),norm(actual)))
 originals=json.loads((OUT/"original_asr.json").read_text())
 assert [r["utt"] for r in originals]==[r[0] for r in meta]
 for row in []:
  originals.append(dict(utt=row[0],text=row[3],**score(row[4],row[3])))
  write("original_asr.json",originals)
  write("scoring_status.json",dict(status="original_asr",completed=len(originals),expected=len(meta)))
 results=[]
 jobs=json.loads((OUT/"jobs.json").read_text())
 for job in jobs:
  for row in meta[:job["limit"]]:
   directory=OUT/job["name"]
   wav=directory/"wavs"/(row[0]+".wav")
   failure=directory/"tokens"/(row[0]+".failure.json")
   tokenfile=directory/"tokens"/(row[0]+".pt")
   while not ((wav.exists() and tokenfile.exists()) or failure.exists()):
    state=json.loads((directory/"run.json").read_text()) if (directory/"run.json").exists() else {}
    gen=json.loads((OUT/"generation_status.json").read_text()) if (OUT/"generation_status.json").exists() else {}
    if state.get("status") in ("failed","complete","complete_with_failures") or gen.get("status")=="failed":
     raise RuntimeError("Generation ended without expected output: "+job["name"]+"/"+row[0])
    time.sleep(3)
   item=dict(group=job["name"],utt=row[0],text=row[3],initial_masks=job["initial_masks"])
   if failure.exists():
    fail=json.loads(failure.read_text())
    item.update(status="generation_failed",failure_reason=fail["failure_reason"],transcript="",
                **counts(norm(row[3]),""))
   else:
    y,sr=sf.read(wav,dtype="float64")
    token=torch.load(tokenfile,map_location="cpu",weights_only=True).flatten()
    if not len(y) or not np.isfinite(y).all():raise RuntimeError("Invalid WAV: "+str(wav))
    values,c=token.unique(return_counts=True)
    trajectory=json.loads((directory/"tokens"/(row[0]+".trajectory.json")).read_text())
    original_info=sf.info(row[4])
    item.update(status="complete",wav=str(wav),duration_seconds=len(y)/sr,
                reference_duration_seconds=original_info.duration,
                rms=float(np.sqrt(np.mean(y*y))),peak=float(np.max(np.abs(y))),
                speech_tokens=len(token),unique_tokens=len(values),dominant_fraction=float(c.max()/len(token)),
                adjacent_repeat_ratio=float((token[1:]==token[:-1]).float().mean()) if len(token)>1 else 0.,
                expand_count=trajectory["expand_count"],delete_count=trajectory["delete_count"],
                forward_passes=len(trajectory["steps"]),**score(wav,row[3]))
   results.append(item)
   write("results.partial.json",dict(originals=originals,results=results))
   write("scoring_status.json",dict(status="scoring",completed=len(results),expected=sum(j["limit"] for j in jobs),
                                  current_group=job["name"],current_utt=row[0]))
 summaries={}
 for job in jobs:
  rows=[r for r in results if r["group"]==job["name"]]
  ok=[r for r in rows if r["status"]=="complete"]
  errors=lambda xs:sum(r["substitutions"]+r["deletions"]+r["insertions"] for r in xs)
  words=lambda xs:sum(r["reference_words"] for r in xs)
  summaries[job["name"]]=dict(attempted=len(rows),completed=len(ok),failed=len(rows)-len(ok),
    corpus_wer_with_failures_as_deletions=errors(rows)/max(1,words(rows)),
    successful_audio_corpus_wer=errors(ok)/max(1,words(ok)) if ok else None,
    below_rms_1e_4=sum(r["rms"]<1e-4 for r in ok),
    mean_repeat=float(np.mean([r["adjacent_repeat_ratio"] for r in ok])) if ok else None,
    mean_duration_ratio=float(np.mean([r["duration_seconds"]/r["reference_duration_seconds"] for r in ok])) if ok else None)
 original_wer=sum(r["substitutions"]+r["deletions"]+r["insertions"] for r in originals)/sum(r["reference_words"] for r in originals)
 full=dict(status="complete",summary=summaries,original_corpus_wer=original_wer,originals=originals,results=results,
  limitations="20 fixed dev-clean sentences, one seed, n64 dynamic generation with DELETE action reweighting. CPU Whisper small.en diagnostic, not official full-test WER/MOS. Missing generation counts as deletions and coverage is reported separately.")
 write("results.json",full)
 write("scoring_status.json",dict(status="complete",completed=len(results)))
 lines=["# Route B checkpoint audio evaluation","",
        "Saved checkpoints evaluated while training continues. GPU mapping is in jobs.json; ASR used CPU.",
        "", "| Group | Completed | WER incl. failed outputs | Near-silent RMS<1e-4 | Mean adjacent repeat |",
        "|---|---:|---:|---:|---:|"]
 for name,s in summaries.items():
  repeat="n/a" if s["mean_repeat"] is None else f'{s["mean_repeat"]:.2%}'
  lines.append(f'| {name} | {s["completed"]}/{s["attempted"]} | {s["corpus_wer_with_failures_as_deletions"]:.2%} | {s["below_rms_1e_4"]} | {repeat} |')
 lines+=["",f"Original recordings diagnostic WER: {original_wer:.2%}.","",full["limitations"],
         "", "Detailed ASR transcripts, token lengths, edit counts and per-utterance audio statistics: results.json.",
         "No subjective listening or speaker-similarity score is claimed."]
 (OUT/"REPORT.md").write_text("\n".join(lines))
except Exception:
 write("scoring_status.json",dict(status="failed",error=traceback.format_exc()))
 raise
