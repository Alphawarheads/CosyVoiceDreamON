import os,sys,json,random,gc,traceback,uuid
from pathlib import Path
ROOT=Path('/home/lize/CosyVoiceDreamON')
sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/E008_B_all_e2_backend_cross'
OUT.mkdir(parents=True,exist_ok=True)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import numpy as np,torch,soundfile as sf
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.dreamon_eval import read_eval_meta,utterance_seed
from dreamon_cosyvoice_test import decode_and_save
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.55,0)
META=ROOT/'experiments/protocols/P01_dev20_n64_t08/eval.meta.lst'
PRED=ROOT/'experiments/E002_Bv1_r16_trAll_s1986/eval/epoch_002_step_266001/P01_dev20_n64_t08'
def write(name,obj):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def seed(v):
 random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v)
def direct(cosy,cond,tokens,path):
 model=cosy.model;key=str(uuid.uuid4());model.hift_cache_dict[key]=None
 try:
  with torch.inference_mode():
   wave=model.token2wav(token=tokens.to(torch.int32),prompt_token=cond['flow_prompt_speech_token'],
    prompt_feat=cond['prompt_speech_feat'],embedding=cond['flow_embedding'],token_offset=0,uuid=key,finalize=True)
  arr=wave.detach().cpu().float().squeeze(0).numpy()
  sf.write(path,arr,cosy.sample_rate,subtype='PCM_16')
 finally: model.hift_cache_dict.pop(key,None)
 return arr
def pcm(path):return sf.read(path,dtype='int16')[0]
try:
 rows=read_eval_meta(META)
 for group in ['predicted','real']:
  for method in ['tts','direct']:(OUT/group/method/'wavs').mkdir(parents=True,exist_ok=True)
 cosy=CosyVoice2(str(ROOT/'CosyVoice2-0.5B'),fp16=False,load_jit=False,load_trt=False,load_vllm=False)
 # No DreamOn checkpoint is loaded: fixed input tokens isolate the codec backend.
 cosy.model.llm=None;gc.collect();torch.cuda.empty_cache()
 records=[]
 for i,row in enumerate(rows,1):
  cond=cosy.frontend.frontend_zero_shot(row.text,row.prompt_text,str(row.prompt_wav),cosy.sample_rate,'')
  actual,_=cosy.frontend._extract_speech_token(str(row.target_wav))
  token_by_group={'predicted':torch.load(PRED/'tokens'/(row.utt+'.pt'),map_location='cpu',weights_only=True).reshape(1,-1).int(),
                  'real':actual.cpu().reshape(1,-1).int()}
  for group,tok in token_by_group.items():
   paths={method:OUT/group/method/'wavs'/(row.utt+'.wav') for method in ['tts','direct']}
   sv=utterance_seed(1986,row.utt)
   seed(sv);decode_and_save(cosy,cond,tok,paths['tts'])
   seed(sv);direct(cosy,cond,tok,paths['direct'])
   a,b=pcm(paths['tts']),pcm(paths['direct'])
   z={'utt':row.utt,'group':group,'tokens':int(tok.numel()),'samples_tts':len(a),'samples_direct':len(b),
      'pcm_equal':bool(np.array_equal(a,b)),
      'max_abs_pcm_difference':int(np.max(np.abs(a.astype(np.int32)-b.astype(np.int32)))) if len(a)==len(b) else None}
   records.append(z)
  write('status.json',{'status':'decoding','completed':i,'expected':len(rows)})
  write('pcm_comparison.json',records)
 # ASR each path, same scoring convention as P01.
 import whisper
 from whisper.normalizers import EnglishTextNormalizer
 from difflib import SequenceMatcher
 norm=EnglishTextNormalizer();asr=whisper.load_model('small.en',device='cpu',download_root='/home/lize/.cache/whisper')
 detail=[];totals={}
 for group in ['predicted','real']:
  for method in ['tts','direct']:
   nref=errors=0
   for i,row in enumerate(rows,1):
    wav=OUT/group/method/'wavs'/(row.utt+'.wav')
    hyp=asr.transcribe(str(wav),language='en',temperature=0.,fp16=False,condition_on_previous_text=False)['text'].strip()
    ref=norm(row.text).split();pred=norm(hyp).split();ops=SequenceMatcher(None,ref,pred,autojunk=False).get_opcodes()
    error=sum(max(i2-i1,j2-j1) for tag,i1,i2,j1,j2 in ops if tag!='equal')
    nref+=len(ref);errors+=error
    detail.append({'group':group,'method':method,'utt':row.utt,'reference':row.text,'asr':hyp,'errors':error,'reference_words':len(ref)})
    write('status.json',{'status':'scoring','group':group,'method':method,'completed':i,'expected':len(rows)})
   totals[group+'_'+method]={'wer':errors/nref,'errors':errors,'reference_words':nref}
 write('results.json',{'status':'complete','summary':totals,'pcm_equal_counts':{g:sum(r['pcm_equal'] for r in records if r['group']==g) for g in ['predicted','real']},'pcm_comparison':records,'asr':detail,
   'note':'Same 20 P01 samples, fixed reference, same token arrays and per-utterance seeds. tts() already calls token2wav() internally. SequenceMatcher word edit counts used for this cross-check; PCM equality is the primary backend comparison.'})
 write('status.json',{'status':'complete'})
except Exception:
 write('status.json',{'status':'failed','error':traceback.format_exc()});raise
