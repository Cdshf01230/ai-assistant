#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, os, statistics, time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ubuntu/ai-assistant/models/hf")

def norm(s): return ' '.join(str(s).lower().strip().split())
def calc_wer(ref,hyp):
    r,h=norm(ref).split(),norm(hyp).split(); d=list(range(len(h)+1))
    for i in range(1,len(r)+1):
        prev,d[0]=d[0],i
        for j in range(1,len(h)+1): cur=d[j]; d[j]=min(d[j]+1,d[j-1]+1,prev+(r[i-1]!=h[j-1])); prev=cur
    return d[-1]/max(len(r),1)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='vinai/PhoWhisper-small'); p.add_argument('--limit',type=int,default=50); p.add_argument('--device',choices=('cpu','cuda'),default='cuda'); p.add_argument('--output',required=True); a=p.parse_args()
    import librosa, torch
    from datasets import Audio, load_dataset
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    ds=load_dataset('google/fleurs','vi_vn',split='validation',cache_dir='/home/ubuntu/ai-assistant/benchmarks/data/fleurs/hf').cast_column('audio',Audio(decode=False)); ds=ds.select(range(min(a.limit,len(ds))))
    dtype=torch.float16 if a.device=='cuda' else torch.float32; processor=AutoProcessor.from_pretrained(a.model); model=WhisperForConditionalGeneration.from_pretrained(a.model,dtype=dtype).to(a.device).eval(); sr=processor.feature_extractor.sampling_rate; rows=[]
    for row in ds:
        speech,source_sr=librosa.load(io.BytesIO(row['audio']['bytes']),sr=None,mono=True); speech=librosa.resample(speech,orig_sr=source_sr,target_sr=sr) if source_sr!=sr else speech; t=time.perf_counter(); x=processor(speech,sampling_rate=sr,return_tensors='pt').input_features.to(a.device,dtype=dtype)
        with torch.inference_mode(): ids=model.generate(x,language='vi',task='transcribe',max_new_tokens=96)
        text=processor.batch_decode(ids,skip_special_tokens=True)[0].strip(); ms=(time.perf_counter()-t)*1000; rows.append({'id':row['id'],'reference':row['transcription'],'prediction':text,'wer':calc_wer(row['transcription'],text),'latency_ms':round(ms),'duration_s':len(speech)/sr})
    report={'dataset':'google/fleurs vi_vn validation','model':a.model,'device':a.device,'samples':len(rows),'wer':statistics.mean(r['wer'] for r in rows),'median_latency_ms':statistics.median(r['latency_ms'] for r in rows),'p90_latency_ms':sorted(r['latency_ms'] for r in rows)[min(len(rows)-1,int(len(rows)*.9))],'rtf':statistics.mean(r['latency_ms']/1000/r['duration_s'] for r in rows),'rows':rows}; Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
