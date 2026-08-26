#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, statistics, time
from pathlib import Path

def norm(s): return ' '.join(str(s).lower().strip(' .!?,').split())
def wer(ref,hyp):
 r,h=norm(ref).split(),norm(hyp).split(); d=list(range(len(h)+1))
 for i in range(1,len(r)+1):
  prev,d[0]=d[0],i
  for j in range(1,len(h)+1): cur=d[j]; d[j]=min(d[j]+1,d[j-1]+1,prev+(r[i-1]!=h[j-1])); prev=cur
 return d[-1]/max(len(r),1)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']=''
 from vieneu import Vieneu
 import soundfile as sf
 tts=Vieneu(); sr=getattr(tts,'sample_rate',48000); texts=['Dừng lại.','Rẽ trái.','Có vật cản phía trước bên phải.','Đưa tôi đến Vincom.']
 voices=[v[1] for v in tts.list_preset_voices()]
 rows=[]
 for voice in voices:
  for text in texts:
   try:
    start=time.perf_counter(); audio=tts.infer(text,voice=voice); ms=(time.perf_counter()-start)*1000; path=f'/tmp/tts_{voice}_{len(rows)}.wav'; tts.save(audio,path); rows.append({'voice':voice,'reference':text,'latency_ms':round(ms),'wav':path})
   except Exception as exc: rows.append({'voice':voice,'reference':text,'error':repr(exc)})
 # The audio files are scored by the existing STT command in a follow-up step.
 report={'model':'pnnbao-ump/VieNeu-TTS-v3-Turbo','voices':voices,'samples':len(rows),'rows':rows,'note':'Audio generated for subsequent STT round-trip scoring'}
 Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'voices':len(voices),'samples':len(rows),'median_latency_ms':statistics.median([r['latency_ms'] for r in rows if 'latency_ms' in r])},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
