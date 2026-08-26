#!/usr/bin/env python3
"""Indoor demo pipeline: VLM semantics + relative depth + temporal decisions."""
from __future__ import annotations
import argparse, io, json, os, statistics, sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ubuntu/ai-assistant/models/hf")

ROOT=Path('/home/ubuntu/ai-assistant')
FRAMES=ROOT/'assets/frames'
VLM_MODEL='Qwen/Qwen3-VL-4B-Instruct'
DEPTH_MODEL='depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf'
LONG_EDGE=640; JPEG_QUALITY=80; CONFIRM=2; CLEAR=3

@dataclass
class TemporalState:
    active=False; positive=0; clear=0; last_key=''

def parse_pipe(text):
    line=text.strip().splitlines()[0].strip() if text.strip() else ''
    p=[x.strip().lower() for x in line.split('|')]
    if len(p)!=4 or p[0] not in {'person','vehicle','pole','step','door','furniture','hole','wall','object','clear'} or p[1] not in {'front','front_left','front_right','left','right','none'} or p[2] not in {'near','mid','far','none'} or p[3] not in {'none','slow','stop','move_left','move_right'}: return None
    return {'hazard':p[0]!='clear','type':p[0],'position':p[1],'distance':p[2],'action':p[3]}

def prep(image):
    from PIL import Image
    image=image.convert('RGB'); w,h=image.size
    if max(w,h)>LONG_EDGE:
        s=LONG_EDGE/max(w,h); image=image.resize((round(w*s),round(h*s)),Image.LANCZOS)
    b=io.BytesIO(); image.save(b,format='JPEG',quality=JPEG_QUALITY); b.seek(0)
    return Image.open(b).convert('RGB')

def relative_depth_score(depth, path_masks, free_space):
    import numpy as np
    band,col=path_masks(*depth.shape); free,_=free_space(depth,band,col,10)
    finite=free[np.isfinite(free)]
    if not len(finite): return {'valid':False,'near_score':0.0,'free_rank':None}
    shortest=float(np.min(finite)); farthest=float(np.max(finite))
    free_rank=float(shortest/farthest) if farthest else 0.0
    return {'valid':True,'near_score':float(max(0,min(1,1.0-free_rank))), 'free_rank':free_rank, 'free_per_col':[None if x!=x else float(x) for x in free]}

def decide(vlm, depth, state):
    valid_depth=bool(depth and depth.get('valid'))
    vlm_hazard=bool(vlm and vlm.get('hazard'))
    depth_hazard=valid_depth and depth.get('near_score',0)>=0.5
    key=(vlm.get('type'),vlm.get('position')) if vlm_hazard else ('unknown','front')
    evidence=vlm_hazard or depth_hazard
    if evidence:
        state.positive += 1; state.clear=0; state.last_key='|'.join(key)
    else:
        state.clear += 1; state.positive=0
    if not state.active and state.positive>=CONFIRM: state.active=True
    if state.active and state.clear>=CLEAR: state.active=False
    if not state.active: action='NONE'
    elif vlm_hazard: action=vlm.get('action') if vlm.get('action')!='none' else 'SLOW'
    else: action='STOP'
    return {'warn':state.active,'action':action,'type':vlm.get('type') if vlm else None,'position':vlm.get('position') if vlm else 'front','fallback':'depth_only' if depth_hazard and not vlm_hazard else ('vlm_only' if vlm_hazard and not valid_depth else None),'evidence':{'vlm_hazard':vlm_hazard,'depth_valid':valid_depth,'depth_hazard':depth_hazard},'streak':state.positive}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--frames-dir',default=str(FRAMES)); ap.add_argument('--limit',type=int,default=10); ap.add_argument('--output',default=str(ROOT/'logs/indoor_demo_pipeline.json')); a=ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor,AutoModelForDepthEstimation,AutoProcessor,AutoModelForImageTextToText
    sys.path.insert(0,str(ROOT/'scripts')); import depth_path as dp
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    vp=AutoProcessor.from_pretrained(VLM_MODEL); vm=AutoModelForImageTextToText.from_pretrained(VLM_MODEL,dtype=torch.float16,attn_implementation='sdpa',device_map='cuda:0').eval()
    dp_proc=AutoImageProcessor.from_pretrained(DEPTH_MODEL); dm=AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL,dtype=torch.float16).to('cuda:0').eval()
    torch.cuda.synchronize(); vram={'allocated_gib':round(torch.cuda.memory_allocated()/1024**3,3),'reserved_gib':round(torch.cuda.memory_reserved()/1024**3,3),'peak_reserved_gib':round(torch.cuda.max_memory_reserved()/1024**3,3)}
    prompt=('You are an indoor mobility-aid vision module. Describe the most relevant object in or near the walking path. Do not decide whether to warn. Answer exactly OBJECT|WHERE|DISTANCE|ACTION using object PERSON, VEHICLE, POLE, STEP, DOOR, FURNITURE, HOLE, WALL, OBJECT, CLEAR; where FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT, NONE; distance NEAR, MID, FAR, NONE; action NONE, SLOW, STOP, MOVE_LEFT, MOVE_RIGHT.')
    files=sorted(Path(a.frames_dir).glob('*.jpg'))[:a.limit]; state=TemporalState(); rows=[]
    for path in files:
        image=prep(Image.open(path)); msg=[{'role':'system','content':[{'type':'text','text':prompt}]},{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Frame:'}]}]
        x=vp.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to(vm.device); n=x['input_ids'].shape[-1]
        with torch.inference_mode(): out=vm.generate(**x,max_new_tokens=16,do_sample=False)
        raw=vp.decode(out[0][n:],skip_special_tokens=True); parsed=parse_pipe(raw)
        try:
            z=dp_proc(images=image,return_tensors='pt').to('cuda:0'); z['pixel_values']=z['pixel_values'].half()
            with torch.inference_mode(): dout=dm(**z)
            depth=dp_proc.post_process_depth_estimation(dout,target_sizes=[(image.height,image.width)])[0]['predicted_depth'].float().cpu().numpy().squeeze()
            dscore=relative_depth_score(depth,dp.path_masks,dp.free_space)
        except Exception as exc:
            dscore={'valid':False,'near_score':0.0,'error':repr(exc)}
        decision=decide(parsed,dscore,state); rows.append({'frame':path.name,'vlm_raw':raw,'vlm':parsed,'depth':dscore,'decision':decision})
    report={'profile':{'name':'indoor','vlm_model':VLM_MODEL,'depth_model':DEPTH_MODEL,'long_edge':LONG_EDGE,'depth_mode':'relative_free_space_rank','confirm_frames':CONFIRM,'clear_frames':CLEAR},'vram':vram,'frames':rows}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'profile':report['profile'],'vram':report['vram'],'frames':len(rows),'warnings':sum(r['decision']['warn'] for r in rows)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
