#!/usr/bin/env python3
"""Benchmark VLM and depth on the same annotated SUN RGB-D images."""
from __future__ import annotations
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np

ROOT=Path('/home/ubuntu/ai-assistant')
SUN=ROOT/'benchmarks/data/sunrgbd/extracted/SUNRGBD'
PROMPT=('Describe the most relevant visible object. Answer four uppercase fields joined by |: '
        'object PERSON, VEHICLE, POLE, STEP, DOOR, FURNITURE, HOLE, WALL, OBJECT, or CLEAR; '
        'where FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT, or NONE; distance NEAR, MID, FAR, or NONE; '
        'action NONE, SLOW, STOP, MOVE_LEFT, or MOVE_RIGHT. Output one line only.')
MAP={'chair':'furniture','table':'furniture','desk':'furniture','sofa':'furniture','cabinet':'furniture',
     'person':'person','people':'person','door':'door','vehicle':'vehicle','car':'vehicle','bed':'furniture'}

def samples(limit):
    out=[]
    for ann in sorted(SUN.rglob('annotation2D3D/index.json')):
        root=ann.parent.parent
        image=next(root.glob('image/*.jpg'),None)
        depth=next(root.glob('depth/*_abs.png'),None)
        if not image or not depth: continue
        data=json.load(open(ann,encoding='utf-8')); frame=data.get('frames',[{}])[0]
        objects=data.get('objects',[]); polys=frame.get('polygon',[])
        gt=[]
        for poly in polys:
            idx=poly.get('object')
            if not isinstance(idx,int) or idx>=len(objects): continue
            xs=poly.get('x',[]); ys=poly.get('y',[])
            if not xs or not ys: continue
            name=str(objects[idx].get('name','object')).lower()
            gt.append({'name':name,'type':MAP.get(name,'object'),'bbox':[max(0,min(xs)),max(0,min(ys)),min(640,max(xs)),min(480,max(ys))]})
        if gt: out.append((image,depth,gt))
        if len(out)>=limit: break
    return out

def parse(text):
    p=text.strip().splitlines()[0].strip().split('|') if text.strip() else []
    return {'type':p[0].lower(),'position':p[1].lower(),'distance':p[2].lower(),'action':p[3].lower()} if len(p)==4 else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=20); ap.add_argument('--output',required=True); a=ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor,AutoModelForDepthEstimation,AutoProcessor,AutoModelForImageTextToText
    import sys; sys.path.insert(0,str(ROOT/'scripts')); import mvp_config as cfg
    rows=samples(a.limit)
    dp=AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
    dm=AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',dtype=torch.float16).to('cuda:0').eval()
    vp=AutoProcessor.from_pretrained(cfg.VLM_MODEL)
    vm=AutoModelForImageTextToText.from_pretrained(cfg.VLM_MODEL,dtype=torch.float16,attn_implementation=cfg.ATTN_IMPL,device_map='cuda:0').eval()
    results=[]; type_ok=pos_ok=depth_n=0; depth_err=[]; fusion_err=[]
    for image_path,depth_path,gt in rows:
        image=Image.open(image_path).convert('RGB'); w,h=image.size
        msg=[{'role':'system','content':[{'type':'text','text':PROMPT}]},{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Frame:'}]}]
        x=vp.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to(vm.device); n=x['input_ids'].shape[-1]
        t=time.perf_counter()
        with torch.inference_mode(): out=vm.generate(**x,max_new_tokens=16,do_sample=False)
        vlm=parse(vp.decode(out[0][n:],skip_special_tokens=True)); latency=(time.perf_counter()-t)*1000
        dinput=dp(images=image,return_tensors='pt').to('cuda:0'); dinput['pixel_values']=dinput['pixel_values'].half()
        with torch.inference_mode(): dout=dm(**dinput)
        pred=np.squeeze(dp.post_process_depth_estimation(dout,target_sizes=[(h,w)])[0]['predicted_depth'].float().cpu().numpy())
        truth=np.array(Image.open(depth_path),dtype=np.float32); truth=truth/1000.0 if truth.max()>100 else truth
        depth_n+=1; med=np.median(pred[np.isfinite(pred)&(pred>0)]); gtmed=np.median(truth[truth>0]); depth_err.append(abs(med-gtmed)/max(gtmed,1e-3))
        if vlm:
            candidates=[g for g in gt if g['type']==vlm['type']] or gt
            if candidates:
                best=min(candidates,key=lambda g:abs(((g['bbox'][0]+g['bbox'][2])/2/w)-0.5))
                type_ok += vlm['type']==best['type']; cx=(best['bbox'][0]+best['bbox'][2])/(2*w); expected='left' if cx<.4 else 'right' if cx>.6 else 'front'; pos_ok += vlm['position'].startswith(expected)
                x1,y1,x2,y2=map(int,best['bbox']); box_truth=truth[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]; box_pred=pred[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]; box_truth=box_truth[box_truth>0]; box_pred=box_pred[np.isfinite(box_pred)&(box_pred>0)]
                if len(box_truth) and len(box_pred): fusion_err.append(abs(float(np.median(box_pred))-float(np.median(box_truth)))/max(float(np.median(box_truth)),1e-3))
        results.append({'image':str(image_path.relative_to(SUN)),'ground_truth':gt,'vlm':vlm,'depth_median':float(med),'gt_depth_median':float(gtmed),'vlm_ms':latency})
    report={'samples':len(results),'vlm_type_accuracy':type_ok/max(len(results),1),'vlm_position_accuracy':pos_ok/max(len(results),1),'depth_median_relative_error':float(np.mean(depth_err)) if depth_err else None,'fusion_bbox_depth_relative_error':float(np.mean(fusion_err)) if fusion_err else None,'fusion_note':'Same-frame semantic+depth consistency; no public hazard/action label','rows':results}
    Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
