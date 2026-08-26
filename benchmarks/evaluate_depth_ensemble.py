#!/usr/bin/env python3
"""Evaluate single and scale-normalized depth ensembles on NYU/DIODE."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
os.environ.setdefault("HF_HOME", "/home/ubuntu/ai-assistant/models/hf")
ROOT=Path('/home/ubuntu/ai-assistant'); DIODE=ROOT/'benchmarks/data/diode/val'; NYU=ROOT/'benchmarks/data/nyu/nyu_depth_v2_labeled.mat'

def metric(pred, truth):
    m=np.isfinite(pred)&np.isfinite(truth)&(pred>0)&(truth>0); p,t=pred[m],truth[m]
    scale=np.median(t/p); p=p*scale; r=np.maximum(p/t,t/p)
    return {'abs_rel':float(np.mean(np.abs(p-t)/t)),'rmse':float(np.sqrt(np.mean((p-t)**2))),'delta1':float(np.mean(r<1.25))}

def predict(proc, model, image, shape):
    import torch
    x=proc(images=image,return_tensors='pt').to('cuda:0'); x['pixel_values']=x['pixel_values'].half()
    with torch.inference_mode(): y=model(**x)
    return np.squeeze(proc.post_process_depth_estimation(y,target_sizes=[shape])[0]['predicted_depth'].float().cpu().numpy())

def images(limit, domain=None):
    from PIL import Image
    out=[]; base=DIODE/domain if domain else DIODE
    for dp in sorted(base.rglob('*_depth.npy'))[:limit]:
        rgb=dp.with_name(dp.name.replace('_depth.npy','.png')); mask=dp.with_name(dp.name.replace('_depth.npy','_depth_mask.npy'))
        if rgb.exists() and mask.exists(): out.append((Image.open(rgb).convert('RGB'),np.squeeze(np.load(dp).astype('float32')),np.squeeze(np.load(mask).astype(bool))))
    return out

def evaluate(proc_models, items):
    predictions=[[] for _ in proc_models]
    for index,(image,truth,valid) in enumerate(items):
        for j,(proc,model) in enumerate(proc_models): predictions[j].append((predict(proc,model,image,truth.shape),truth,valid))
    result={}
    for j,(_,_) in enumerate(proc_models): result[f'model_{j+1}']=mean([metric(p[valid],t[valid]) for p,t,valid in predictions[j]])
    if len(predictions)>1:
        ens=[]
        for i in range(len(items)):
            maps=[p[i][0] for p in predictions]
            norm=[x/max(float(np.median(x[x>0])),1e-6) for x in maps]
            ens.append(metric(np.mean(norm,axis=0)[predictions[0][i][2]],predictions[0][i][1][predictions[0][i][2]]))
        result[f'ensemble_{len(predictions)}']=mean(ens)
    return result

def mean(rows): return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models',nargs='+',required=True); ap.add_argument('--limit',type=int,default=50); ap.add_argument('--output',required=True); a=ap.parse_args()
    import torch
    from transformers import AutoImageProcessor,AutoModelForDepthEstimation
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    all_results={}
    for domain in ('indoors','outdoor'):
        items=images(a.limit,domain); all_results[domain]={}
        for model_name in a.models:
            proc=AutoImageProcessor.from_pretrained(model_name); model=AutoModelForDepthEstimation.from_pretrained(model_name,dtype=torch.float16).to('cuda:0').eval()
            single=evaluate([(proc,model)],items); all_results[domain][model_name]=single['model_1']; del model
            torch.cuda.empty_cache()
        loaded=[]
        for model_name in a.models:
            proc=AutoImageProcessor.from_pretrained(model_name); model=AutoModelForDepthEstimation.from_pretrained(model_name,dtype=torch.float16).to('cuda:0').eval(); loaded.append((proc,model))
        all_results[domain][f'ensemble_{len(a.models)}']=evaluate(loaded,items)[f'ensemble_{len(a.models)}']
        del loaded; torch.cuda.empty_cache()
    Path(a.output).write_text(json.dumps({'models':a.models,'limit':a.limit,'metrics':'per-image median-scale aligned','results':all_results},indent=2)+'\n')
    print(json.dumps(json.loads(Path(a.output).read_text()),indent=2))
if __name__=='__main__': main()
