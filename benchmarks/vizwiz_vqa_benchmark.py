#!/usr/bin/env python3
"""Run a small VizWiz-VQA validation benchmark with Qwen3-VL."""
from __future__ import annotations
import argparse, io, json, os, time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ubuntu/ai-assistant/models/hf")

ROOT=Path('/home/ubuntu/ai-assistant')
ANN=ROOT/'benchmarks/data/vizwiz_vqa/annotations/val.json'
IMG=ROOT/'benchmarks/data/vizwiz_vqa/val'

def norm(text):
    return ' '.join(str(text).lower().strip().split())

def prepare(image):
    from PIL import Image
    image=image.convert('RGB')
    if max(image.size)>640:
        scale=640/max(image.size)
        image=image.resize((round(image.width*scale),round(image.height*scale)),Image.LANCZOS)
    buf=io.BytesIO(); image.save(buf,format='JPEG',quality=80); buf.seek(0)
    return Image.open(buf).convert('RGB')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=20); p.add_argument('--output',required=True); a=p.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    rows=json.loads(ANN.read_text(encoding='utf-8'))
    rows=[r for r in rows if (IMG/r['image']).exists()][:a.limit]
    model_name='Qwen/Qwen3-VL-4B-Instruct'
    processor=AutoProcessor.from_pretrained(model_name)
    model=AutoModelForImageTextToText.from_pretrained(model_name,dtype=torch.float16,attn_implementation='sdpa',device_map='cuda:0').eval()
    prompt=('You are answering a visual question asked by a blind person. Look carefully at the image. '
            'If the image cannot answer the question, begin with UNANSWERABLE. Otherwise begin with ANSWERABLE. '
            'Then give a short direct answer in English. Output one line only.')
    results=[]
    for row in rows:
        image=prepare(Image.open(IMG/row['image']))
        messages=[{'role':'system','content':[{'type':'text','text':prompt}]},{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':row['question']}]}]
        inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to(model.device); n=inputs['input_ids'].shape[-1]
        t=time.perf_counter()
        with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=16,do_sample=False)
        text=processor.decode(out[0][n:],skip_special_tokens=True).strip()
        pred_answerable=not norm(text).startswith('unanswerable')
        results.append({'image':row['image'],'question':row['question'],'ground_truth_answerable':bool(row.get('answerable',1)),'prediction_answerable':pred_answerable,'ground_truth':row.get('answer_type'),'prediction':text,'latency_ms':round((time.perf_counter()-t)*1000)})
    acc=sum(r['ground_truth_answerable']==r['prediction_answerable'] for r in results)/max(len(results),1)
    report={'dataset':'VizWiz-VQA validation','model':model_name,'samples':len(results),'answerability_accuracy':acc,'results':results}
    Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'dataset':report['dataset'],'samples':len(results),'answerability_accuracy':acc,'median_latency_ms':__import__('statistics').median(r['latency_ms'] for r in results)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
