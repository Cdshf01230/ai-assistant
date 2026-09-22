import { useCallback, useEffect, useRef, useState } from 'react';
import { Segmenter, encodeWav } from '@/lib/vad';
export function useVoiceCapture(onUtterance:(blob:Blob)=>void) {
  const [speech,setSpeech]=useState(false);
  const callback=useRef(onUtterance); callback.current=onUtterance;
  const state=useRef<{node:AudioWorkletNode;source:MediaStreamAudioSourceNode;mute:GainNode;vad:Segmenter;enabled:boolean}|null>(null);
  const stop=useCallback(()=>{
    const old=state.current; state.current=null;
    if(old) { old.node.port.onmessage=null; old.node.disconnect(); old.source.disconnect(); old.mute.disconnect(); old.vad.reset(); }
  },[]);
  const start=useCallback(async(stream:MediaStream,ctx:AudioContext,silence=800,max=8000)=>{
    stop();
    const code=`class Capture extends AudioWorkletProcessor {
      constructor(){super();this.buffer=new Float32Array(1024);this.offset=0;}
      process(inputs){const input=inputs[0]?.[0];if(input)for(const sample of input){
        this.buffer[this.offset++]=sample;
        if(this.offset===1024){this.port.postMessage(this.buffer);this.buffer=new Float32Array(1024);this.offset=0;}
      }return true;}
    } registerProcessor('voice-capture',Capture);`;
    const url=URL.createObjectURL(new Blob([code],{type:'application/javascript'}));
    try { await ctx.audioWorklet.addModule(url); } finally {URL.revokeObjectURL(url);}
    const node=new AudioWorkletNode(ctx,'voice-capture');
    const source=ctx.createMediaStreamSource(stream),mute=ctx.createGain();mute.gain.value=0;
    const vad=new Segmenter(ctx.sampleRate,setSpeech,(chunks,rate)=>{
      // Close the gate synchronously before invoking async STT.
      if(state.current)state.current.enabled=false;
      callback.current(encodeWav(chunks,rate));
    },{silenceMs:silence,maxMs:max,preRollMs:300,minSpeechMs:180,threshold:0.012});
    state.current={node,source,mute,vad,enabled:false};
    node.port.onmessage=e=>{if(state.current?.enabled)vad.push(new Float32Array(e.data));};
    source.connect(node);node.connect(mute);mute.connect(ctx.destination);
  },[stop]);
  const enable=useCallback((value:boolean)=>{
    const current=state.current;if(!current)return;
    if(!value)current.vad.reset();current.enabled=value;
  },[]);
  useEffect(()=>stop,[stop]);
  return {start,stop,enable,speech};
}
