import {useRef,useState,useCallback,useEffect} from 'react';
import {buildWsUrl} from '@/api';
import type {VisionResponse,WsState} from '@/types';
type SocketState={
  ws:WebSocket|null;run:boolean;retry:ReturnType<typeof setTimeout>|null;
  timeout:ReturnType<typeof setTimeout>|null;nextFrameId:number;waitingForResponse:boolean;
};
const MAX_BUFFERED_BYTES=1<<20;
export function useVisionWebSocket(onMessage:(msg:VisionResponse)=>void) {
  const handler=useRef(onMessage);handler.current=onMessage;
  const [wsState,setState]=useState<WsState>('disconnected');
  const state=useRef<SocketState>
    ({ws:null,run:false,retry:null,timeout:null,nextFrameId:0,waitingForResponse:false});
  const connect=useCallback(function connectNow(){
    const s=state.current;if(!s.run||s.ws)return;
    setState('connecting');const ws=new WebSocket(buildWsUrl());s.ws=ws;
    const clear=()=>{if(s.timeout)clearTimeout(s.timeout);s.timeout=null;s.waitingForResponse=false;};
    s.timeout=setTimeout(()=>ws.close(),15000);
    ws.onopen=()=>{if(s.ws!==ws)return;clear();setState('connected');};
    ws.onmessage=e=>{
      if(s.ws!==ws)return;clear();
      try {handler.current(JSON.parse(e.data));}catch{setState('error');}
    };
    ws.onerror=()=>{if(s.ws===ws){setState('error');ws.close();}};
    ws.onclose=()=>{
      if(s.ws!==ws)return;clear();s.ws=null;setState('disconnected');
      if(s.run)s.retry=setTimeout(()=>{s.retry=null;connectNow();},1500);
    };
  },[]);
  const start=useCallback(()=>{state.current.run=true;connect();},[connect]);
  const stop=useCallback(()=>{
    const s=state.current;s.run=false;
    if(s.retry)clearTimeout(s.retry);if(s.timeout)clearTimeout(s.timeout);
    const ws=s.ws;s.ws=null;s.waitingForResponse=false;s.retry=null;s.timeout=null;ws?.close();setState('disconnected');
  },[]);
  const sendFrame=useCallback((capture:()=>string)=>{
    const s=state.current,ws=s.ws;
    // One frame in flight. The sampling timer may tick at 5 Hz, but capture the
    // next *current* image only after the previous result arrives. Sending at a
    // fixed rate while inference is slower creates a stream of stale answers.
    if(ws?.readyState!==WebSocket.OPEN||s.waitingForResponse||ws.bufferedAmount>MAX_BUFFERED_BYTES)return;
    try {
      const data=capture();
      ws.send(JSON.stringify({image_base64:data,frame_id:++s.nextFrameId}));
      if(!s.waitingForResponse){
        s.waitingForResponse=true;
        s.timeout=setTimeout(()=>ws.close(),15000);
      }
    } catch {s.waitingForResponse=false;ws.close();}
  },[]);
  useEffect(()=>stop,[stop]);
  return {wsState,start,stop,sendFrame};
}
