import {useCallback,useEffect,useRef,useState} from 'react';
import * as api from '@/api';
import {AudioQueue,type Speech} from '@/lib/audio';
import {useVoiceCapture} from './useVoiceCapture';
import {useVisionWebSocket} from './useVisionWebSocket';
import type {Bootstrap,Mode,Route,Warning,Diagnostics,VisionResponse,AppStatus} from '@/types';

interface State {
  screen:'permission'|'initializing'|'main'|'denied'; mode:Mode; resumeMode:'guide'|'narration';
  busy:'idle'|'transcribing'|'waiting'; speaking:boolean; transcript:string; answer:string; error:string;
  warning:Warning|null; hidden:boolean;
}
const initial:State={screen:'permission',mode:'waiting',resumeMode:'guide',busy:'idle',speaking:false,transcript:'',answer:'',error:'',warning:null,hidden:false};
export function captureFrame(video:HTMLVideoElement) {
  if(video.readyState<2||!video.videoWidth)throw new Error('Camera chưa sẵn sàng.');
  const ratio=Math.min(1,640/Math.max(video.videoWidth,video.videoHeight));
  const c=document.createElement('canvas');c.width=Math.round(video.videoWidth*ratio);c.height=Math.round(video.videoHeight*ratio);
  c.getContext('2d')!.drawImage(video,0,0,c.width,c.height);
  return c.toDataURL('image/jpeg',.8).split(',')[1];
}
export function descriptionIsComplete(text:string) {
  const clean=text.trim(),words=clean.replace(/[.!?…]+$/u,'').toLocaleLowerCase('vi').split(/\s+/u);
  const dangling=new Set(['đang','sẽ','và','hoặc','nhưng','ở','có','là','một','những','các','phía','bên','trên','dưới','trong','với','để','của','khi','do','từ','đến','về']);
  return words.length>=4&&/[.!?…]$/u.test(clean)&&!dangling.has(words.at(-1)??'');
}
export default function useAssistant() {
  const [state,setState]=useState(initial),current=useRef(initial);
  const update=useCallback((patch:Partial<State>)=>{current.current={...current.current,...patch};setState(current.current);},[]);
  const videoRef=useRef<HTMLVideoElement>(null),stream=useRef<MediaStream|null>(null);
  const context=useRef<AudioContext|null>(null),player=useRef<AudioQueue|null>(null);
  const config=useRef<Bootstrap|null>(null),cache=useRef(new Map<string,AudioBuffer>());
  const phrases=useRef<Record<string,string>>({}),last=useRef<Speech|null>(null);
  const epoch=useRef(0),audioEpoch=useRef(0),descriptionEpoch=useRef(0),sttBusy=useRef(false),describeBusy=useRef(false);
  const controllers=useRef(new Map<AbortController,'control'|'audio'|'description'>());
  const utteranceHandler=useRef<(blob:Blob)=>void>(()=>{});
  const voice=useVoiceCapture(blob=>utteranceHandler.current(blob));
  const voiceRef=useRef(voice);voiceRef.current=voice;
  const resumeTimer=useRef<ReturnType<typeof setTimeout>|null>(null);
  const warningTimer=useRef<ReturnType<typeof setTimeout>|null>(null);
  const [diag,setDiag]=useState<Diagnostics>({fps:0,depthLatency:0,vlmLatency:0,freePath:0,
    queueWait:0,serverTotal:0,droppedFrames:0,wsState:'disconnected'});
  const fps=useRef<number[]>([]);
  const hadConnection=useRef(false),outage=useRef(false);
  const mounted=useRef(true);
  const reconcile=useCallback(()=>{
    if(resumeTimer.current)clearTimeout(resumeTimer.current);
    voiceRef.current.enable(false);
    const s=current.current;
    if(s.screen!=='main'||s.speaking||s.hidden||sttBusy.current)return;
    // Paused mode still listens for "tiếp tục"; vision is stopped.
    resumeTimer.current=setTimeout(()=>{
      if(mounted.current&&!current.current.speaking&&!sttBusy.current&&!current.current.hidden)
        voiceRef.current.enable(true);
    },config.current?.voice_capture.resume_after_playback_ms??300);
  },[]);
  const abortRequests=useCallback((mediaOnly=false)=>{
    controllers.current.forEach((scope,c)=>{
      if(!mediaOnly||scope!=='control'){c.abort();controllers.current.delete(c);}
    });
  },[]);
  const withRequest=useCallback(async<T,>(fn:(signal:AbortSignal)=>Promise<T>,scope:'control'|'audio'|'description'='control')=>{
    const c=new AbortController();controllers.current.set(c,scope);
    try{return await fn(c.signal);}finally{controllers.current.delete(c);}
  },[]);
  const report=useCallback((error:unknown)=>{
    update({error:error instanceof Error?error.message:'Đã xảy ra lỗi. Hãy thử lại.'});
  },[update]);
  const decode=useCallback(async(key:string,base64?:string,signal?:AbortSignal)=>{
    const saved=cache.current.get(key);if(saved)return saved;
    if(!context.current)throw new Error('Âm thanh chưa sẵn sàng.');
    let bytes:ArrayBuffer;
    if(base64)bytes=Uint8Array.from(atob(base64),c=>c.charCodeAt(0)).buffer;
    else {
      bytes=await api.audioBytes(key,signal);
    }
    const buffer=await context.current.decodeAudioData(bytes);
    // Keep fixed prompts and most recent dynamic replies only.
    if(cache.current.size>=64)cache.current.delete(cache.current.keys().next().value!);
    cache.current.set(key,buffer);return buffer;
  },[]);
  const say=useCallback(async(text:string,priority=60,url?:string|null,base64?:string)=>{
    const turn=epoch.current, version=audioEpoch.current;
    try{
      const buffer=await withRequest(async signal=>{
        if(url)return decode(url,undefined,signal);
        if(base64)return decode(text,base64);
        const saved=cache.current.get(text);if(saved)return saved;
        const result=await api.tts(text,signal);return decode(text,result.audio_base64);
      },'audio');
      if(turn!==epoch.current||version!==audioEpoch.current||!player.current)return;
      const speech={buffer,text,priority};if(priority!==20)last.current=speech;
      await player.current.enqueue(speech);
    }catch(error){if(turn===epoch.current&&version===audioEpoch.current)report(error);}
  },[decode,withRequest,report]);
  const fixed=useCallback((code:string,priority=60)=>{
    const text=config.current?.system_phrases[code]??phrases.current[code];
    return text?say(text,priority,config.current?.system_audio[code]??(code in phrases.current?'/audio/'+code+'.wav':null)):Promise.resolve();
  },[say]);
  const handleVision=useCallback((msg:VisionResponse)=>{
    // Continuous obstacle inference belongs only to guide mode. Narration
    // captures a frame on demand when the user asks a question.
    if(current.current.mode!=='guide'||current.current.hidden)return;
    if(!msg.ok){update({error:'Không xử lý được hình ảnh. Đang thử frame tiếp theo.'});return;}
    const now=performance.now();fps.current.push(now);
    fps.current=fps.current.filter(t=>now-t<5000);
    const times=fps.current;
    setDiag(d=>({...d,fps:times.length>1?1000*(times.length-1)/Math.max(1,now-times[0]):0,
      depthLatency:msg.depth_latency_ms??msg.latency_ms??0,vlmLatency:msg.vlm_latency_ms??0,
      freePath:msg.geometry?.free_min??0,queueWait:msg.queue_wait_ms??0,
      serverTotal:msg.server_total_ms??msg.latency_ms??0,droppedFrames:msg.dropped_frames??0}));
    const decision=msg.decision;
    // Current backend calls this state `warn`; accept `active` from old builds.
    if(decision&&(decision.warn===false||decision.active===false))update({warning:null});
    if(decision?.alert&&decision.message_code){
      const code=decision.message_code, text=phrases.current[code]??'Có vật cản. Hãy dừng lại.';
      update({warning:{code,text,ts:Date.now()}});
      if(warningTimer.current)clearTimeout(warningTimer.current);
      warningTimer.current=setTimeout(()=>update({warning:null}),4500);
      // Invalidates in-flight filler/answer audio and discards the active description.
      descriptionEpoch.current++;audioEpoch.current++;abortRequests(true);
      player.current?.clear();update({busy:sttBusy.current?'transcribing':'idle'});
      void say(text,100,'/audio/'+code+'.wav');
    }
  },[update,say,abortRequests]);
  const vision=useVisionWebSocket(handleVision);
  const visionRef=useRef(vision);visionRef.current=vision;
  const cancelTurn=useCallback(()=>{
    epoch.current++;audioEpoch.current++;abortRequests();player.current?.clear();
    update({busy:'idle',error:''});voiceRef.current.enable(false);
  },[abortRequests,update]);
  const ask=useCallback(async(question:string,route:Route)=>{
    if(describeBusy.current){await say('Mình đang xử lý câu hỏi trước. Bạn chờ một chút nhé.',60);return;}
    let image:string;try{image=captureFrame(videoRef.current!);}catch(error){report(error);return;}
    describeBusy.current=true;
    const turn=epoch.current, description=descriptionEpoch.current, requestId=crypto.randomUUID();
    update({busy:'waiting',answer:'',transcript:question});
    // Start Gemini now; filler is independent and must never gate the response.
    const response=withRequest(signal=>api.describe(image,question,requestId,signal),'description');
    void fixed(route.reply_code??'HEARD_THINKING',20);
    reconcile();
    try {
      const result=await response;
      if(turn!==epoch.current||description!==descriptionEpoch.current||(result.request_id!=null&&result.request_id!==requestId))return;
      if(!descriptionIsComplete(result.text))throw new Error('Câu trả lời bị gián đoạn. Hãy hỏi lại.');
      audioEpoch.current++; // Drop any filler TTS still in flight.
      update({answer:result.text,busy:'idle'});
      await say(result.text,40,null,result.audio_base64);
    }catch(error){if(turn===epoch.current&&description===descriptionEpoch.current)report(error);}
    finally{
      describeBusy.current=false;
      if(turn===epoch.current&&!sttBusy.current)update({busy:'idle'});
      reconcile();
    }
  },[fixed,reconcile,report,say,update,withRequest]);
  const applyRoute=useCallback(async(route:Route)=>{
    const previous=current.current.mode;
    if(route.intent==='pause'||route.intent==='resume'||route.intent==='switch_mode'||(route.should_describe&&previous!=='narration')){
      cancelTurn();
      // Close the safety channel before publishing a non-guide mode. Waiting
      // for the React effect leaves a window where an old guide alert can play.
      if(route.mode!=='guide')visionRef.current.stop();
      if(warningTimer.current){clearTimeout(warningTimer.current);warningTimer.current=null;}
      update({mode:route.mode,resumeMode:route.resume_mode??current.current.resumeMode,warning:null});
    }
    if(route.intent==='repeat_last'){
      const spoken=last.current;
      if(spoken)await player.current?.enqueue({...spoken,priority:80});
      else await say('Chưa có câu trả lời để lặp lại.',60);
    }else if(route.should_describe&&route.question){
      void ask(route.question,route);
    }else if(route.reply_code){
      await fixed(route.reply_code,route.audio_priority??60);
    }
    reconcile();
  },[ask,cancelTurn,fixed,reconcile,say,update]);
  utteranceHandler.current=async blob=>{
    if(sttBusy.current||current.current.speaking||current.current.hidden)return;
    sttBusy.current=true;voiceRef.current.enable(false);
    const turn=epoch.current,turnId=crypto.randomUUID();
    update({busy:'transcribing',error:''});
    try {
      const result=await withRequest(signal=>api.stt(blob,current.current.mode,current.current.resumeMode,turnId,signal));
      if(turn!==epoch.current||(result.turn_id!=null&&result.turn_id!==turnId))return;
      update({transcript:result.text,busy:describeBusy.current?'waiting':'idle'});
      // Never execute a low-confidence transcript, even when no repeat WAV exists.
      if(result.repeat)await say('Xin nói lại.',80,result.reply_audio_url,result.audio_base64);
      else {
        // Older notebook APIs return only the transcript. The local epoch still
        // protects against stale responses; ask the backend router when needed.
        const route=result.intent===undefined
          ? await withRequest(signal=>api.intent(result.text,current.current.mode,current.current.resumeMode,signal))
          : result;
        if(turn!==epoch.current)return;
        if(!['waiting','guide','narration','paused'].includes(route.mode)||!route.intent)
          throw new Error('Backend chưa hỗ trợ lệnh giọng nói. Hãy chạy lại notebook mới.');
        await applyRoute(route);
      }
    }catch(error){if(turn===epoch.current)report(error);}
    finally{
      sttBusy.current=false;
      if(turn===epoch.current)update({busy:describeBusy.current?'waiting':'idle'});
      reconcile();
    }
  };
  const command=useCallback(async(text:string)=>{
    void context.current?.resume().catch(report);
    cancelTurn();
    const turn=epoch.current;
    try{
      const route=await withRequest(signal=>api.intent(text,current.current.mode,current.current.resumeMode,signal));
      if(turn===epoch.current)await applyRoute(route);
    }catch(error){if(turn===epoch.current)report(error);}
    finally{reconcile();}
  },[applyRoute,cancelTurn,reconcile,report,withRequest]);
  const attachVideo=useCallback(async()=>{
    const video=videoRef.current;
    if(!video||!stream.current)return;
    if(video.srcObject!==stream.current)video.srcObject=stream.current;
    // iOS and embedded WebViews can reject an explicit play() even for a muted,
    // inline video. The autoplay attribute will retry when the stream has data;
    // this is not a camera-permission failure and must not become a red warning.
    try{await video.play();}
    catch(error){
      if((error as DOMException)?.name!=='NotAllowedError')throw error;
      video.addEventListener('loadedmetadata',()=>{void video.play().catch(()=>{});},{once:true});
    }
  },[]);
  const start=useCallback(async()=>{
    if(current.current.screen==='initializing')return;
    cancelTurn();const turn=epoch.current;
    update({screen:'initializing'});
    try {
      // Resume synchronously inside the user gesture, before any network await.
      const ctx=new AudioContext();context.current=ctx;
      const unlocked=ctx.resume();
      const media=navigator.mediaDevices.getUserMedia({
        video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}},
        audio:{echoCancellation:true,noiseSuppression:true,channelCount:1},
      });
      const acquired=await media;
      if(turn!==epoch.current){acquired.getTracks().forEach(t=>t.stop());return;}
      stream.current=acquired;await unlocked;
      acquired.getTracks().forEach(track=>track.addEventListener('ended',()=>{
        if(!mounted.current||stream.current!==acquired)return;
        cancelTurn();voiceRef.current.enable(false);voiceRef.current.stop();visionRef.current.stop();
        acquired.getTracks().forEach(t=>t.stop());stream.current=null;
        void ctx.close();context.current=null;
        update({screen:'permission',error:'Camera hoặc micro đã bị ngắt. Hãy bắt đầu lại.',speaking:false});
      },{once:true}));
      player.current=new AudioQueue(ctx,playing=>{
        if(!mounted.current)return;
        update({speaking:playing});reconcile();
      });
      const [settings,alerts]=await withRequest(signal=>Promise.all([api.bootstrap(signal),api.fetchAlerts(signal)]));
      if(turn!==epoch.current)return;
      config.current=settings;phrases.current=alerts.phrases;
      await voiceRef.current.start(new MediaStream(acquired.getAudioTracks()),ctx,settings.voice_capture.silence_timeout_ms,settings.voice_capture.max_utterance_ms);
      // Preload the critical WAVs before enabling guide/narration modes.
      await withRequest(signal=>Promise.all(Object.keys(alerts.phrases).map(code=>decode('/audio/'+code+'.wav',undefined,signal))));
      await withRequest(signal=>Promise.all(Object.values(settings.system_audio)
        .filter((url):url is string=>Boolean(url)).map(url=>decode(url,undefined,signal))));
      if(turn!==epoch.current)return;
      update({screen:'main',mode:'waiting',busy:'idle'});
      // Video attaches in the effect after the main screen has mounted.
      await fixed(settings.welcome.reply_code);
      reconcile();
    }catch(error){
      voiceRef.current.stop();stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;
      await context.current?.close().catch(()=>{});context.current=null;
      if(turn===epoch.current){
        update({screen:(error as Error)?.name==='NotAllowedError'?'denied':'permission',speaking:false});
        report(error);
      }
    }
  },[cancelTurn,decode,fixed,reconcile,report,update,withRequest]);
  useEffect(()=>{
    if(state.screen==='main')void attachVideo().catch(report);
  },[state.screen,state.mode,attachVideo,report]);
  useEffect(()=>{
    if(state.mode!=='guide'||state.hidden){
      hadConnection.current=false;outage.current=false;return;
    }
    if(vision.wsState==='connected'){
      hadConnection.current=true;
      if(outage.current){outage.current=false;void fixed('SAFETY_RESUMED');}
    }else if(hadConnection.current&&!outage.current){
      outage.current=true;void fixed('SAFETY_UNAVAILABLE',80);
    }
  },[vision.wsState,state.mode,state.hidden,fixed]);
  const visionActive=state.screen==='main'&&state.mode==='guide'&&!state.hidden;
  useEffect(()=>{
    if(!visionActive){
      vision.stop();setDiag(d=>({...d,fps:0}));fps.current=[];return;
    }
    vision.start();
    const raw=Number(new URLSearchParams(location.search).get('fps')??5);
    const hz=Number.isFinite(raw)?Math.min(5,Math.max(1,raw)):5;
    const timer=setInterval(()=>{
      if(videoRef.current?.readyState&&videoRef.current.readyState>=2)
        visionRef.current.sendFrame(()=>captureFrame(videoRef.current!));
    },1000/hz);
    return()=>{clearInterval(timer);visionRef.current.stop();};
  },[visionActive,vision.start,vision.stop]);
  useEffect(()=>{
    const hidden=()=>{
      update({hidden:document.hidden});
      if(document.hidden&&current.current.screen==='main'){cancelTurn();voiceRef.current.enable(false);}
      else {void context.current?.resume().then(reconcile).catch(report);reconcile();}
    };
    document.addEventListener('visibilitychange',hidden);
    return()=>document.removeEventListener('visibilitychange',hidden);
  },[cancelTurn,reconcile,report,update]);
  useEffect(()=>{
    mounted.current=true;
    return()=>{
      mounted.current=false;epoch.current++;abortRequests();player.current?.clear();
      voiceRef.current.stop();visionRef.current.stop();
      stream.current?.getTracks().forEach(t=>t.stop());void context.current?.close();
      if(resumeTimer.current)clearTimeout(resumeTimer.current);
      if(warningTimer.current)clearTimeout(warningTimer.current);
    };
  },[abortRequests]);
  const status:AppStatus=state.speaking?'speaking':state.busy==='transcribing'?'transcribing':voice.speech?'speech'
    :state.mode==='paused'?'paused':state.busy==='waiting'?'waiting':'listening';
  return {state,status,videoRef,start,command,diag:{...diag,wsState:vision.wsState}};
}
