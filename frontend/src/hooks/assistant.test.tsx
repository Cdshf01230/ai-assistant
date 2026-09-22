// @vitest-environment jsdom
import {act} from 'react';
import {createRoot,type Root} from 'react-dom/client';
import {describe,it,expect,vi,beforeEach,afterEach} from 'vitest';
import useAssistant,{descriptionIsComplete} from './useAssistant';
import * as api from '@/api';
const mocks=vi.hoisted(()=>({
  utterance:null as any, vision:null as any, enable:vi.fn(), start:vi.fn(), stop:vi.fn(), send:vi.fn(),
  sources:[] as any[], tracks:[] as any[],
}));
vi.mock('@/api',()=>({
  bootstrap:vi.fn(),fetchAlerts:vi.fn(),stt:vi.fn(),intent:vi.fn(),describe:vi.fn(),tts:vi.fn(),
  apiUrl:(p:string)=>p,audioBytes:async()=>new ArrayBuffer(1),
}));
vi.mock('./useVoiceCapture',()=>({
  useVoiceCapture:(cb:any)=>{mocks.utterance=cb;return {enable:mocks.enable,start:async()=>{},stop:vi.fn(),speech:false};},
}));
vi.mock('./useVisionWebSocket',()=>({
  useVisionWebSocket:(cb:any)=>{mocks.vision=cb;return {start:mocks.start,stop:mocks.stop,sendFrame:mocks.send,wsState:'connected'};},
}));
let a:ReturnType<typeof useAssistant>,root:Root,host:HTMLElement;
const route=(mode:string)=>({intent:'switch_mode',mode,resume_mode:mode,reply_code:null,should_describe:false}) as any;
const deferred=()=>{let resolve!:(x:any)=>void;const promise=new Promise<any>(r=>resolve=r);return{resolve,promise};};
async function flush(){await act(async()=>{await Promise.resolve();await Promise.resolve();});}
beforeEach(()=>{
  vi.useFakeTimers();vi.clearAllMocks();mocks.sources=[];
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  class Context {
    destination={};sampleRate=48000;
    resume=async()=>{};close=async()=>{};decodeAudioData=async()=>({duration:1});
    createBufferSource(){const s={buffer:null,onended:null,connect:vi.fn(),disconnect:vi.fn(),stop:vi.fn(),start:vi.fn()};mocks.sources.push(s);return s;}
  }
  vi.stubGlobal('AudioContext',Context);
  vi.stubGlobal('MediaStream',class {constructor(public tracks:any[]){}getAudioTracks(){return this.tracks;}getTracks(){return this.tracks;}});
  const track={stop:vi.fn(),addEventListener:vi.fn()};mocks.tracks=[track];
  Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:async()=>({getTracks:()=>[track],getAudioTracks:()=>[track]})}});
  vi.spyOn(HTMLMediaElement.prototype,'play').mockResolvedValue();
  vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue({drawImage:vi.fn()} as any);
  vi.spyOn(HTMLCanvasElement.prototype,'toDataURL').mockReturnValue('data:image/jpeg;base64,abc');
  vi.stubGlobal('fetch',vi.fn(async()=>({ok:true,arrayBuffer:async()=>new ArrayBuffer(1)})));
  vi.mocked(api.bootstrap).mockResolvedValue({
    welcome:{reply_code:'WELCOME',reply_text:'hello'},system_phrases:{},system_audio:{},audio_priorities:{},
    voice_capture:{silence_timeout_ms:800,max_utterance_ms:8000,resume_after_playback_ms:300},
  });
  vi.mocked(api.fetchAlerts).mockResolvedValue({phrases:{STOP:'Dừng lại'},audio_url_pattern:'/audio/{code}.wav'});
  vi.mocked(api.tts).mockResolvedValue({audio_base64:'AAAA',cached:false,message_code:null});
  vi.mocked(api.intent).mockImplementation(async(text)=>text==='tạm dừng'?{...route('paused'),intent:'pause',resume_mode:'narration'}
    :text==='tiếp tục'?{...route('narration'),intent:'resume'}:route(text==='dẫn đường'?'guide':'narration'));
  host=document.createElement('div');document.body.append(host);root=createRoot(host);
});
async function mount(){
  function Harness(){a=useAssistant();return a.state.screen==='main'?<video ref={a.videoRef}/>:null;}
  await act(async()=>root.render(<Harness/>));
  await act(async()=>{await a.start();});
  Object.defineProperty(a.videoRef.current,'readyState',{value:4,configurable:true});
  Object.defineProperty(a.videoRef.current,'videoWidth',{value:640,configurable:true});
  Object.defineProperty(a.videoRef.current,'videoHeight',{value:360,configurable:true});
}
afterEach(async()=>{await act(async()=>root.unmount());host.remove();vi.useRealTimers();vi.restoreAllMocks();vi.unstubAllGlobals();});
describe('assistant integration',()=>{
  it('rejects visibly punctuated and unpunctuated answer fragments',()=>{
    expect(descriptionIsComplete('Một người ở phía bên phải đang')).toBe(false);
    expect(descriptionIsComplete('Một người ở phía bên phải đang.')).toBe(false);
    expect(descriptionIsComplete('Có một người ở phía bên phải.')).toBe(true);
  });
  it('keeps a spoken mode switch when an obstacle alert arrives during STT',async()=>{
    await mount();await act(async()=>a.command('dẫn đường'));
    const pending=deferred();vi.mocked(api.stt).mockReturnValue(pending.promise);
    act(()=>{void mocks.utterance(new Blob());});
    const call=vi.mocked(api.stt).mock.calls[0];
    await act(async()=>{mocks.vision({ok:true,decision:{alert:true,message_code:'STOP'}});});
    expect(call[4]?.aborted).toBe(false);
    await act(async()=>pending.resolve({...route('narration'),turn_id:call[3],text:'thuyết minh',repeat:false,confidence:.9}));
    expect(a.state.mode).toBe('narration');
  });
  it('routes a legacy STT transcript without turn_id instead of silently ignoring speech',async()=>{
    await mount();
    vi.mocked(api.stt).mockResolvedValue({text:'thuyết minh',confidence:.9,repeat:false} as any);
    await act(async()=>{void mocks.utterance(new Blob());});await flush();
    expect(a.state.transcript).toBe('thuyết minh');
    expect(a.state.mode).toBe('narration');
  });
  it('attaches camera; runs continuous warnings only in guide; pause still listens for resume',async()=>{
    await mount();expect(a.videoRef.current?.srcObject).toBeTruthy();
    await act(async()=>a.command('dẫn đường'));
    await act(async()=>{vi.advanceTimersByTime(400);});
    expect(mocks.start).toHaveBeenCalled();expect(mocks.send).toHaveBeenCalled();
    mocks.stop.mockClear();
    await act(async()=>a.command('thuyết minh'));
    expect(a.state.mode).toBe('narration');expect(mocks.stop).toHaveBeenCalled();
    await act(async()=>{mocks.vision({ok:true,decision:{alert:true,message_code:'STOP'}});});
    expect(a.state.warning).toBeNull();
    await act(async()=>a.command('tạm dừng'));
    expect(a.state.mode).toBe('paused');
    await act(async()=>{vi.advanceTimersByTime(400);});
    expect(mocks.enable).toHaveBeenLastCalledWith(true);
    await act(async()=>a.command('tiếp tục'));expect(a.state.mode).toBe('narration');
  });
  it('clears an active guide warning and its audio before entering narration',async()=>{
    await mount();await act(async()=>a.command('dẫn đường'));
    await act(async()=>{mocks.vision({ok:true,decision:{alert:true,message_code:'STOP'}});});
    await flush();
    expect(a.state.warning?.code).toBe('STOP');
    const warningAudio=mocks.sources.at(-1);
    expect(warningAudio?.start).toHaveBeenCalled();

    await act(async()=>a.command('thuyết minh'));
    expect(a.state.mode).toBe('narration');
    expect(a.state.warning).toBeNull();
    expect(warningAudio.stop).toHaveBeenCalled();

    await act(async()=>{mocks.vision({ok:true,decision:{alert:true,message_code:'STOP'}});});
    expect(a.state.warning).toBeNull();
  });
  it('does not report an iOS autoplay block as a permission error',async()=>{
    vi.mocked(HTMLMediaElement.prototype.play).mockRejectedValue(new DOMException(
      'The request is not allowed by the user agent or the platform in the current context.','NotAllowedError',
    ));
    await mount();await flush();
    expect(a.videoRef.current?.srcObject).toBeTruthy();
    expect(a.state.screen).toBe('main');expect(a.state.error).toBe('');
  });
  it('allows only one STT and rejects low-confidence commands without requiring repeat audio',async()=>{
    await mount();const pending=deferred();
    vi.mocked(api.stt).mockReturnValue(pending.promise);
    act(()=>{mocks.utterance(new Blob());mocks.utterance(new Blob());});
    expect(api.stt).toHaveBeenCalledTimes(1);
    const id=vi.mocked(api.stt).mock.calls[0][3];
    await act(async()=>pending.resolve({turn_id:id,text:'dẫn đường',repeat:true,...route('guide')}));
    await flush();expect(a.state.mode).toBe('waiting');
  });
  it('starts filler before Gemini finishes and ignores a late answer after switching mode',async()=>{
    await mount();await act(async()=>a.command('thuyết minh'));
    const settings=await api.bootstrap();
    settings.system_phrases.HEARD_THINKING='Đang kiểm tra cảnh';
    const pending=deferred();vi.mocked(api.describe).mockReturnValue(pending.promise);
    vi.mocked(api.stt).mockImplementation(async(_b,_m,_r,id)=>({
      ...route('narration'),turn_id:id,text:'có gì',repeat:false,confidence:.9,
      intent:'ask_description',question:'có gì',should_describe:true,reply_code:'HEARD_THINKING',
    }));
    await act(async()=>{void mocks.utterance(new Blob());});await flush();
    expect(api.describe).toHaveBeenCalledOnce();expect(mocks.sources.length).toBeGreaterThan(0);
    expect(a.state.answer).toBe('');
    await act(async()=>a.command('dẫn đường'));
    const id=vi.mocked(api.describe).mock.calls[0][2];
    await act(async()=>pending.resolve({request_id:id,text:'stale',audio_base64:'AAAA'}));
    expect(a.state.answer).not.toBe('stale');expect(a.state.mode).toBe('guide');
  });
});
