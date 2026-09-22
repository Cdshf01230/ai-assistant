// @vitest-environment jsdom
import {describe,it,expect,vi,beforeEach,afterEach} from 'vitest';
import {AudioQueue} from './audio';
import {Segmenter,encodeWav} from './vad';
describe('audio priority and lifecycle',()=>{
  function setup(){
    const sources:any[]=[];
    const ctx={destination:{},createBufferSource:()=>{const source={connect:vi.fn(),disconnect:vi.fn(),start:vi.fn(),stop:vi.fn(),onended:null};sources.push(source);return source;}};
    const change=vi.fn(),queue=new AudioQueue(ctx as any,change);
    const speech=(priority:number)=>({text:'test',buffer:{} as AudioBuffer,priority});
    return {queue,sources,change,speech};
  }
  it('emergency stops answer and removes queued filler; old completion cannot advance queue',async()=>{
    const {queue,sources,speech,change}=setup();
    const answer=queue.enqueue(speech(40));
    const stale=sources[0].onended;
    const filler=queue.enqueue(speech(20));
    const warning=queue.enqueue(speech(100));
    expect(await answer).toBe(false);expect(await filler).toBe(false);
    expect(sources[0].stop).toHaveBeenCalledOnce();
    stale();expect(sources).toHaveLength(2);
    sources[1].onended();expect(await warning).toBe(true);
    expect(change).toHaveBeenLastCalledWith(false);
  });
  it('answer interrupts filler, and clear resolves all promises',async()=>{
    const {queue,sources,speech}=setup();
    const filler=queue.enqueue(speech(20)),answer=queue.enqueue(speech(40));
    expect(await filler).toBe(false);
    queue.clear();expect(await answer).toBe(false);
    expect(sources[1].onended).toBe(null);
  });
});
describe('VAD',()=>{
  it('preserves pre-roll and emits only after silence; ignores short clicks',()=>{
    const done=vi.fn(),vad=new Segmenter(1000,()=>{},done);
    const quiet=()=>new Float32Array(100),loud=()=>new Float32Array(100).fill(.1);
    for(let i=0;i<5;i++)vad.push(quiet());
    for(let i=0;i<4;i++)vad.push(loud());
    for(let i=0;i<7;i++)vad.push(quiet());expect(done).not.toHaveBeenCalled();
    vad.push(quiet());expect(done).toHaveBeenCalledOnce();
    expect(done.mock.calls[0][0][0][0]).toBe(0);
    vad.push(loud());for(let i=0;i<8;i++)vad.push(quiet());
    expect(done).toHaveBeenCalledOnce();
  });
  it('reset discards a half utterance rather than submitting assistant audio',()=>{
    const done=vi.fn(),vad=new Segmenter(1000,()=>{},done);
    for(let i=0;i<4;i++)vad.push(new Float32Array(100).fill(.1));
    vad.reset();
    for(let i=0;i<10;i++)vad.push(new Float32Array(100));
    expect(done).not.toHaveBeenCalled();
  });
  it('WAV declares actual capture rate',async()=>{
    // jsdom Blob lacks arrayBuffer; FileReader verifies exported WAV.
    const blob=encodeWav([new Float32Array(480)],48000);
    const bytes=await new Promise<ArrayBuffer>(resolve=>{const r=new FileReader();r.onload=()=>resolve(r.result as ArrayBuffer);r.readAsArrayBuffer(blob);});
    expect(new DataView(bytes).getUint32(24,true)).toBe(48000);
    expect(bytes.byteLength).toBe(1004);
  });
});
