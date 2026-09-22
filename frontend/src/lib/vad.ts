export interface VadOptions { silenceMs: number; maxMs: number; preRollMs: number; minSpeechMs: number; threshold: number }
export class Segmenter {
  private pre: Float32Array[] = [];
  private chunks: Float32Array[] = [];
  private talking = false;
  private silence = 0;
  private duration = 0;
  private voiced = 0;
  constructor(private sampleRate: number, private speech: (value:boolean)=>void,
    private utterance: (chunks: Float32Array[], sampleRate:number)=>void,
    private options: VadOptions = {silenceMs:800,maxMs:8000,preRollMs:300,minSpeechMs:180,threshold:0.012}) {}
  reset() {
    this.pre = []; this.chunks = []; this.talking=false;
    this.silence=0; this.duration=0; this.voiced=0; this.speech(false);
  }
  push(chunk: Float32Array) {
    const ms = chunk.length / this.sampleRate * 1000;
    const rms = Math.sqrt(chunk.reduce((s,v)=>s+v*v,0)/chunk.length);
    const voice = rms >= this.options.threshold;
    if (!this.talking) {
      this.pre.push(chunk);
      while (this.pre.length > Math.ceil(this.options.preRollMs/ms)) this.pre.shift();
      if (!voice) return;
      this.talking=true; this.chunks=this.pre.splice(0); this.speech(true);
    } else this.chunks.push(chunk);
    this.duration+=ms; this.voiced+= voice ? ms : 0;
    this.silence=voice ? 0 : this.silence+ms;
    if (this.silence>=this.options.silenceMs || this.duration>=this.options.maxMs) {
      const chunks=this.chunks; const valid=this.voiced>=this.options.minSpeechMs;
      this.reset();
      if(valid) this.utterance(chunks, this.sampleRate);
    }
  }
}
export function encodeWav(chunks: Float32Array[], rate: number): Blob {
  const n=chunks.reduce((s,c)=>s+c.length,0), bytes=new ArrayBuffer(44+n*2), v=new DataView(bytes);
  const str=(o:number,s:string)=>[...s].forEach((c,i)=>v.setUint8(o+i,c.charCodeAt(0)));
  str(0,'RIFF'); v.setUint32(4,36+n*2,true); str(8,'WAVEfmt ');
  v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
  v.setUint32(24,rate,true); v.setUint32(28,rate*2,true); v.setUint16(32,2,true); v.setUint16(34,16,true);
  str(36,'data'); v.setUint32(40,n*2,true);
  let offset=44; for(const chunk of chunks) for(const sample of chunk) {
    const value=Math.max(-1,Math.min(1,sample)); v.setInt16(offset,value<0?value*32768:value*32767,true); offset+=2;
  }
  return new Blob([bytes],{type:'audio/wav'});
}
