export interface Speech {
  buffer: AudioBuffer; text: string; priority: number;
}
type Entry = Speech & { done: (played: boolean) => void };
// Higher number = higher priority, matching the backend.
export class AudioQueue {
  private queue: Entry[] = [];
  private current: { entry: Entry; source: AudioBufferSourceNode } | null = null;
  constructor(private context: AudioContext, private changed: (playing: boolean) => void) {}
  enqueue(speech: Speech): Promise<boolean> {
    return new Promise(done => {
      if (this.current && speech.priority > this.current.entry.priority) {
        this.cancelCurrent();
        this.queue = this.queue.filter(e => {
          if (e.priority < speech.priority) { e.done(false); return false; } return true;
        });
      }
      this.queue.push({...speech, done});
      this.queue.sort((a,b) => b.priority - a.priority);
      this.next();
    });
  }
  private cancelCurrent() {
    const old = this.current; this.current = null;
    if (old) { old.source.onended = null; old.source.stop(); old.source.disconnect(); old.entry.done(false); }
  }
  private next() {
    if (this.current) return;
    const entry = this.queue.shift();
    if (!entry) { this.changed(false); return; }
    const source = this.context.createBufferSource();
    source.buffer = entry.buffer; source.connect(this.context.destination);
    this.current = {entry, source}; this.changed(true);
    source.onended = () => {
      if (this.current?.source !== source) return;
      source.disconnect(); this.current = null; entry.done(true); this.next();
    };
    source.start();
  }
  clear() {
    this.cancelCurrent();
    this.queue.splice(0).forEach(e=>e.done(false));
    this.changed(false);
  }
}
