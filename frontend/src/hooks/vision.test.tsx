// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {vi,it,expect} from 'vitest';
import {useVisionWebSocket} from './useVisionWebSocket';
it('bounds outstanding frames, reconnects stalled requests, and ignores stale sockets',async()=>{
  vi.useFakeTimers();(globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  const sockets:any[]=[];
  class Socket {
    static OPEN=1;static CONNECTING=0;readyState=0;bufferedAmount=0;
    onopen:any;onclose:any;onerror:any;onmessage:any;
    send=vi.fn();close=vi.fn(()=>{this.readyState=3;this.onclose?.();});
    constructor(){sockets.push(this);}
  }
  vi.stubGlobal('WebSocket',Socket);
  const callback=vi.fn(),capture=vi.fn(()=>'jpeg');
  let hook!:ReturnType<typeof useVisionWebSocket>;
  function Harness(){hook=useVisionWebSocket(callback);return null;}
  const host=document.createElement('div'),root=createRoot(host);
  await act(async()=>root.render(<Harness/>));
  act(()=>hook.start());act(()=>{sockets[0].readyState=1;sockets[0].onopen();});
  act(()=>{hook.sendFrame(capture);hook.sendFrame(capture);});
  expect(capture).toHaveBeenCalledTimes(1);
  expect(JSON.parse(sockets[0].send.mock.calls[0][0]).frame_id).toBe(1);
  act(()=>sockets[0].onmessage({data:JSON.stringify({ok:true,frame_id:1})}));
  expect(callback).toHaveBeenCalledOnce();
  act(()=>hook.sendFrame(capture));
  expect(capture).toHaveBeenCalledTimes(2);
  expect(JSON.parse(sockets[0].send.mock.calls[1][0]).frame_id).toBe(2);
  act(()=>{vi.advanceTimersByTime(16500);});
  expect(sockets).toHaveLength(2);
  const old=sockets[0];old.onmessage({data:JSON.stringify({ok:true})});
  expect(callback).toHaveBeenCalledTimes(1);
  act(()=>{sockets[1].readyState=1;sockets[1].onopen();sockets[1].bufferedAmount=2<<20;hook.sendFrame(capture);});
  expect(capture).toHaveBeenCalledTimes(2);
  act(()=>{sockets[1].bufferedAmount=0;hook.sendFrame(capture);});
  expect(capture).toHaveBeenCalledTimes(3);
  await act(async()=>root.unmount());
  act(()=>vi.advanceTimersByTime(30000));
  expect(sockets).toHaveLength(2);
  vi.useRealTimers();vi.unstubAllGlobals();
});
