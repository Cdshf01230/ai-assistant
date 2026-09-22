import {useEffect,useRef} from 'react';
import useAssistant from '@/hooks/useAssistant';
import VoiceIndicator from '@/components/VoiceIndicator';
import StatusPill from '@/components/StatusPill';
import WarningCard from '@/components/WarningCard';
import DiagnosticsPanel from '@/components/DiagnosticsPanel';
import PermissionScreen from '@/components/PermissionScreen';

export default function App(){
  const a=useAssistant(), {state,status}=a;
  const heading=useRef<HTMLHeadingElement>(null);
  useEffect(()=>{if(state.screen==='main')heading.current?.focus();},[state.screen]);
  if(state.screen!=='main')return <div className="onboarding">
    <PermissionScreen onStart={a.start} initializing={state.screen==='initializing'}/>
    {state.error&&<div className="startup-error" role="alert">
      <p>{state.screen==='denied'?'Hãy cho phép camera và micro trong cài đặt của trang, rồi thử lại.':state.error}</p>
      <button onClick={a.start}>Thử lại</button>
    </div>}
  </div>;
  const active=state.mode==='guide'||state.mode==='narration';
  const safetyActive=state.mode==='guide';
  return <main className="assistant">
    <div className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark" aria-hidden="true">◎</span>
          <h1 ref={heading} tabIndex={-1}>Trợ lý thị giác</h1>
        </div>
        {safetyActive?<StatusPill state={a.diag.wsState}/>:<span className={'ready-pill '+(state.mode==='narration'?'narration-pill':'')}>
          {state.mode==='paused'?'Tạm dừng':state.mode==='narration'?'Thuyết minh':'Sẵn sàng'}
        </span>}
      </header>
      <section className="notices" aria-label="Thông báo">
        {state.warning&&<WarningCard warning={state.warning}/>}
        {safetyActive&&a.diag.wsState!=='connected'&&<p className="notice" role="status">
          {a.diag.wsState==='connecting'?'Đang kết nối camera với trợ lý…':'Mất kết nối cảnh báo. Đang kết nối lại…'}
        </p>}
        {state.error&&<div className="notice error" role="alert">{state.error}
          <button onClick={()=>a.command('trợ giúp')}>Thử lại</button></div>}
      </section>
      <section className={'camera-stage '+(active?'active':'')} aria-label={active?'Hình ảnh trực tiếp từ camera':'Camera đang chờ'}>
        <video ref={a.videoRef} muted autoPlay playsInline className="camera" aria-hidden="true"/>
        {active&&<div className="camera-overlay" aria-hidden="true">
          <span>{state.mode==='guide'?'DẪN ĐƯỜNG':'THUYẾT MINH'}</span>
          <span className="live-dot">● CAMERA</span>
        </div>}
      </section>
      <section className={'conversation '+(active?'camera-active':'')} aria-label="Trợ lý giọng nói">
        {!active&&<VoiceIndicator status={status} mode={null} transcript={state.transcript}/>}
        {active&&<div className="compact-listening" role="status">
          <span aria-hidden="true">●</span>
          {status==='transcribing'?'Đang nhận giọng nói…':status==='speaking'?'Trợ lý đang nói…':status==='waiting'?'Đang suy nghĩ…':'Micro vẫn đang nghe'}
          {state.transcript&&<small>“{state.transcript}”</small>}
        </div>}
        <p className="voice-hint">{state.mode==='waiting'?'Nói “dẫn đường” hoặc “thuyết minh” để bắt đầu.'
          :state.mode==='paused'?'Nói “tiếp tục” khi bạn sẵn sàng.'
          :state.mode==='narration'?'Hỏi về cảnh trước mặt để trợ lý thuyết minh.'
          :'Đưa camera hướng về phía trước. Nói “thuyết minh” để đổi chế độ.'}</p>
        {state.answer&&<article className="answer-card" aria-label="Câu trả lời gần nhất">
          <span className="eyebrow">TRỢ LÝ</span><p>{state.answer}</p>
          <button onClick={()=>a.command('lặp lại')}>Nghe lại câu trả lời</button>
        </article>}
      </section>
      <footer className="controls">
        <div className="mode-controls">
          <button className={'mode-button guide '+(state.mode==='guide'?'selected':'')} aria-pressed={state.mode==='guide'} onClick={()=>a.command('dẫn đường')}>
            <span aria-hidden="true">↗</span><span>Dẫn đường</span>
          </button>
          <button className={'mode-button narration '+(state.mode==='narration'?'selected':'')} aria-pressed={state.mode==='narration'} onClick={()=>a.command('thuyết minh')}>
            <span aria-hidden="true">≋</span><span>Thuyết minh</span>
          </button>
        </div>
        <button className="pause-button" onClick={()=>a.command(state.mode==='paused'?'tiếp tục':'tạm dừng')}>
          <span aria-hidden="true">{state.mode==='paused'?'▷':'Ⅱ'}</span> {state.mode==='paused'?'Tiếp tục trợ lý':'Tạm dừng'}
        </button>
        <DiagnosticsPanel data={a.diag}/>
      </footer>
    </div>
  </main>;
}
