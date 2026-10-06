import React, { useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = 'https://aman-jordan-breastvision-api.onrender.com'

function App() {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [explain, setExplain] = useState(true)

  const status = busy ? 'ANALYZING' : result ? 'ANALYSIS COMPLETE' : 'SYSTEM READY'

  const confidence = useMemo(() => {
  if (!result) return 0
  return Number(result.prediction?.probability ?? 0)
}, [result])

  function chooseImage(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError('')
  }

  function reset() {
    setFile(null); setPreview(''); setResult(null); setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  async function analyze() {
    if (!file) return
    setBusy(true); setError(''); setResult(null)
    const fd = new FormData()
    fd.append('image', file)
    fd.append('explain', String(explain))
    try {
      const r = await fetch(`${API}/api/analyze`, { method: 'POST', body: fd })
      const data = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(data.error || data.message || `Server returned ${r.status}`)
      setResult(data)
    } catch (e) {
      setError(e.message || 'Unable to reach the analysis server.')
    } finally {
      setBusy(false)
    }
  }

  const prediction = result?.prediction?.class ?? result?.class_name ?? result?.label ?? '—'
  const probs = result?.prediction?.probabilities || {}
  const heatmap = result?.prediction?.explainability?.overlay || null

  return (
    <main className="app-shell">
      <div className="ambient a1"/><div className="ambient a2"/>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><span>BV</span></div>
          <div><div className="brand-name">BREAST<span>VISION</span> AI</div><div className="brand-sub">ULTRASOUND RESEARCH CONSOLE · v2.0</div></div>
        </div>
        <div className="system-meta">
          <div className="live-dot"/><span>{status}</span><b>●</b><span>LOCAL INFERENCE</span>
        </div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow">AI-ASSISTED IMAGE CLASSIFICATION</div>
          <h1>Breast ultrasound<br/><em>analysis console.</em></h1>
          <p>Upload a BUSI-style ultrasound image and run the trained EfficientNet-B0 research model. Designed for academic experimentation — not clinical diagnosis.</p>
          <div className="trust-row"><span>ENCRYPTED IN MEMORY</span><span>•</span><span>NO UPLOAD STORAGE</span><span>•</span><span>RESEARCH MODE</span></div>
        </div>
        <div className="hero-orbit">
          <div className="orbit o1"/><div className="orbit o2"/><div className="orbit o3"/>
          <div className="core"><div className="core-cross"/><div className="core-pulse"/></div>
          <div className="orbit-label l1">3-CLASS</div><div className="orbit-label l2">EFFICIENTNET</div>
        </div>
      </section>

      <section className="workspace">
        <aside className="rail">
          <div className="rail-title">01 / INPUT</div>
          <button className="dropzone" onClick={() => inputRef.current?.click()}>
            {preview ? <img src={preview} alt="Selected ultrasound"/> : <div className="upload-symbol"><span>+</span></div>}
            <div className="drop-copy">
              <strong>{file ? file.name : 'SELECT ULTRASOUND'}</strong>
              <small>{file ? `${(file.size/1024/1024).toFixed(2)} MB · READY` : 'JPG / JPEG / PNG · MAX 10 MB'}</small>
            </div>
          </button>
          <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" onChange={chooseImage} hidden/>
          <div className="controls">
            <label className="toggle-row"><span>Generate Grad-CAM</span><input type="checkbox" checked={explain} onChange={e=>setExplain(e.target.checked)}/><i/></label>
          </div>
          <button className="primary" disabled={!file || busy} onClick={analyze}>
            <span>{busy ? 'PROCESSING' : 'RUN ANALYSIS'}</span><b>↗</b>
          </button>
          {file && <button className="ghost" onClick={reset}>CLEAR CURRENT SCAN</button>}
          {error && <div className="error"><b>CONNECTION / MODEL ERROR</b><span>{error}</span></div>}
        </aside>

        <section className="stage">
          <div className="stage-head"><div><span>02 / IMAGING FIELD</span><strong>{preview ? 'SPECIMEN PREVIEW' : 'AWAITING INPUT'}</strong></div><span className="scan-id">{file ? 'SCAN READY' : 'NO ACTIVE SCAN'}</span></div>
          <div className="imaging">
            <div className="gridlines"/><div className="scan-beam"/>
            {!preview && <div className="empty-core"><div className="crosshair"/><span>UPLOAD AN IMAGE TO BEGIN</span></div>}
            {preview && <img className="scan-image" src={preview} alt="Ultrasound preview"/>}
            <div className="hud top-left">GAIN <b>72</b><br/>DEPTH <b>5.0 CM</b></div>
            <div className="hud top-right">MODE <b>B-MODE</b><br/>FRAME <b>LIVE</b></div>
            <div className="hud bottom-left">X <b>000.42</b><br/>Y <b>000.68</b></div>
            <div className="hud bottom-right">MODEL <b>EF-B0</b><br/>EXPLAIN <b>{explain ? 'ON' : 'OFF'}</b></div>
          </div>
        </section>

        <aside className="result-panel">
          <div className="rail-title">03 / MODEL OUTPUT</div>
          <div className={`result-hero ${result ? 'has-result' : ''}`}>
            <span>PRIMARY CLASS</span>
           <strong>{result ? String(typeof prediction === 'object' ? (prediction.label ?? prediction.class_name ?? prediction.name ?? 'UNKNOWN') : prediction).toUpperCase() : '—'}</strong>
            <small>{result ? 'MODEL PREDICTION' : 'RUN ANALYSIS TO GENERATE'}</small>
          </div>
          <div className="confidence">
            <div><span>CONFIDENCE</span><b>{result ? `${(confidence <= 1 ? confidence*100 : confidence).toFixed(1)}%` : '—'}</b></div>
            <div className="meter"><i style={{width: `${Math.min(100, confidence <= 1 ? confidence*100 : confidence)}%`}}/></div>
          </div>
          <div className="prob-list">
            {['malignant','benign','normal'].map(k => {
              const v = Number(probs[k] ?? probs[k[0].toUpperCase()+k.slice(1)] ?? 0)
              const pct = v <= 1 ? v*100 : v
              return <div className="prob" key={k}><span>{k}</span><div><i style={{width:`${Math.min(100,pct)}%`}}/></div><b>{pct.toFixed(1)}%</b></div>
            })}
          </div>
          {heatmap && <div className="explain"><span>GRAD-CAM ATTENTION</span><img src={heatmap} alt="Grad-CAM explanation"/></div>}
          <div className="disclaimer">Research prototype. This model output is not a medical diagnosis and must not be used alone for clinical decisions.</div>
        </aside>
      </section>

      <footer>
        <span>BREASTVISION AI</span><span>BUSI · 3 CLASSES</span><span>PYTORCH · EFFICIENTNET-B0</span><span>FLASK API · LOCAL</span><span>© 2026 RESEARCH PROTOTYPE</span>
      </footer>
    </main>
  )
}

createRoot(document.getElementById('root')).render(<App />)
