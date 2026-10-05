import { useEffect, useMemo, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000/api";

function App() {
  return (
    <div className="app">
      <AmbientBackground />
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analyze" element={<Analyze />} />
        <Route path="/predict" element={<Analyze />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/about" element={<About />} />
      </Routes>
      <Footer />
    </div>
  );
}

function AmbientBackground() {
  return (
    <div className="ambient" aria-hidden="true">
      <div className="orb orb-a" />
      <div className="orb orb-b" />
      <div className="grid-glow" />
      <div className="scan-beam" />
    </div>
  );
}

function Navbar() {
  const links = [
    ["/", "Overview"],
    ["/analyze", "Analyze"],
    ["/dashboard", "Research"],
    ["/about", "System"],
  ];

  return (
    <header className="nav-wrap">
      <nav className="navbar shell">
        <Link to="/" className="brand">
          <span className="brand-mark">
            <span className="pulse-dot" />
            <span className="wave-line" />
          </span>
          <span>
            <strong>BreastVision</strong>
            <small>AI RESEARCH LAB</small>
          </span>
        </Link>

        <div className="nav-links">
          {links.map(([to, label]) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            >
              {label}
            </NavLink>
          ))}
        </div>

        <Link to="/analyze" className="nav-cta">
          <span>Run analysis</span>
          <span>↗</span>
        </Link>
      </nav>
    </header>
  );
}

function Home() {
  return (
    <main>
      <section className="hero shell">
        <div className="hero-copy reveal">
          <div className="eyebrow">
            <span className="status-dot" />
            ACADEMIC MEDICAL AI · RESEARCH PROTOTYPE
          </div>

          <h1>
            See beyond the
            <span className="gradient-text"> ultrasound.</span>
          </h1>

          <p className="hero-lead">
            A deep-learning research platform for classifying breast ultrasound
            images into <strong>benign, malignant, or normal</strong> classes
            using EfficientNet-B0 and optional Grad-CAM explainability.
          </p>

          <div className="hero-actions">
            <Link to="/analyze" className="btn btn-primary">
              <span className="btn-icon">⌁</span>
              Analyze an ultrasound
              <span>→</span>
            </Link>
            <Link to="/dashboard" className="btn btn-ghost">
              Explore model performance
            </Link>
          </div>

          <div className="hero-trust">
            <span>BUSI DATASET</span>
            <i />
            <span>PYTORCH</span>
            <i />
            <span>EFFICIENTNET-B0</span>
            <i />
            <span>GRAD-CAM</span>
          </div>
        </div>

        <div className="hero-visual reveal delay-1">
          <UltrasoundOrb />
          <div className="floating-card card-signal">
            <span className="mini-label">LIVE INFERENCE</span>
            <strong>MODEL READY</strong>
            <span className="signal"><i /><i /><i /><i /><i /><i /><i /></span>
          </div>
          <div className="floating-card card-class">
            <span className="mini-label">3-CLASS OUTPUT</span>
            <div><b>BENIGN</b><span>•</span><em>MALIGNANT</em><span>•</span><small>NORMAL</small></div>
          </div>
        </div>
      </section>

      <section className="metrics-strip shell">
        <MetricTile value="03" label="Image classes" accent />
        <MetricTile value="224×224" label="Model input" />
        <MetricTile value="B0" label="EfficientNet backbone" />
        <MetricTile value="7×7" label="Grad-CAM resolution" />
      </section>

      <section className="section shell">
        <SectionHeading
          kicker="THE PIPELINE"
          title="From scan to insight."
          text="A transparent workflow keeps the machine-learning pipeline visible instead of hiding it behind a single prediction button."
        />

        <div className="pipeline">
          {[
            ["01", "Upload", "Validate the ultrasound image in memory."],
            ["02", "Preprocess", "RGB conversion, 224×224 resize and ImageNet normalization."],
            ["03", "Infer", "EfficientNet-B0 produces three class probabilities."],
            ["04", "Explain", "Optional Grad-CAM visualizes model attention."],
          ].map(([num, title, text], i) => (
            <div className="pipeline-card" key={num}>
              <span className="pipeline-num">{num}</span>
              <div className="pipeline-icon">{["↥", "◈", "◎", "✦"][i]}</div>
              <h3>{title}</h3>
              <p>{text}</p>
              {i < 3 && <span className="pipeline-arrow">→</span>}
            </div>
          ))}
        </div>
      </section>

      <section className="research-band">
        <div className="shell research-grid">
          <div>
            <div className="eyebrow">WHY THIS INTERFACE</div>
            <h2>Designed like a research instrument, not a form.</h2>
          </div>
          <div className="research-points">
            <ResearchPoint title="Explainable" text="Grad-CAM can be displayed beside the original image." />
            <ResearchPoint title="Measured" text="Evaluation metrics come from the held-out test set." />
            <ResearchPoint title="Responsible" text="Outputs are presented as model classifications, not medical diagnoses." />
          </div>
        </div>
      </section>

      <section className="section shell">
        <SectionHeading kicker="MODEL STACK" title="The technology behind the interface." />
        <div className="tech-grid">
          {[
            ["PYTORCH", "Deep-learning framework", "01"],
            ["EFFICIENTNET-B0", "Transfer-learning classifier", "02"],
            ["FLASK", "Inference REST API", "03"],
            ["REACT + VITE", "Interactive research interface", "04"],
          ].map(([title, text, n]) => (
            <div className="tech-card" key={title}>
              <span>{n}</span>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

function Analyze() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [explain, setExplain] = useState(true);
  const [dragging, setDragging] = useState(false);

  const chooseFile = (selected) => {
    const next = selected?.[0];
    if (!next) return;
    if (!["image/png", "image/jpeg"].includes(next.type)) {
      setError("Please choose a PNG or JPEG ultrasound image.");
      return;
    }
    if (next.size > 10 * 1024 * 1024) {
      setError("The image must be 10 MB or smaller.");
      return;
    }
    setFile(next);
    setPreview(URL.createObjectURL(next));
    setResult(null);
    setError("");
  };

  const analyze = async () => {
    if (!file) {
      setError("Select an ultrasound image first.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const body = new FormData();
      body.append("image", file);
      body.append("explain", String(explain));

      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body,
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.success) {
        throw new Error(data?.error?.message || "The analysis server returned an error.");
      }
      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Could not reach the Flask analysis server. Make sure the backend is running on port 5000."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page shell">
      <PageHero
        kicker="ANALYSIS LAB"
        title="Run an ultrasound through the model."
        text="Upload one ultrasound image. The interface sends it to the existing Flask inference API and presents the returned classification and explainability."
      />

      <div className="analysis-layout">
        <section className="lab-panel">
          <div className="panel-top">
            <div>
              <span className="mini-label">INPUT CHANNEL</span>
              <h2>Ultrasound image</h2>
            </div>
            <span className="secure-pill">● IN-MEMORY</span>
          </div>

          {!file ? (
            <label
              className={`dropzone ${dragging ? "dragging" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                chooseFile(e.dataTransfer.files);
              }}
            >
              <input
                type="file"
                accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                onChange={(e) => chooseFile(e.target.files)}
              />
              <div className="upload-ring"><span>↥</span></div>
              <h3>Drop ultrasound image here</h3>
              <p>PNG or JPEG · maximum 10 MB</p>
              <span className="browse-btn">Browse image</span>
              <small>Images are validated and processed in memory.</small>
            </label>
          ) : (
            <div className="selected-image">
              <div className="image-frame">
                <img src={preview} alt="Selected ultrasound" />
                <div className="scan-line" />
                {loading && <div className="processing-badge">SCANNING · AI INFERENCE</div>}
              </div>
              <div className="selected-meta">
                <div>
                  <span className="mini-label">SELECTED IMAGE</span>
                  <strong>{file.name}</strong>
                  <small>{(file.size / 1024 / 1024).toFixed(2)} MB</small>
                </div>
                <button className="text-button" onClick={() => { setFile(null); setPreview(""); setResult(null); }}>
                  Replace
                </button>
              </div>
            </div>
          )}

          <div className="analysis-controls">
            <label className="toggle">
              <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} />
              <span className="toggle-track"><i /></span>
              <span>
                <strong>Generate Grad-CAM</strong>
                <small>Show an attention overlay with the result</small>
              </span>
            </label>

            <button className="btn btn-primary analyze-button" disabled={loading} onClick={analyze}>
              {loading ? "Analyzing…" : "Start AI analysis"}
              <span>{loading ? "◌" : "→"}</span>
            </button>
          </div>

          {error && <div className="error-box">⚠ {error}</div>}
        </section>

        <section className="lab-side">
          <div className="live-card">
            <div className="live-header">
              <span><i className="status-dot" /> SYSTEM STATUS</span>
              <span>LOCAL</span>
            </div>
            <div className="live-main">
              <div className="radar"><div /><div /><div /></div>
              <div>
                <strong>Inference endpoint</strong>
                <p>EfficientNet-B0</p>
              </div>
            </div>
            <div className="live-row"><span>Input</span><b>224 × 224 RGB</b></div>
            <div className="live-row"><span>Classes</span><b>03</b></div>
            <div className="live-row"><span>Explainability</span><b>Grad-CAM</b></div>
          </div>

          <div className="class-card">
            <span className="mini-label">OUTPUT SPACE</span>
            <ClassDot name="Benign" type="benign" />
            <ClassDot name="Malignant" type="malignant" />
            <ClassDot name="Normal" type="normal" />
          </div>
        </section>
      </div>

      {loading && (
        <section className="processing-section">
          <div className="processing-head">
            <span className="eyebrow"><span className="status-dot" /> COMPUTATION IN PROGRESS</span>
            <span>EfficientNet-B0</span>
          </div>
          <div className="process-track">
            {["Image validation", "Preprocessing", "Model inference", "Grad-CAM"].map((x, i) => (
              <div key={x} className="process-step active">
                <span>{i + 1}</span><b>{x}</b>
              </div>
            ))}
          </div>
        </section>
      )}

      {result && <ResultPanel result={result} preview={preview} />}
    </main>
  );
}

function ResultPanel({ result, preview }) {
  const prediction = result.prediction || {};
  const probabilities = prediction.probabilities || {};
  const ordered = ["Benign", "Malignant", "Normal"].map((name) => ({
    name,
    value: Number(probabilities[name] ?? 0),
  }));
  const confidence = Number(prediction.probability ?? Math.max(...ordered.map((x) => x.value), 0));
  const klass = prediction.class || "Unknown";
  const overlay = prediction.explainability?.overlay;

  return (
    <section className="result-section">
      <div className="result-banner">
        <div>
          <span className="eyebrow"><span className="status-dot" /> ANALYSIS COMPLETE</span>
          <h2>Model classification</h2>
        </div>
        <span className={`result-chip ${klass.toLowerCase()}`}>{klass}</span>
      </div>

      <div className="result-grid">
        <div className="result-image-card">
          <div className="image-tabs"><span className="active">ORIGINAL</span>{overlay && <span>GRAD-CAM AVAILABLE</span>}</div>
          <img src={preview} alt="Analyzed ultrasound" />
        </div>

        <div className="result-main-card">
          <span className="mini-label">PREDICTED CLASS</span>
          <div className={`big-result ${klass.toLowerCase()}`}>
            {klass}
            <span>{formatPercent(confidence)}</span>
          </div>
          <div className="confidence-bar"><i style={{ width: `${Math.min(confidence * 100, 100)}%` }} /></div>

          <div className="prob-list">
            {ordered.map((item) => (
              <div className="prob-row" key={item.name}>
                <div><span className={`legend-dot ${item.name.toLowerCase()}`} />{item.name}</div>
                <b>{formatPercent(item.value)}</b>
                <div className="prob-track"><i style={{ width: `${Math.min(item.value * 100, 100)}%` }} /></div>
              </div>
            ))}
          </div>

          <div className="model-foot">
            <span>MODEL <b>{result.model?.name || "EfficientNet-B0"}</b></span>
            <span>MODE <b>RESEARCH</b></span>
          </div>
        </div>
      </div>

      {overlay && (
        <div className="gradcam-card">
          <div className="gradcam-copy">
            <span className="eyebrow">EXPLAINABILITY</span>
            <h3>Where did the model look?</h3>
            <p>
              Grad-CAM provides a visual indication of regions contributing
              to the selected model output. It is an explanation aid, not a
              lesion segmentation or clinical conclusion.
            </p>
          </div>
          <div className="gradcam-image">
            <img src={overlay} alt="Grad-CAM model attention overlay" />
          </div>
        </div>
      )}

      <div className="research-disclaimer">
        <span>ⓘ</span>
        <p><strong>Research / educational prototype.</strong> This output is a model classification on a public dataset, not a medical diagnosis or treatment recommendation.</p>
      </div>
    </section>
  );
}

function Dashboard() {
  const [data, setData] = useState(null);
  const [info, setInfo] = useState(null);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/model-metrics`).then((r) => r.json()),
      fetch(`${API_BASE}/model-info`).then((r) => r.json()),
      fetch(`${API_BASE}/health`).then((r) => r.json()),
    ])
      .then(([metrics, model, h]) => {
        setData(metrics);
        setInfo(model);
        setHealth(h);
      })
      .catch((e) => setError(e.message));
  }, []);

  const metrics = normalizeMetrics(data?.metrics || data?.data?.metrics || {});
  const dataset = data?.dataset || data?.data?.dataset || info?.dataset || {};
  const classCounts = extractCounts(dataset);
  const total = classCounts.reduce((s, x) => s + x.value, 0);

  return (
    <main className="page shell">
      <PageHero
        kicker="RESEARCH CONSOLE"
        title="Model performance, without the black box."
        text="A visual research dashboard for the held-out evaluation artifacts produced by the project."
      />

      {error && <div className="error-box">⚠ Could not load model metrics. Make sure the Flask backend is running.</div>}

      <div className="dashboard-status">
        <span><i className={`status-dot ${health?.model_loaded ? "" : "offline"}`} /> API {health?.model_loaded ? "MODEL READY" : "CHECK BACKEND"}</span>
        <span>ARCHITECTURE · {info?.model?.name || info?.name || "EfficientNet-B0"}</span>
        <span>DATASET · BUSI</span>
      </div>

      <section className="metric-grid">
        {[
          ["Accuracy", metrics.accuracy, "held-out test set"],
          ["Precision", metrics.precision, "macro average"],
          ["Recall", metrics.recall, "sensitivity"],
          ["Specificity", metrics.specificity, "macro average"],
          ["F1-score", metrics.f1, "macro average"],
          ["ROC-AUC", metrics.rocAuc, "one-vs-rest"],
        ].map(([label, value, sub]) => (
          <MetricTile key={label} value={value == null ? "—" : formatPercent(value)} label={label} sub={sub} large />
        ))}
      </section>

      <div className="dashboard-grid">
        <section className="data-card">
          <div className="card-heading"><div><span className="mini-label">CLASS DISTRIBUTION</span><h3>Dataset composition</h3></div><span>BUSI</span></div>
          <div className="distribution">
            {classCounts.length ? classCounts.map((item) => (
              <div className="dist-row" key={item.name}>
                <div><span className={`legend-dot ${item.name.toLowerCase()}`} /><b>{item.name}</b><small>{item.value} images</small></div>
                <div className="dist-track"><i style={{ width: `${total ? item.value / total * 100 : 0}%` }} /></div>
              </div>
            )) : <p className="muted">Dataset statistics will appear here when returned by the API.</p>}
          </div>
        </section>

        <section className="data-card">
          <div className="card-heading"><div><span className="mini-label">MODEL CARD</span><h3>Inference configuration</h3></div><span className="green-text">READY</span></div>
          <div className="model-table">
            <InfoRow k="Architecture" v={info?.model?.name || info?.name || "EfficientNet-B0"} />
            <InfoRow k="Input size" v="224 × 224" />
            <InfoRow k="Classes" v="Benign · Malignant · Normal" />
            <InfoRow k="Framework" v="PyTorch" />
            <InfoRow k="Explainability" v="Grad-CAM" />
            <InfoRow k="API" v="Flask REST" />
          </div>
        </section>
      </div>

      <section className="method-card">
        <div className="method-copy">
          <span className="eyebrow">EVALUATION NOTE</span>
          <h2>What these numbers mean.</h2>
          <p>
            These metrics describe benchmark performance on the held-out test
            set used by this project. They are not clinical accuracy, and the
            dashboard should not be used to make treatment decisions.
          </p>
          <Link to="/about" className="text-link">Read the methodology →</Link>
        </div>
        <div className="metric-radar">
          {["Accuracy", "Precision", "Recall", "F1"].map((x, i) => (
            <div key={x} style={{ "--delay": `${i * 120}ms` }}><span>{x}</span><i /></div>
          ))}
        </div>
      </section>
    </main>
  );
}

function About() {
  return (
    <main className="page shell">
      <PageHero
        kicker="SYSTEM ARCHITECTURE"
        title="Built as an end-to-end medical AI prototype."
        text="The interface is the final layer of a reproducible pipeline spanning data preparation, transfer learning, evaluation, inference and explainability."
      />

      <div className="architecture">
        {[
          ["01", "BUSI DATASET", "Public breast ultrasound images organized into three classes."],
          ["02", "PREPROCESSING", "RGB conversion, resize to 224×224 and ImageNet normalization."],
          ["03", "EFFICIENTNET-B0", "ImageNet-pretrained CNN fine-tuned for three-class classification."],
          ["04", "EVALUATION", "Accuracy, precision, recall, specificity, F1, ROC-AUC and confusion matrix."],
          ["05", "FLASK API", "Loads the saved model once and exposes the inference endpoint."],
          ["06", "REACT INTERFACE", "Upload, prediction visualization and research dashboard."],
        ].map(([n, title, text]) => (
          <div className="arch-node" key={n}>
            <span>{n}</span>
            <div className="arch-line" />
            <h3>{title}</h3>
            <p>{text}</p>
          </div>
        ))}
      </div>

      <section className="about-split">
        <div className="about-card">
          <span className="eyebrow">DATASET CLASSES</span>
          <h2>Three possible outputs.</h2>
          <p className="muted">These names describe the class labels used by the BUSI dataset.</p>
          <ClassExplanation name="Benign" text="A non-cancerous lesion class in the dataset." type="benign" />
          <ClassExplanation name="Malignant" text="A cancerous lesion class in the dataset." type="malignant" />
          <ClassExplanation name="Normal" text="The normal-class ultrasound images in the dataset." type="normal" />
        </div>

        <div className="about-card dark-card">
          <span className="eyebrow">RESPONSIBLE AI</span>
          <h2>Useful, but deliberately limited.</h2>
          <ul className="clean-list">
            <li>Model output is presented as a classification, not a diagnosis.</li>
            <li>Grad-CAM shows model attention, not exact lesion location.</li>
            <li>Softmax probabilities are not calibrated medical risk.</li>
            <li>The project has no external clinical validation.</li>
            <li>Uploaded images are processed in memory by the backend.</li>
          </ul>
        </div>
      </section>
    </main>
  );
}

function UltrasoundOrb() {
  return (
    <div className="ultrasound-stage">
      <div className="ultrasound-grid" />
      <div className="ultrasound-circle c1" />
      <div className="ultrasound-circle c2" />
      <div className="ultrasound-circle c3" />
      <div className="scan-cone">
        <span className="tissue t1" />
        <span className="tissue t2" />
        <span className="tissue t3" />
        <span className="tissue t4" />
      </div>
      <div className="radial-line r1" />
      <div className="radial-line r2" />
      <div className="radial-line r3" />
      <div className="target-ring"><span /></div>
      <div className="scan-caption">ULTRASOUND / AI ANALYSIS</div>
    </div>
  );
}

function SectionHeading({ kicker, title, text }) {
  return (
    <div className="section-heading">
      <span className="eyebrow">{kicker}</span>
      <h2>{title}</h2>
      {text && <p>{text}</p>}
    </div>
  );
}

function PageHero({ kicker, title, text }) {
  return (
    <section className="page-hero reveal">
      <div className="eyebrow"><span className="status-dot" /> {kicker}</div>
      <h1>{title}</h1>
      <p>{text}</p>
    </section>
  );
}

function MetricTile({ value, label, sub, accent, large }) {
  return (
    <div className={`metric-tile ${accent ? "accent" : ""} ${large ? "large" : ""}`}>
      <strong>{value}</strong>
      <span>{label}</span>
      {sub && <small>{sub}</small>}
    </div>
  );
}

function ResearchPoint({ title, text }) {
  return <div className="research-point"><span>✦</span><div><h3>{title}</h3><p>{text}</p></div></div>;
}

function ClassDot({ name, type }) {
  return <div className="class-dot-row"><span className={`legend-dot ${type}`} /><b>{name}</b><span>classification</span></div>;
}

function ClassExplanation({ name, text, type }) {
  return <div className="class-explanation"><span className={`legend-dot ${type}`} /><div><strong>{name}</strong><p>{text}</p></div></div>;
}

function InfoRow({ k, v }) {
  return <div className="info-row"><span>{k}</span><b>{v}</b></div>;
}

function Footer() {
  return (
    <footer className="footer">
      <div className="shell footer-inner">
        <div>
          <strong>BreastVision AI</strong>
          <span>Academic medical-AI research prototype</span>
        </div>
        <div className="footer-note">NOT A MEDICAL DEVICE · NOT FOR CLINICAL DECISIONS</div>
      </div>
    </footer>
  );
}

function normalizeMetrics(raw) {
  const read = (...keys) => {
    for (const key of keys) {
      const v = raw?.[key];
      if (typeof v === "number") return v;
      if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) return Number(v);
    }
    return null;
  };
  return {
    accuracy: read("accuracy", "test_accuracy"),
    precision: read("precision", "macro_precision"),
    recall: read("recall", "sensitivity", "macro_recall"),
    specificity: read("specificity", "macro_specificity"),
    f1: read("f1", "f1_score", "macro_f1"),
    rocAuc: read("roc_auc", "rocAUC", "roc_auc_macro", "auc"),
  };
}

function extractCounts(dataset) {
  const counts = dataset?.class_counts || dataset?.counts || dataset?.distribution || dataset?.classes;
  if (!counts) return [];
  if (Array.isArray(counts)) {
    return counts.map((x) => ({ name: x.class || x.name || "Class", value: Number(x.count || x.value || 0) }));
  }
  return Object.entries(counts).map(([name, value]) => ({ name, value: Number(value) || 0 }));
}

function formatPercent(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const percent = n <= 1 ? n * 100 : n;
  return `${percent.toFixed(1)}%`;
}

export default App;
