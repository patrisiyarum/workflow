import React from 'react';
import { ArrowLeft } from 'lucide-react';
import './ImageResult.css';

export default function ImageResult({ result, previewUrl, onReset }) {
  const { phase, all_phases, inference_time_ms } = result;

  return (
    <div className="image-result">
      <button className="back-btn" onClick={onReset}>
        <ArrowLeft size={14} /> New analysis
      </button>

      <div className="result-grid">
        <div className="result-preview">
          {previewUrl && <img src={previewUrl} alt="Uploaded" />}
          <span className="result-badge" style={{ background: phase.color }}>
            {phase.phase_name}
          </span>
        </div>

        <div className="result-details">
          <div className="result-phase" style={{ borderLeftColor: phase.color }}>
            <h2>{phase.phase_name}</h2>
            <p>{phase.description}</p>
          </div>

          <div className="result-confidence">
            <div className="conf-header">
              <span>Confidence</span>
              <span className="conf-value">{(phase.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="conf-bar">
              <div
                className="conf-fill"
                style={{ width: `${phase.confidence * 100}%`, background: phase.color }}
              />
            </div>
          </div>

          <div className="result-phases">
            {all_phases.map((p) => (
              <div key={p.phase_id} className="phase-row">
                <div className="phase-row-name">
                  <span className="dot" style={{ background: p.color }} />
                  {p.phase_name}
                </div>
                <span className="phase-row-pct">
                  {(p.confidence * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>

          <span className="result-meta">{inference_time_ms.toFixed(0)} ms</span>
        </div>
      </div>
    </div>
  );
}
