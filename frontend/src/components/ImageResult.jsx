import React from 'react';
import { ArrowLeft, Clock, Zap } from 'lucide-react';
import './ImageResult.css';

export default function ImageResult({ result, previewUrl, onReset }) {
  const { phase, all_phases, inference_time_ms } = result;

  return (
    <div className="image-result">
      <button className="back-btn" onClick={onReset}>
        <ArrowLeft size={16} /> Analyze another
      </button>

      <div className="result-grid">
        {/* Image preview */}
        <div className="result-preview">
          {previewUrl && <img src={previewUrl} alt="Uploaded OR" />}
          <div className="result-badge" style={{ background: phase.color }}>
            {phase.phase_name}
          </div>
        </div>

        {/* Prediction details */}
        <div className="result-details">
          <div className="predicted-phase" style={{ borderColor: phase.color }}>
            <div
              className="phase-dot"
              style={{ background: phase.color }}
            />
            <div>
              <h2>{phase.phase_name}</h2>
              <p className="phase-desc">{phase.description}</p>
            </div>
          </div>

          <div className="confidence-meter">
            <div className="confidence-label">
              <span>Confidence</span>
              <span className="confidence-value">
                {(phase.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="confidence-bar-bg">
              <div
                className="confidence-bar-fill"
                style={{
                  width: `${phase.confidence * 100}%`,
                  background: phase.color,
                }}
              />
            </div>
          </div>

          <div className="all-phases">
            <h4>All Phases</h4>
            {all_phases.map((p) => (
              <div key={p.phase_id} className="phase-row">
                <div className="phase-row-left">
                  <div
                    className="phase-dot-sm"
                    style={{ background: p.color }}
                  />
                  <span>{p.phase_name}</span>
                </div>
                <div className="phase-row-right">
                  <div className="mini-bar-bg">
                    <div
                      className="mini-bar-fill"
                      style={{
                        width: `${p.confidence * 100}%`,
                        background: p.color,
                      }}
                    />
                  </div>
                  <span className="phase-pct">
                    {(p.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="meta-row">
            <span><Clock size={14} /> {inference_time_ms.toFixed(0)} ms</span>
            <span><Zap size={14} /> ResNet + LSTM</span>
          </div>
        </div>
      </div>
    </div>
  );
}
