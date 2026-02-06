import React from 'react';
import { ArrowLeft } from 'lucide-react';
import './VideoResult.css';

export default function VideoResult({ result, onReset }) {
  const { timeline, summary, total_frames, inference_time_ms } = result;

  const transitions = [];
  for (let i = 1; i < timeline.length; i++) {
    if (timeline[i].phase_id !== timeline[i - 1].phase_id) {
      transitions.push({
        frame: timeline[i].frame_index,
        time: timeline[i].time_seconds,
        from: timeline[i - 1],
        to: timeline[i],
      });
    }
  }

  return (
    <div className="video-result">
      <button className="back-btn" onClick={onReset}>
        <ArrowLeft size={14} /> New analysis
      </button>

      <div className="summary-row">
        {Object.entries(summary).map(([name, data]) => (
          <div key={name} className="summary-item">
            <div className="summary-label">
              <span className="dot" style={{ background: data.color }} />
              {name}
            </div>
            <span className="summary-value">{data.percentage}%</span>
          </div>
        ))}
      </div>

      <div className="section">
        <h3>Timeline</h3>
        <div className="timeline-ribbon">
          {timeline.map((entry, idx) => (
            <div
              key={idx}
              className="timeline-cell"
              style={{ background: entry.color }}
              title={`${entry.phase_name} (${(entry.confidence * 100).toFixed(0)}%)`}
            />
          ))}
        </div>
        <div className="timeline-axis">
          <span>Start</span>
          <span>End</span>
        </div>
      </div>

      {transitions.length > 0 && (
        <div className="section">
          <h3>Transitions</h3>
          <div className="transition-list">
            {transitions.map((t, idx) => (
              <div key={idx} className="transition-row">
                <span className="transition-time">{t.time.toFixed(1)}s</span>
                <span className="transition-badge" style={{ background: t.from.color }}>
                  {t.from.phase_name}
                </span>
                <span className="transition-arrow">&rarr;</span>
                <span className="transition-badge" style={{ background: t.to.color }}>
                  {t.to.phase_name}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <span className="result-meta">
        {total_frames} frames &middot; {inference_time_ms.toFixed(0)} ms
      </span>
    </div>
  );
}
