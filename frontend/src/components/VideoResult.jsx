import React from 'react';
import { ArrowLeft, Clock, BarChart3 } from 'lucide-react';
import './VideoResult.css';

const PHASE_COLORS = ['#94a3b8', '#3b82f6', '#ef4444', '#22c55e'];

export default function VideoResult({ result, onReset }) {
  const { timeline, summary, total_frames, inference_time_ms } = result;

  return (
    <div className="video-result">
      <button className="back-btn" onClick={onReset}>
        <ArrowLeft size={16} /> Analyze another
      </button>

      {/* Summary cards */}
      <div className="summary-cards">
        {Object.entries(summary).map(([name, data]) => (
          <div key={name} className="summary-card">
            <div className="summary-card-header">
              <div
                className="summary-dot"
                style={{ background: data.color }}
              />
              <span className="summary-name">{name}</span>
            </div>
            <div className="summary-pct">{data.percentage}%</div>
            <div className="summary-bar-bg">
              <div
                className="summary-bar-fill"
                style={{
                  width: `${data.percentage}%`,
                  background: data.color,
                }}
              />
            </div>
            <div className="summary-count">
              {data.frame_count} / {total_frames} frames
            </div>
          </div>
        ))}
      </div>

      {/* Timeline ribbon */}
      <div className="timeline-section">
        <h3>
          <BarChart3 size={18} /> Phase Timeline
        </h3>
        <div className="timeline-ribbon">
          {timeline.map((entry, idx) => (
            <div
              key={idx}
              className="timeline-cell"
              style={{ background: entry.color }}
              title={`Frame ${entry.frame_index}: ${entry.phase_name} (${(entry.confidence * 100).toFixed(0)}%)`}
            />
          ))}
        </div>
        <div className="timeline-axis">
          <span>Start</span>
          <span>End</span>
        </div>

        {/* Legend */}
        <div className="timeline-legend">
          {Object.entries(summary).map(([name, data]) => (
            <div key={name} className="legend-item">
              <div
                className="legend-dot"
                style={{ background: data.color }}
              />
              <span>{name}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Transition log */}
      <div className="transitions-section">
        <h3>Phase Transitions</h3>
        <div className="transitions-list">
          {(() => {
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
            if (transitions.length === 0) {
              return (
                <p className="no-transitions">
                  No phase transitions detected.
                </p>
              );
            }
            return transitions.map((t, idx) => (
              <div key={idx} className="transition-row">
                <span className="transition-time">
                  {t.time.toFixed(1)}s
                </span>
                <span
                  className="transition-badge"
                  style={{ background: t.from.color }}
                >
                  {t.from.phase_name}
                </span>
                <span className="transition-arrow">→</span>
                <span
                  className="transition-badge"
                  style={{ background: t.to.color }}
                >
                  {t.to.phase_name}
                </span>
              </div>
            ));
          })()}
        </div>
      </div>

      <div className="meta-row">
        <span>
          <Clock size={14} /> {inference_time_ms.toFixed(0)} ms total
        </span>
        <span>{total_frames} frames analyzed</span>
      </div>
    </div>
  );
}
