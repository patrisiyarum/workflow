import React from 'react';
import './PhaseInfo.css';

const phases = [
  { name: 'Idle', color: '#555' },
  { name: 'Preparation', color: '#3b82f6' },
  { name: 'Procedure', color: '#ef4444' },
  { name: 'Closure', color: '#22c55e' },
];

export default function PhaseInfo() {
  return (
    <div className="phase-info">
      <span className="phase-info-label">Detects</span>
      {phases.map((p) => (
        <span key={p.name} className="phase-tag">
          <span className="phase-dot" style={{ background: p.color }} />
          {p.name}
        </span>
      ))}
    </div>
  );
}
