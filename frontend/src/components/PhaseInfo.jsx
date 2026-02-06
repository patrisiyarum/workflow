import React from 'react';
import { Circle, Users, Stethoscope, PackageOpen } from 'lucide-react';
import './PhaseInfo.css';

const phases = [
  {
    name: 'Idle / Empty',
    color: '#94a3b8',
    icon: Circle,
    desc: 'No people detected in the operating room. Background activity or empty state.',
  },
  {
    name: 'Preparation',
    color: '#3b82f6',
    icon: Users,
    desc: 'Patient positioned, clinicians arriving, equipment being set up.',
  },
  {
    name: 'Procedure Active',
    color: '#ef4444',
    icon: Stethoscope,
    desc: 'Multiple clinicians actively engaged in the surgical intervention.',
  },
  {
    name: 'Closure / Cleanup',
    color: '#22c55e',
    icon: PackageOpen,
    desc: 'Procedure complete. Winding down, cleaning, clinicians departing.',
  },
];

export default function PhaseInfo() {
  return (
    <section className="phase-info">
      <h3>Detectable Phases</h3>
      <div className="phase-cards">
        {phases.map((p) => {
          const Icon = p.icon;
          return (
            <div key={p.name} className="phase-card">
              <div className="phase-card-icon" style={{ color: p.color }}>
                <Icon size={22} />
              </div>
              <div>
                <h4 style={{ color: p.color }}>{p.name}</h4>
                <p>{p.desc}</p>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
