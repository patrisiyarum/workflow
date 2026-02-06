import React from 'react';
import { Activity } from 'lucide-react';
import './Header.css';

export default function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <Activity size={24} className="header-icon" />
          <span className="header-title">SurgPhase AI</span>
        </div>
        <nav className="header-nav">
          <a
            href="https://github.com/CAMMA-public/MVOR"
            target="_blank"
            rel="noreferrer"
          >
            MVOR Dataset
          </a>
          <a
            href="https://arxiv.org/abs/2502.13883"
            target="_blank"
            rel="noreferrer"
          >
            PreViPS Paper
          </a>
        </nav>
      </div>
    </header>
  );
}
