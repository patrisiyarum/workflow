import React from 'react';
import './Header.css';

export default function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <span className="header-title">SurgPhase</span>
        <nav className="header-nav">
          <a
            href="https://github.com/CAMMA-public/MVOR"
            target="_blank"
            rel="noreferrer"
          >
            Dataset
          </a>
          <a
            href="https://arxiv.org/abs/2502.13883"
            target="_blank"
            rel="noreferrer"
          >
            Paper
          </a>
        </nav>
      </div>
    </header>
  );
}
