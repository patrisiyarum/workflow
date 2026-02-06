import React from 'react';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <a
        href="https://github.com/CAMMA-public/MVOR"
        target="_blank"
        rel="noreferrer"
      >
        MVOR
      </a>
      <span className="sep">&middot;</span>
      <a
        href="https://arxiv.org/abs/2502.13883"
        target="_blank"
        rel="noreferrer"
      >
        PreViPS
      </a>
      <span className="sep">&middot;</span>
      <span>CAMMA, Strasbourg</span>
    </footer>
  );
}
