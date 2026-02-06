import React from 'react';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <p>
        Built with the{' '}
        <a
          href="https://github.com/CAMMA-public/MVOR"
          target="_blank"
          rel="noreferrer"
        >
          MVOR dataset
        </a>{' '}
        &middot; Architecture inspired by{' '}
        <a
          href="https://arxiv.org/abs/2502.13883"
          target="_blank"
          rel="noreferrer"
        >
          PreViPS
        </a>{' '}
        &middot; CAMMA, University of Strasbourg
      </p>
    </footer>
  );
}
