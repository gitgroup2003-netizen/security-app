import React from 'react';

export const GitGroupLogo = ({ className = 'w-20 h-20' }: { className?: string }) => (
  <svg viewBox="0 0 100 100" className={className} fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Git Group logo">
    <path d="M50 6 L88 20 V50 C88 72 72 88 50 95 C28 88 12 72 12 50 V20 Z" stroke="white" strokeWidth="3" fill="rgba(255,255,255,0.06)" />
    <path d="M62 36 C58 31 54 29 49 29 C38 29 30 38 30 50 C30 62 38 71 49 71 C57 71 63 67 65 60 V52 H50" stroke="white" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
