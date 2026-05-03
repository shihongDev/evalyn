import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './v2/styles.css';
import { V2App } from './v2/V2App';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <V2App />
  </StrictMode>,
);
