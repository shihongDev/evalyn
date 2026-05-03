import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/index.css';
import App from './App.tsx';

// Initial theme is applied by <App>'s tweak effect; set a sane default
// before the first paint to avoid a flash of dark on light deployments.
document.body.dataset.theme = 'light';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
