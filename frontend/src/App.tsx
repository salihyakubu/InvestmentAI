import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import ProtectedRoute from './components/auth/ProtectedRoute';
import Login from './pages/Login';

// Dev-only visual harness; statically false in production builds, so the
// route and its chunk are eliminated from the bundle.
const DevPreview = import.meta.env.DEV
  ? lazy(() => import('./pages/DevPreview'))
  : null;
import Dashboard from './pages/Dashboard';
import Portfolio from './pages/Portfolio';
import Trading from './pages/Trading';
import RiskManagement from './pages/RiskManagement';
import MLModels from './pages/MLModels';
import Backtesting from './pages/Backtesting';
import AuditLog from './pages/AuditLog';
import Settings from './pages/Settings';

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      {DevPreview && (
        <Route
          path="/dev-preview"
          element={
            <Suspense fallback={null}>
              <DevPreview />
            </Suspense>
          }
        />
      )}
      <Route element={<ProtectedRoute />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/trading" element={<Trading />} />
        <Route path="/risk" element={<RiskManagement />} />
        <Route path="/models" element={<MLModels />} />
        <Route path="/backtesting" element={<Backtesting />} />
        <Route path="/audit" element={<AuditLog />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
