import { Routes, Route } from 'react-router-dom';
import ProtectedRoute from './components/auth/ProtectedRoute';
import Login from './pages/Login';
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
