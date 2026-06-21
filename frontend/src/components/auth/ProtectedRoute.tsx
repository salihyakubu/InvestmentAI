import { Navigate, Outlet } from 'react-router-dom';
import { useAppStore } from '../../store';
import Layout from '../layout/Layout';

/**
 * Gate for authenticated routes: without a token, bounce to /login; otherwise
 * render the app shell (Layout) around the matched child route.
 */
export default function ProtectedRoute() {
  const token = useAppStore((s) => s.token);
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  return (
    <Layout>
      <Outlet />
    </Layout>
  );
}
