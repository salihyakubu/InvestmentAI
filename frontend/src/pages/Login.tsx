import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import apiClient from '../api/client';
import { useAppStore } from '../store';

interface TokenResponse {
  access_token: string;
  token_type: string;
}

export default function Login() {
  const navigate = useNavigate();
  const setToken = useAppStore((s) => s.setToken);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const { data } = await apiClient.post<TokenResponse>('/auth/token', {
        email,
        password,
      });
      setToken(data.access_token);
      navigate('/', { replace: true });
    } catch (err) {
      if (axios.isAxiosError(err) && err.response?.status === 401) {
        setError('Invalid email or password.');
      } else {
        setError('Login failed. Please try again.');
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-ink flex items-center justify-center px-4">
      <div className="w-full max-w-sm animate-fade-up">
        <div className="mb-10 text-center">
          <div className="mx-auto mb-5 w-14 h-14 rounded-2xl bg-gradient-to-br from-accent to-accent-dark flex items-center justify-center shadow-[0_12px_32px_rgba(10,132,255,0.4)]">
            <span className="text-white text-2xl font-bold tracking-tight">
              iA
            </span>
          </div>
          <h1 className="text-[32px] font-semibold text-white tracking-[-0.02em]">
            InvestAI
          </h1>
          <p className="text-sm text-gray-500 mt-1.5">
            Sign in to your account
          </p>
        </div>
        <form onSubmit={(e) => void handleSubmit(e)} className="card space-y-5 p-6">
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm rounded-xl px-3.5 py-2.5">
              {error}
            </div>
          )}
          <div>
            <label
              htmlFor="email"
              className="block text-[11px] font-medium text-gray-500 uppercase tracking-[0.14em] mb-2"
            >
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="username"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="input-field"
            />
          </div>
          <div>
            <label
              htmlFor="password"
              className="block text-[11px] font-medium text-gray-500 uppercase tracking-[0.14em] mb-2"
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="input-field"
            />
          </div>
          <button
            type="submit"
            disabled={submitting}
            className="btn-primary w-full disabled:opacity-50 disabled:shadow-none"
          >
            {submitting ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  );
}
