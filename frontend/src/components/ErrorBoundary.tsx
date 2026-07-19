import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Top-level error boundary: a render crash in any page/widget shows a readable
 * error card instead of unmounting React and leaving a blank screen.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('Unhandled render error:', error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center px-4">
          <div className="max-w-lg w-full bg-gray-950 border border-red-500/30 rounded-lg p-6">
            <h1 className="text-lg font-bold text-red-400 mb-2">
              Something went wrong
            </h1>
            <p className="text-sm text-gray-400 mb-4">
              The page hit an unexpected error. Reloading usually fixes it — if
              it keeps happening, the error below is what to report.
            </p>
            <pre className="text-xs text-red-300 bg-red-500/10 rounded p-3 overflow-x-auto mb-4">
              {this.state.error.message}
            </pre>
            <button
              onClick={() => {
                this.setState({ error: null });
                window.location.href = '/';
              }}
              className="w-full bg-accent hover:bg-accent-dark text-white font-semibold rounded px-3 py-2 text-sm transition-colors"
            >
              Reload dashboard
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
