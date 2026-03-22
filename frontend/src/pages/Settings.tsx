import { useState } from 'react';
import clsx from 'clsx';
import { useAppStore } from '../store';

export default function Settings() {
  const { tradingMode, setTradingMode } = useAppStore();

  const [riskParams, setRiskParams] = useState({
    max_position_size: 10,
    max_portfolio_risk: 5,
    max_drawdown_limit: 15,
    var_limit: 50000,
    daily_loss_limit: 10000,
  });

  const [activeSymbols, setActiveSymbols] = useState(
    'AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA',
  );

  const [apiKeyName, setApiKeyName] = useState('');
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <h1 className="text-2xl font-bold text-white">Settings</h1>

      {/* Trading Mode */}
      <div className="card">
        <div className="card-header">Trading Mode</div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setTradingMode('paper')}
            className={clsx(
              'px-6 py-3 rounded-lg font-bold text-sm transition-colors',
              tradingMode === 'paper'
                ? 'bg-yellow-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700',
            )}
          >
            Paper Trading
          </button>
          <button
            onClick={() => setTradingMode('live')}
            className={clsx(
              'px-6 py-3 rounded-lg font-bold text-sm transition-colors',
              tradingMode === 'live'
                ? 'bg-red-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700',
            )}
          >
            Live Trading
          </button>
        </div>
        {tradingMode === 'live' && (
          <div className="mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-sm text-red-400 font-medium">
              WARNING: Live trading uses real funds. Ensure all risk parameters
              are properly configured.
            </p>
          </div>
        )}
      </div>

      {/* Risk Parameters */}
      <div className="card">
        <div className="card-header">Risk Parameters</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Max Position Size (%)
            </label>
            <input
              type="number"
              value={riskParams.max_position_size}
              onChange={(e) =>
                setRiskParams({
                  ...riskParams,
                  max_position_size: parseFloat(e.target.value),
                })
              }
              className="input-field w-full font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Max Portfolio Risk (%)
            </label>
            <input
              type="number"
              value={riskParams.max_portfolio_risk}
              onChange={(e) =>
                setRiskParams({
                  ...riskParams,
                  max_portfolio_risk: parseFloat(e.target.value),
                })
              }
              className="input-field w-full font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Max Drawdown Limit (%)
            </label>
            <input
              type="number"
              value={riskParams.max_drawdown_limit}
              onChange={(e) =>
                setRiskParams({
                  ...riskParams,
                  max_drawdown_limit: parseFloat(e.target.value),
                })
              }
              className="input-field w-full font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              VaR Limit ($)
            </label>
            <input
              type="number"
              value={riskParams.var_limit}
              onChange={(e) =>
                setRiskParams({
                  ...riskParams,
                  var_limit: parseFloat(e.target.value),
                })
              }
              className="input-field w-full font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Daily Loss Limit ($)
            </label>
            <input
              type="number"
              value={riskParams.daily_loss_limit}
              onChange={(e) =>
                setRiskParams({
                  ...riskParams,
                  daily_loss_limit: parseFloat(e.target.value),
                })
              }
              className="input-field w-full font-mono"
            />
          </div>
        </div>
      </div>

      {/* Active Symbols */}
      <div className="card">
        <div className="card-header">Active Symbols</div>
        <div>
          <label className="block text-xs text-gray-500 mb-1 uppercase">
            Comma-separated symbols
          </label>
          <input
            type="text"
            value={activeSymbols}
            onChange={(e) => setActiveSymbols(e.target.value)}
            className="input-field w-full font-mono"
            placeholder="AAPL, GOOGL, MSFT"
          />
          <p className="text-xs text-gray-500 mt-2">
            These symbols will be actively monitored and traded by the AI models.
          </p>
        </div>
      </div>

      {/* API Keys */}
      <div className="card">
        <div className="card-header">API Key Management</div>
        <div className="space-y-3">
          <div className="flex gap-3">
            <input
              type="text"
              value={apiKeyName}
              onChange={(e) => setApiKeyName(e.target.value)}
              placeholder="Key name (e.g., Alpaca, Polygon)"
              className="input-field flex-1"
            />
            <button className="btn-primary text-sm">Add Key</button>
          </div>
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
            <div className="flex items-center justify-between py-2">
              <div>
                <span className="text-sm text-gray-300">Alpaca</span>
                <span className="text-xs text-green-400 ml-2">Connected</span>
              </div>
              <button className="text-xs text-red-400 hover:text-red-300">
                Revoke
              </button>
            </div>
            <div className="flex items-center justify-between py-2 border-t border-gray-800">
              <div>
                <span className="text-sm text-gray-300">Polygon.io</span>
                <span className="text-xs text-green-400 ml-2">Connected</span>
              </div>
              <button className="text-xs text-red-400 hover:text-red-300">
                Revoke
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <button onClick={handleSave} className="btn-primary px-8">
          {saved ? 'Saved!' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}
