import React, { useEffect } from 'react';
import { useAIStore } from '../../stores/ai';
import { AdvisoryPanel } from '../copilot/AdvisoryPanel';
import { SweepPanel } from '../copilot/SweepPanel';
import { ScalingCurve } from '../copilot/ScalingCurve';
import { InverseDesignPanel } from '../copilot/InverseDesignPanel';
import { ChatPanel } from '../copilot/ChatPanel';

export const CoPilotSidebar: React.FC = () => {
  const aiAvailable = useAIStore((s) => s.aiAvailable);
  const checkAIStatus = useAIStore((s) => s.checkAIStatus);
  const sweepStatus = useAIStore((s) => s.sweepStatus);
  const sweepResults = useAIStore((s) => s.sweepResults);
  const sweepVariable = useAIStore((s) => s.sweepVariable);
  const sweepMetric = useAIStore((s) => s.sweepMetric);
  const setSweepMetric = useAIStore((s) => s.setSweepMetric);

  // Check AI status on mount
  useEffect(() => {
    checkAIStatus();
  }, [checkAIStatus]);

  const METRIC_OPTIONS = [
    { value: 'max_Te', label: 'Max Tₑ' },
    { value: 'max_rho', label: 'Max ρ' },
    { value: 'max_Ti', label: 'Max Tᵢ' },
    { value: 'max_B', label: 'Max B' },
    { value: 'neutron_rate', label: 'Neutron Rate' },
    { value: 'R_plasma', label: 'R_plasma' },
    { value: 'total_radiated_energy', label: 'Radiated Energy' },
    { value: 'Z_bar', label: 'Z̄' },
  ];

  return (
    <div className="h-full flex flex-col bg-dpf-panel">
      {/* Header */}
      <div className="px-4 py-3 border-b border-dpf-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h2 className="dpf-label text-sm font-semibold tracking-wide">AI CO-PILOT</h2>
          </div>
          <div className="flex items-center gap-2">
            <span className={`status-dot ${aiAvailable ? 'status-dot-ok' : 'status-dot-idle'}`} />
            <span className="font-mono text-label-xs text-gray-500">
              {aiAvailable ? 'Online' : 'Heuristic'}
            </span>
          </div>
        </div>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Advisory Messages */}
        <AdvisoryPanel />

        {/* Parameter Sweep */}
        <SweepPanel />

        {/* Scaling Curve with metric selector (shown after sweep completes) */}
        {sweepStatus === 'complete' && sweepResults.length > 0 && sweepVariable && (
          <div className="space-y-2">
            {/* Metric selector */}
            <div>
              <label className="dpf-label text-xs mb-1 block">PLOT METRIC</label>
              <select
                value={sweepMetric}
                onChange={(e) => setSweepMetric(e.target.value)}
                className="dpf-input w-full text-sm"
              >
                {METRIC_OPTIONS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>

            <ScalingCurve
              variable={sweepVariable}
              metric={sweepMetric}
              data={sweepResults.map((r) => {
                const cfg = r.config as Record<string, any>;
                const paramVal = cfg.circuit?.[sweepVariable] ?? cfg[sweepVariable] ?? 0;
                const metricVal = r.metrics?.[sweepMetric] ?? 0;
                return [paramVal as number, metricVal] as [number, number];
              })}
            />
          </div>
        )}

        {/* Inverse Design */}
        <InverseDesignPanel />

        {/* Chat with WALRUS */}
        <ChatPanel />
      </div>

      {/* Footer: WALRUS Badge */}
      <div className="px-4 py-3 border-t border-dpf-border">
        <div className="flex items-center justify-center gap-2">
          <svg className="w-4 h-4 text-accent-cyan" fill="currentColor" viewBox="0 0 20 20">
            <path d="M13 7H7v6h6V7z" />
            <path
              fillRule="evenodd"
              d="M7 2a1 1 0 012 0v1h2V2a1 1 0 112 0v1h2a2 2 0 012 2v2h1a1 1 0 110 2h-1v2h1a1 1 0 110 2h-1v2a2 2 0 01-2 2h-2v1a1 1 0 11-2 0v-1H9v1a1 1 0 11-2 0v-1H5a2 2 0 01-2-2v-2H2a1 1 0 110-2h1V9H2a1 1 0 010-2h1V5a2 2 0 012-2h2V2zM5 5h10v10H5V5z"
              clipRule="evenodd"
            />
          </svg>
          <span className="dpf-label text-xs font-medium text-gray-500">Powered by WALRUS</span>
        </div>
      </div>
    </div>
  );
};
