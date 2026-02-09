import React, { useEffect } from 'react';
import { useAIStore } from '../../stores/ai';
import { AdvisoryPanel } from '../copilot/AdvisoryPanel';
import { SweepPanel } from '../copilot/SweepPanel';

export const CoPilotSidebar: React.FC = () => {
  const aiAvailable = useAIStore((s) => s.aiAvailable);
  const checkAIStatus = useAIStore((s) => s.checkAIStatus);

  // Check AI status on mount
  useEffect(() => {
    checkAIStatus();
  }, [checkAIStatus]);

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
