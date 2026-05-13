import React from 'react';

interface TopBarProps {
  backends: {
    python: boolean;
    athena: boolean;
    athenak: boolean;
    metal: boolean;
    mlx: boolean;
    hybrid: boolean;
  };
  simStatus: 'idle' | 'running' | 'paused' | 'finished' | 'error';
  step: number;
  time: number;
  resultLabel?: string;
  canSupportValidationClaims?: boolean;
  blockerCount?: number;
  digitizationStatus?: string;
  readinessScopeNote?: string;
  coPilotOpen: boolean;
  onToggleCoPilot: () => void;
}

export const formatSimulationTime = (timeSeconds: number): string => {
  if (!Number.isFinite(timeSeconds)) return 'n/a';

  const absSeconds = Math.abs(timeSeconds);
  if (absSeconds === 0) return '0.00 ns';
  if (absSeconds < 1e-6) return `${(timeSeconds * 1e9).toFixed(2)} ns`;
  if (absSeconds < 1e-3) return `${(timeSeconds * 1e6).toFixed(2)} μs`;
  if (absSeconds < 1) return `${(timeSeconds * 1e3).toFixed(2)} ms`;
  return `${timeSeconds.toFixed(2)} s`;
};

export const TopBar: React.FC<TopBarProps> = ({
  backends,
  simStatus,
  step,
  time,
  resultLabel,
  canSupportValidationClaims = false,
  blockerCount = 0,
  digitizationStatus,
  readinessScopeNote,
  coPilotOpen,
  onToggleCoPilot,
}) => {
  const getStatusConfig = () => {
    switch (simStatus) {
      case 'idle':
        return { text: 'Idle', dotClass: 'status-dot-idle', pulse: false };
      case 'running':
        return { text: 'Running', dotClass: 'status-dot-ok', pulse: true };
      case 'paused':
        return { text: 'Paused', dotClass: 'status-dot-warn', pulse: false };
      case 'finished':
        return { text: 'Finished', dotClass: 'status-dot-ok', pulse: false };
      case 'error':
        return { text: 'Error', dotClass: 'status-dot-error', pulse: false };
    }
  };

  const statusConfig = getStatusConfig();
  const activeBackends = Object.entries(backends)
    .filter(([_, active]) => active)
    .map(([name]) => name);

  return (
    <div className="dpf-panel flex items-center justify-between px-6 py-3 border-b border-[#333333]">
      {/* Left: Title and Version */}
      <div className="flex items-center gap-4">
        <h1 className="text-[#00E5FF] font-mono text-xl font-bold tracking-wider">
          DPF SIMULATOR
        </h1>
        <span className="dpf-label text-xs px-2 py-1 bg-[#2A2A2A] rounded">
          v{__APP_VERSION__}
        </span>
      </div>

      {/* Center: Status Indicator */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div
              className={`status-dot ${statusConfig.dotClass} ${
                statusConfig.pulse ? 'animate-pulse' : ''
              }`}
            />
            {simStatus === 'finished' && (
              <svg
                className="absolute inset-0 w-3 h-3 text-[#00E5FF]"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
            )}
          </div>
          <span className="dpf-label font-medium">{statusConfig.text}</span>
        </div>
        <div className="h-4 w-px bg-[#333333]" />
        <div className="flex items-center gap-4">
          <div className="dpf-label">
            <span className="text-[#666666]">Step:</span>{' '}
            <span className="font-mono font-medium">{step.toLocaleString()}</span>
          </div>
          <div className="dpf-label">
            <span className="text-[#666666]">Time:</span>{' '}
            <span className="font-mono font-medium">{formatSimulationTime(time)}</span>
          </div>
        </div>
      </div>

      {/* Right: Backend Badges and Co-Pilot Toggle */}
      <div className="flex items-center gap-3">
        {activeBackends.length > 0 ? (
          activeBackends.map((backend) => (
            <span
              key={backend}
              className="dpf-label text-xs px-3 py-1 bg-[#2A2A2A] rounded border border-[#00E5FF]/30"
            >
            {backend === 'python'
                ? 'Python'
                : backend === 'athena'
                ? 'Athena++'
                : backend === 'athenak'
                ? 'AthenaK'
                : backend === 'metal'
                ? 'Metal GPU'
                : backend === 'mlx'
                ? 'MLX'
                : backend === 'hybrid'
                ? 'Hybrid'
                : backend}
            </span>
          ))
        ) : (
          <span className="dpf-label text-xs px-3 py-1 bg-[#2A2A2A] rounded border border-[#666666]">
            No Backend
          </span>
        )}
        {resultLabel && (
          <span
            className={`dpf-label text-xs px-3 py-1 rounded border ${
              canSupportValidationClaims
                ? 'bg-accent-green-dim text-accent-green border-accent-green/40'
                : 'bg-accent-amber-dim text-accent-amber border-accent-amber/40'
            }`}
            title={
              canSupportValidationClaims
                ? 'Reference-authorized result'
                : 'Preview or non-certifying result'
            }
          >
            {resultLabel}
          </span>
        )}
        {blockerCount > 0 && (
          <span
            className="dpf-label text-xs px-3 py-1 rounded border border-accent-crimson/40 bg-accent-crimson-dim text-accent-crimson"
            title={readinessScopeNote || digitizationStatus || 'Readiness blockers present'}
          >
            Blockers {blockerCount}
          </span>
        )}
        <div className="h-4 w-px bg-[#333333]" />
        <button
          onClick={onToggleCoPilot}
          className={`dpf-label text-sm px-4 py-2 rounded transition-all ${
            coPilotOpen
              ? 'bg-[#00E5FF] text-[#121212] font-medium'
              : 'bg-[#2A2A2A] hover:bg-[#333333] border border-[#333333]'
          }`}
        >
          AI Co-Pilot
        </button>
      </div>
    </div>
  );
};
