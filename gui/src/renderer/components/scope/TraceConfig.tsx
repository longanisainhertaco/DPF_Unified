import React from 'react';

interface TraceConfigProps {
  logScale: boolean;
  onToggleLogScale: () => void;
  autoRange: boolean;
  onToggleAutoRange: () => void;
}

export const TraceConfig: React.FC<TraceConfigProps> = ({
  logScale,
  onToggleLogScale,
  autoRange,
  onToggleAutoRange,
}) => {
  return (
    <div className="flex items-center gap-4 px-4 py-2" style={{ backgroundColor: '#0A0A0A' }}>
      {/* Log/Linear scale toggle */}
      <button
        onClick={onToggleLogScale}
        className="dpf-btn-ghost text-xs font-mono"
        style={{
          color: logScale ? '#00E5FF' : '#888888',
          borderColor: logScale ? '#00E5FF' : '#1A1A1A',
        }}
      >
        {logScale ? 'LOG' : 'LIN'}
      </button>

      {/* Auto/Manual range toggle */}
      <button
        onClick={onToggleAutoRange}
        className="dpf-btn-ghost text-xs font-mono"
        style={{
          color: autoRange ? '#00E5FF' : '#888888',
          borderColor: autoRange ? '#00E5FF' : '#1A1A1A',
        }}
      >
        {autoRange ? 'AUTO' : 'MANUAL'}
      </button>
    </div>
  );
};
