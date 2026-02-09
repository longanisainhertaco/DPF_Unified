import React, { useState } from 'react';
import { useAIStore } from '../../stores/ai';
import { useConfigStore } from '../../stores/config';

const SWEEP_VARIABLES = [
  { value: 'V0', label: 'V\u2080 (Voltage)' },
  { value: 'C', label: 'C (Capacitance)' },
  { value: 'rho0', label: '\u03C1\u2080 (Density)' },
  { value: 'anode_radius', label: 'Anode Radius' },
  { value: 'cathode_radius', label: 'Cathode Radius' },
  { value: 'L0', label: 'L\u2080 (Inductance)' },
];

export const SweepPanel: React.FC = () => {
  const sweepStatus = useAIStore((s) => s.sweepStatus);
  const runSweep = useAIStore((s) => s.runSweep);
  const config = useConfigStore((s) => s.config);

  const [variable, setVariable] = useState('V0');
  const [minValue, setMinValue] = useState('10000');
  const [maxValue, setMaxValue] = useState('50000');
  const [nPoints, setNPoints] = useState(10);

  const handleRunSweep = () => {
    const min = parseFloat(minValue);
    const max = parseFloat(maxValue);

    if (isNaN(min) || isNaN(max) || min >= max || nPoints < 2) {
      return;
    }

    // Generate configs for each point in the sweep, merged with current config
    const step = (max - min) / (nPoints - 1);
    const baseCircuit = config.circuit ?? {};
    const circuitParams = ['V0', 'C', 'L0', 'R0', 'anode_radius', 'cathode_radius'];
    const isCircuitParam = circuitParams.includes(variable);

    const configs = Array.from({ length: nPoints }, (_, i) => {
      const val = min + i * step;
      if (isCircuitParam) {
        return { ...config, circuit: { ...baseCircuit, [variable]: val } };
      }
      return { ...config, [variable]: val };
    });

    runSweep(configs, 100, variable);
  };

  const isRunning = sweepStatus === 'running';

  return (
    <div className="dpf-panel space-y-3">
      <div className="dpf-label text-xs font-semibold">PARAMETER SWEEP</div>

      {/* Variable selector */}
      <div>
        <label className="dpf-label text-xs mb-1 block">VARIABLE</label>
        <select
          value={variable}
          onChange={(e) => setVariable(e.target.value)}
          disabled={isRunning}
          className="dpf-input w-full text-sm"
        >
          {SWEEP_VARIABLES.map(v => (
            <option key={v.value} value={v.value}>
              {v.label}
            </option>
          ))}
        </select>
      </div>

      {/* Range inputs */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="dpf-label text-xs mb-1 block">MIN</label>
          <input
            type="text"
            value={minValue}
            onChange={(e) => setMinValue(e.target.value)}
            disabled={isRunning}
            className="dpf-input w-full font-mono text-sm"
          />
        </div>
        <div>
          <label className="dpf-label text-xs mb-1 block">MAX</label>
          <input
            type="text"
            value={maxValue}
            onChange={(e) => setMaxValue(e.target.value)}
            disabled={isRunning}
            className="dpf-input w-full font-mono text-sm"
          />
        </div>
      </div>

      {/* N points */}
      <div>
        <label className="dpf-label text-xs mb-1 block">N POINTS</label>
        <input
          type="number"
          min="3"
          max="100"
          value={nPoints}
          onChange={(e) => setNPoints(parseInt(e.target.value) || 10)}
          disabled={isRunning}
          className="dpf-input w-full font-mono text-sm"
        />
      </div>

      {/* Run button */}
      <button
        onClick={handleRunSweep}
        disabled={isRunning}
        className={`w-full rounded px-4 py-2 text-sm font-medium transition-all ${
          isRunning
            ? 'bg-dpf-input text-gray-500 cursor-wait'
            : 'bg-accent-cyan text-dpf-bg hover:brightness-110'
        }`}
      >
        {isRunning ? 'RUNNING...' : 'RUN SWEEP'}
      </button>

      {/* Complete indicator */}
      {sweepStatus === 'complete' && (
        <div className="text-accent-green text-xs font-mono text-center">
          Sweep complete
        </div>
      )}
    </div>
  );
};
