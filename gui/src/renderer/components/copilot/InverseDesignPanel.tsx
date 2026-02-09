import React, { useState } from 'react';
import { useAIStore } from '../../stores/ai';
import { useConfigStore } from '../../stores/config';

const TARGET_METRICS = [
  { value: 'max_Te', label: 'Max Tₑ', unit: 'keV' },
  { value: 'max_rho', label: 'Max ρ', unit: 'kg/m³' },
  { value: 'max_Ti', label: 'Max Tᵢ', unit: 'keV' },
  { value: 'neutron_rate', label: 'Neutron Rate', unit: '/s' },
  { value: 'max_B', label: 'Max B', unit: 'T' },
  { value: 'R_plasma', label: 'Plasma Radius', unit: 'm' },
];

const CONSTRAINT_PARAMS = [
  { value: 'V0', label: 'V₀ max', unit: 'V' },
  { value: 'C', label: 'C max', unit: 'F' },
  { value: 'L0', label: 'L₀ max', unit: 'H' },
  { value: 'R0', label: 'R₀ max', unit: 'Ω' },
];

const formatScientific = (num: number): string => {
  if (num === 0) return '0';
  const exp = Math.floor(Math.log10(Math.abs(num)));
  if (exp >= -2 && exp <= 3) {
    return num.toPrecision(4);
  }
  return num.toExponential(3);
};

export const InverseDesignPanel: React.FC = () => {
  const inverseStatus = useAIStore((s) => s.inverseStatus);
  const inverseResult = useAIStore((s) => s.inverseResult);
  const runInverse = useAIStore((s) => s.runInverse);
  const updateCircuit = useConfigStore((s) => s.updateCircuit);
  const updateGrid = useConfigStore((s) => s.updateGrid);

  // Targets state
  const [targetMetric, setTargetMetric] = useState('max_Te');
  const [targetValue, setTargetValue] = useState('1.0');

  // Secondary target
  const [useSecondTarget, setUseSecondTarget] = useState(false);
  const [targetMetric2, setTargetMetric2] = useState('max_rho');
  const [targetValue2, setTargetValue2] = useState('1e-3');

  // Optimization settings
  const [method, setMethod] = useState('bayesian');
  const [nTrials, setNTrials] = useState(100);

  // Constraints
  const [useConstraints, setUseConstraints] = useState(false);
  const [constraintV0, setConstraintV0] = useState('50000');
  const [constraintC, setConstraintC] = useState('1e-3');

  const isRunning = inverseStatus === 'running';

  const handleRunInverse = () => {
    const targets: Record<string, number> = {};
    const val1 = parseFloat(targetValue);
    if (!isNaN(val1)) {
      targets[targetMetric] = val1;
    }
    if (useSecondTarget) {
      const val2 = parseFloat(targetValue2);
      if (!isNaN(val2)) {
        targets[targetMetric2] = val2;
      }
    }

    const constraints: Record<string, number> = {};
    if (useConstraints) {
      const v0Max = parseFloat(constraintV0);
      const cMax = parseFloat(constraintC);
      if (!isNaN(v0Max)) constraints['V0'] = v0Max;
      if (!isNaN(cMax)) constraints['C'] = cMax;
    }

    runInverse(targets, constraints, method, nTrials);
  };

  const handleApplyConfig = () => {
    if (!inverseResult) return;

    const best = inverseResult.best_config;

    // Apply circuit parameters from the result
    const circuitUpdate: Record<string, number> = {};
    const gridUpdate: Record<string, number> = {};

    for (const [key, value] of Object.entries(best)) {
      if (['V0', 'C', 'L0', 'R0', 'anode_radius', 'cathode_radius', 'ESR', 'ESL'].includes(key)) {
        circuitUpdate[key] = value;
      } else if (['rho0', 'T0', 'anomalous_alpha'].includes(key)) {
        gridUpdate[key] = value;
      }
    }

    if (Object.keys(circuitUpdate).length > 0) {
      updateCircuit(circuitUpdate);
    }
    if (Object.keys(gridUpdate).length > 0) {
      updateGrid(gridUpdate);
    }
  };

  return (
    <div className="dpf-panel space-y-3">
      <div className="dpf-label text-xs font-semibold">INVERSE DESIGN</div>
      <p className="text-[10px] text-gray-500">
        Specify target outcomes — WALRUS finds optimal parameters.
      </p>

      {/* Primary Target */}
      <div>
        <label className="dpf-label text-xs mb-1 block">TARGET</label>
        <div className="flex gap-2">
          <select
            value={targetMetric}
            onChange={(e) => setTargetMetric(e.target.value)}
            disabled={isRunning}
            className="dpf-input flex-1 text-sm"
          >
            {TARGET_METRICS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <input
            type="text"
            value={targetValue}
            onChange={(e) => setTargetValue(e.target.value)}
            disabled={isRunning}
            className="dpf-input w-24 font-mono text-sm text-right"
          />
          <span className="dpf-label text-xs text-gray-500 self-center w-10">
            {TARGET_METRICS.find((m) => m.value === targetMetric)?.unit ?? ''}
          </span>
        </div>
      </div>

      {/* Secondary Target Toggle */}
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={useSecondTarget}
          onChange={(e) => setUseSecondTarget(e.target.checked)}
          disabled={isRunning}
          className="accent-[#00E5FF]"
        />
        <span className="dpf-label text-xs text-gray-400">Add second target</span>
      </div>

      {useSecondTarget && (
        <div className="flex gap-2">
          <select
            value={targetMetric2}
            onChange={(e) => setTargetMetric2(e.target.value)}
            disabled={isRunning}
            className="dpf-input flex-1 text-sm"
          >
            {TARGET_METRICS.filter((m) => m.value !== targetMetric).map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <input
            type="text"
            value={targetValue2}
            onChange={(e) => setTargetValue2(e.target.value)}
            disabled={isRunning}
            className="dpf-input w-24 font-mono text-sm text-right"
          />
          <span className="dpf-label text-xs text-gray-500 self-center w-10">
            {TARGET_METRICS.find((m) => m.value === targetMetric2)?.unit ?? ''}
          </span>
        </div>
      )}

      {/* Method + Trials */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="dpf-label text-xs mb-1 block">METHOD</label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            disabled={isRunning}
            className="dpf-input w-full text-sm"
          >
            <option value="bayesian">Bayesian</option>
            <option value="evolutionary">Evolutionary</option>
          </select>
        </div>
        <div>
          <label className="dpf-label text-xs mb-1 block">TRIALS</label>
          <input
            type="number"
            min={10}
            max={1000}
            value={nTrials}
            onChange={(e) => setNTrials(parseInt(e.target.value) || 100)}
            disabled={isRunning}
            className="dpf-input w-full font-mono text-sm"
          />
        </div>
      </div>

      {/* Constraints */}
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={useConstraints}
          onChange={(e) => setUseConstraints(e.target.checked)}
          disabled={isRunning}
          className="accent-[#00E5FF]"
        />
        <span className="dpf-label text-xs text-gray-400">Constraints</span>
      </div>

      {useConstraints && (
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="dpf-label text-xs mb-1 block">V₀ MAX (V)</label>
            <input
              type="text"
              value={constraintV0}
              onChange={(e) => setConstraintV0(e.target.value)}
              disabled={isRunning}
              className="dpf-input w-full font-mono text-sm"
            />
          </div>
          <div>
            <label className="dpf-label text-xs mb-1 block">C MAX (F)</label>
            <input
              type="text"
              value={constraintC}
              onChange={(e) => setConstraintC(e.target.value)}
              disabled={isRunning}
              className="dpf-input w-full font-mono text-sm"
            />
          </div>
        </div>
      )}

      {/* Run button */}
      <button
        onClick={handleRunInverse}
        disabled={isRunning}
        className={`w-full rounded px-4 py-2 text-sm font-medium transition-all ${
          isRunning
            ? 'bg-dpf-input text-gray-500 cursor-wait'
            : 'bg-accent-cyan text-dpf-bg hover:brightness-110'
        }`}
      >
        {isRunning ? 'OPTIMIZING...' : 'FIND OPTIMAL CONFIG'}
      </button>

      {/* Error state */}
      {inverseStatus === 'error' && (
        <div className="text-[#FF5252] text-xs font-mono text-center">
          Inverse design failed — check AI status
        </div>
      )}

      {/* Results */}
      {inverseStatus === 'complete' && inverseResult && (
        <div className="border border-gray-700 rounded-lg bg-gray-900/50 p-3 space-y-2">
          <div className="dpf-label text-xs font-semibold text-accent-cyan">RESULTS</div>

          {/* Show best config values */}
          <div className="space-y-1">
            {Object.entries(inverseResult.best_config).map(([key, value]) => (
              <div key={key} className="flex justify-between font-mono text-xs">
                <span className="text-gray-400">{key}</span>
                <span className="text-gray-200">{formatScientific(value)}</span>
              </div>
            ))}
          </div>

          {/* Loss + Trials */}
          <div className="flex justify-between font-mono text-xs border-t border-gray-700 pt-2">
            <span className="text-gray-500">Loss</span>
            <span className="text-[#FFC107]">{inverseResult.loss.toFixed(4)}</span>
          </div>
          <div className="flex justify-between font-mono text-xs">
            <span className="text-gray-500">Trials</span>
            <span className="text-gray-300">{inverseResult.n_trials}</span>
          </div>

          {/* Apply button */}
          <button
            onClick={handleApplyConfig}
            className="w-full rounded px-4 py-2 text-sm font-medium border-2 border-[#00E5FF] text-[#00E5FF] hover:bg-[#00E5FF] hover:text-[#121212] transition-all"
          >
            APPLY CONFIG
          </button>
        </div>
      )}
    </div>
  );
};
