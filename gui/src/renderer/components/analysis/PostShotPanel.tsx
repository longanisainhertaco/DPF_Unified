import React from 'react';
import { NeutronYield } from './NeutronYield';
import { EnergyPartition } from './EnergyPartition';
import type { ScalarUpdate } from '../../api/types';

interface PostShotPanelProps {
  scalarHistory: ScalarUpdate[];
  capacitance: number;  // C in Farads
  voltage: number;       // V0 in Volts
}

export const PostShotPanel: React.FC<PostShotPanelProps> = ({
  scalarHistory,
  capacitance,
  voltage,
}) => {
  // Peak current and peak current time (always meaningful)
  const { peakCurrent, peakCurrentTime } = React.useMemo(() => {
    let peak = 0;
    let peakTime = 0;
    for (const s of scalarHistory) {
      if (Math.abs(s.current) > peak) {
        peak = Math.abs(s.current);
        peakTime = s.time;
      }
    }
    return { peakCurrent: peak, peakCurrentTime: peakTime };
  }, [scalarHistory]);

  // Pinch time: time at which dI/dt is most negative (may not be reached)
  const { pinchTime, pinchReached } = React.useMemo(() => {
    if (scalarHistory.length < 2) return { pinchTime: 0, pinchReached: false };

    let minDerivative = Infinity;
    let pinchT = 0;
    let foundNegative = false;

    for (let i = 1; i < scalarHistory.length; i++) {
      const dt = scalarHistory[i].time - scalarHistory[i - 1].time;
      if (dt <= 0) continue;
      const dI = scalarHistory[i].current - scalarHistory[i - 1].current;
      const derivative = dI / dt;

      if (derivative < minDerivative) {
        minDerivative = derivative;
        pinchT = scalarHistory[i].time;
      }
      if (derivative < 0) foundNegative = true;
    }

    return { pinchTime: pinchT, pinchReached: foundNegative };
  }, [scalarHistory]);

  // Get final values
  const finalScalar = scalarHistory[scalarHistory.length - 1];
  const totalYield = finalScalar?.total_neutron_yield || 0;
  const peakRate = Math.max(...scalarHistory.map(s => s.neutron_rate || 0));
  const energyConservation = finalScalar?.energy_conservation || 0;

  return (
    <div className="dpf-panel animate-fade-in space-y-4">
      <div className="dpf-label text-lg mb-4">POST-SHOT ANALYSIS</div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="dpf-label text-xs mb-1">PINCH TIME</div>
          <div className="text-cyan-400 font-mono text-lg">
            {pinchReached ? `${(pinchTime * 1e6).toFixed(2)} μs` : 'Not reached'}
          </div>
        </div>
        <div>
          <div className="dpf-label text-xs mb-1">PEAK CURRENT</div>
          <div className="text-cyan-400 font-mono text-lg">
            {(peakCurrent / 1e6).toFixed(2)} MA
          </div>
        </div>
        <div>
          <div className="dpf-label text-xs mb-1">PEAK I TIME</div>
          <div className="text-cyan-400 font-mono text-lg">
            {(peakCurrentTime * 1e6).toFixed(2)} μs
          </div>
        </div>
      </div>

      <NeutronYield
        totalYield={totalYield}
        peakRate={peakRate}
        pinchTime={pinchTime}
      />

      <EnergyPartition
        capacitance={capacitance}
        voltage={voltage}
        energyConservation={energyConservation}
      />
    </div>
  );
};
