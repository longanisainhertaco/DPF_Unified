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
  // Compute pinch time: time at which dI/dt is most negative
  const pinchTime = React.useMemo(() => {
    if (scalarHistory.length < 2) return 0;

    let minDerivative = 0;
    let pinchT = 0;

    for (let i = 1; i < scalarHistory.length; i++) {
      const dt = scalarHistory[i].time - scalarHistory[i - 1].time;
      const dI = scalarHistory[i].current - scalarHistory[i - 1].current;
      const derivative = dI / dt;

      if (derivative < minDerivative) {
        minDerivative = derivative;
        pinchT = scalarHistory[i].time;
      }
    }

    return pinchT;
  }, [scalarHistory]);

  // Peak current
  const peakCurrent = React.useMemo(() => {
    return Math.max(...scalarHistory.map(s => s.current));
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
            {(pinchTime * 1e6).toFixed(2)} μs
          </div>
        </div>
        <div>
          <div className="dpf-label text-xs mb-1">PEAK CURRENT</div>
          <div className="text-cyan-400 font-mono text-lg">
            {(peakCurrent / 1e6).toFixed(2)} MA
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
