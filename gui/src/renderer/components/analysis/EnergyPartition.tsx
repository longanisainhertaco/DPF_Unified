import React from 'react';

interface EnergyPartitionProps {
  capacitance: number;       // C in Farads
  voltage: number;           // V0 in Volts
  energyConservation: number; // from final ScalarUpdate (fraction)
}

const formatEngineering = (value: number): string => {
  if (value === 0) return '0';

  const exponent = Math.floor(Math.log10(Math.abs(value)));
  const mantissa = value / Math.pow(10, exponent);

  return `${mantissa.toFixed(2)} × 10${exponent >= 0 ? '⁺' : '⁻'}${Math.abs(exponent).toString().split('').map(d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[parseInt(d)]).join('')}`;
};

export const EnergyPartition: React.FC<EnergyPartitionProps> = ({
  capacitance,
  voltage,
  energyConservation,
}) => {
  const E_cap = 0.5 * capacitance * voltage * voltage;
  const E_plasma = energyConservation * E_cap;
  const E_loss = E_cap - E_plasma;

  const plasmaPercent = (E_plasma / E_cap) * 100;
  const lossPercent = (E_loss / E_cap) * 100;

  return (
    <div className="border border-gray-700 rounded-lg p-4 bg-gray-900">
      <div className="dpf-label text-xs mb-3">ENERGY PARTITION</div>

      {/* Bar chart */}
      <div className="flex h-8 mb-3 rounded overflow-hidden">
        <div
          className="bg-cyan-500"
          style={{ width: `${plasmaPercent}%` }}
          title={`Plasma: ${plasmaPercent.toFixed(1)}%`}
        />
        <div
          className="bg-amber-500"
          style={{ width: `${lossPercent}%` }}
          title={`Loss: ${lossPercent.toFixed(1)}%`}
        />
      </div>

      {/* Values */}
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <div className="dpf-label text-xs mb-1">BANK ENERGY</div>
          <div className="text-gray-300 font-mono">
            {formatEngineering(E_cap)} J
          </div>
        </div>
        <div>
          <div className="dpf-label text-xs mb-1">
            <span className="inline-block w-2 h-2 bg-cyan-500 mr-1" />
            PLASMA
          </div>
          <div className="text-cyan-400 font-mono">
            {formatEngineering(E_plasma)} J
          </div>
          <div className="text-gray-500 text-xs">
            {plasmaPercent.toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="dpf-label text-xs mb-1">
            <span className="inline-block w-2 h-2 bg-amber-500 mr-1" />
            LOSS
          </div>
          <div className="text-amber-400 font-mono">
            {formatEngineering(E_loss)} J
          </div>
          <div className="text-gray-500 text-xs">
            {lossPercent.toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
};
