import React from 'react';

interface NeutronYieldProps {
  totalYield: number;    // total_neutron_yield from final ScalarUpdate
  peakRate: number;      // max neutron_rate across all updates
  pinchTime: number;     // estimated pinch time in seconds
}

const formatEngineering = (value: number): string => {
  if (value === 0) return '0';

  const exponent = Math.floor(Math.log10(Math.abs(value)));
  const mantissa = value / Math.pow(10, exponent);

  return `${mantissa.toFixed(2)} × 10${exponent >= 0 ? '⁺' : '⁻'}${Math.abs(exponent).toString().split('').map(d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[parseInt(d)]).join('')}`;
};

export const NeutronYield: React.FC<NeutronYieldProps> = ({
  totalYield,
  peakRate,
  pinchTime,
}) => {
  return (
    <div className="border border-gray-700 rounded-lg p-4 bg-gray-900">
      <div className="dpf-label text-xs mb-3">NEUTRON YIELD</div>

      <div className="readout text-4xl mb-4 text-cyan-400">
        {formatEngineering(totalYield)}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="dpf-label text-xs mb-1">PEAK RATE</div>
          <div className="text-gray-300 font-mono text-sm">
            {formatEngineering(peakRate)} /s
          </div>
        </div>
        <div>
          <div className="dpf-label text-xs mb-1">PINCH TIME</div>
          <div className="text-gray-300 font-mono text-sm">
            {(pinchTime * 1e6).toFixed(2)} μs
          </div>
        </div>
      </div>
    </div>
  );
};
