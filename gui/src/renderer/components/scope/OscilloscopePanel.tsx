import React from 'react';
import { TraceChart } from './TraceChart';
import { useSimulationStore } from '../../stores/simulation';

export const OscilloscopePanel: React.FC = () => {
  const scalarHistory = useSimulationStore((state) => state.scalarHistory);

  // Show placeholder when no data available
  if (scalarHistory.length === 0) {
    return (
      <div
        className="flex items-center justify-center h-full"
        style={{ backgroundColor: '#0A0A0A' }}
      >
        <p className="text-gray-600 font-mono text-sm">
          Awaiting simulation data...
        </p>
      </div>
    );
  }

  // Prepare electrical traces (I(t))
  const currentData = scalarHistory.map((point) => [
    point.time,
    point.current,
  ]) as Array<[number, number]>;

  const electricalTraces = [
    {
      name: 'I(t)',
      color: '#00E5FF',
      data: currentData,
      yAxisIndex: 0,
    },
  ];

  // Prepare plasma traces (max_Te, max_rho)
  const maxTeData = scalarHistory.map((point) => [
    point.time,
    point.max_Te,
  ]) as Array<[number, number]>;

  const maxRhoData = scalarHistory.map((point) => [
    point.time,
    point.max_rho,
  ]) as Array<[number, number]>;

  const plasmaTraces = [
    {
      name: 'max_Te',
      color: '#FFC107',
      data: maxTeData,
      yAxisIndex: 0,
    },
    {
      name: 'max_rho',
      color: '#FF5252',
      data: maxRhoData,
      yAxisIndex: 1,
    },
  ];

  return (
    <div className="flex flex-col h-full" style={{ backgroundColor: '#0A0A0A' }}>
      {/* Top chart: Electrical */}
      <div className="flex-1">
        <TraceChart
          title="ELECTRICAL"
          traces={electricalTraces}
          yAxisLabel="Current (A)"
        />
      </div>

      {/* Bottom chart: Plasma */}
      <div className="flex-1">
        <TraceChart
          title="PLASMA"
          traces={plasmaTraces}
          yAxisLabel="max_Te (eV)"
          yAxisLabel2="max_rho (kg/m³)"
        />
      </div>
    </div>
  );
};
