import React, { useMemo } from 'react';
import { TraceChart } from './TraceChart';
import type { ScalarUpdate } from '../../api/types';

interface MultiTraceViewProps {
  scalarHistory: ScalarUpdate[];
}

export const MultiTraceView: React.FC<MultiTraceViewProps> = ({ scalarHistory }) => {
  const { electricalTraces, plasmaTraces } = useMemo(() => {
    if (scalarHistory.length === 0) {
      return { electricalTraces: [], plasmaTraces: [] };
    }

    // Prepare current data (I(t))
    const currentData = scalarHistory.map((point) => [
      point.time,
      point.current,
    ]) as Array<[number, number]>;

    // Compute dI/dt using 3-point centered finite difference
    const dIdtData: Array<[number, number]> = [];
    for (let i = 1; i < scalarHistory.length - 1; i++) {
      const dt = scalarHistory[i + 1].time - scalarHistory[i - 1].time;
      if (dt > 0) {
        const dIdt =
          (scalarHistory[i + 1].current - scalarHistory[i - 1].current) / dt;
        dIdtData.push([scalarHistory[i].time, dIdt]);
      }
    }

    // Electrical traces
    const electrical = [
      {
        name: 'I(t)',
        color: '#00E5FF',
        data: currentData,
        yAxisIndex: 0,
      },
      {
        name: 'dI/dt',
        color: '#FFC107',
        data: dIdtData,
        yAxisIndex: 1,
      },
    ];

    // Prepare plasma data (max_Te, max_rho)
    const maxTeData = scalarHistory.map((point) => [
      point.time,
      point.max_Te,
    ]) as Array<[number, number]>;

    const maxRhoData = scalarHistory.map((point) => [
      point.time,
      point.max_rho,
    ]) as Array<[number, number]>;

    // Plasma traces
    const plasma = [
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

    return { electricalTraces: electrical, plasmaTraces: plasma };
  }, [scalarHistory]);

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

  return (
    <div className="flex flex-col h-full gap-4" style={{ backgroundColor: '#0A0A0A' }}>
      {/* Electrical traces */}
      <div className="flex-1">
        <TraceChart
          title="ELECTRICAL"
          traces={electricalTraces}
          yAxisLabel="Current (A)"
          yAxisLabel2="dI/dt (A/s)"
        />
      </div>

      {/* Plasma traces */}
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
