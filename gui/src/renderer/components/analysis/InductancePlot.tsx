import React from 'react';
import ReactECharts from 'echarts-for-react';
import type { ScalarUpdate } from '../../api/types';

interface InductancePlotProps {
  scalarHistory: ScalarUpdate[];
}

const formatEngineering = (value: number): string => {
  if (value === 0) return '0';

  const absValue = Math.abs(value);
  const exponent = Math.floor(Math.log10(absValue));
  const mantissa = value / Math.pow(10, exponent);

  return `${mantissa.toFixed(1)}e${exponent}`;
};

export const InductancePlot: React.FC<InductancePlotProps> = ({ scalarHistory }) => {
  const option = React.useMemo(() => {
    const times = scalarHistory.map(s => s.time);
    const resistances = scalarHistory.map(s => s.R_plasma || 0);

    return {
      backgroundColor: 'transparent',
      title: {
        text: 'R_PLASMA(t)',
        left: 'center',
        top: 10,
        textStyle: {
          color: '#9CA3AF',
          fontSize: 12,
          fontFamily: 'Inter, sans-serif',
          fontWeight: 500,
        },
      },
      grid: {
        left: 60,
        right: 30,
        top: 50,
        bottom: 50,
      },
      xAxis: {
        type: 'value',
        name: 'Time (s)',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          color: '#6B7280',
          fontSize: 11,
        },
        axisLine: {
          lineStyle: { color: '#333333' },
        },
        axisLabel: {
          color: '#9CA3AF',
          fontSize: 10,
          fontFamily: 'JetBrains Mono, monospace',
          formatter: (value: number) => formatEngineering(value),
        },
        splitLine: {
          lineStyle: { color: '#1E1E1E' },
        },
      },
      yAxis: {
        type: 'value',
        name: 'R (Ω)',
        nameLocation: 'middle',
        nameGap: 45,
        nameTextStyle: {
          color: '#6B7280',
          fontSize: 11,
        },
        axisLine: {
          lineStyle: { color: '#333333' },
        },
        axisLabel: {
          color: '#9CA3AF',
          fontSize: 10,
          fontFamily: 'JetBrains Mono, monospace',
          formatter: (value: number) => formatEngineering(value),
        },
        splitLine: {
          lineStyle: { color: '#1E1E1E' },
        },
      },
      series: [
        {
          type: 'line',
          data: times.map((t, i) => [t, resistances[i]]),
          lineStyle: {
            color: '#FFC107',
            width: 2,
          },
          itemStyle: {
            color: '#FFC107',
          },
          symbol: 'none',
          smooth: false,
        },
      ],
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1E1E1E',
        borderColor: '#333333',
        textStyle: {
          color: '#E5E7EB',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 11,
        },
        formatter: (params: any) => {
          const point = params[0];
          return `t: ${formatEngineering(point.value[0])} s<br/>R: ${formatEngineering(point.value[1])} Ω`;
        },
      },
    };
  }, [scalarHistory]);

  return (
    <div className="border border-gray-700 rounded-lg bg-gray-900 p-2">
      <ReactECharts option={option} style={{ height: '300px' }} />
    </div>
  );
};
