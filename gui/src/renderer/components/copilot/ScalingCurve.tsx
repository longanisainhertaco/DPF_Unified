import React from 'react';
import ReactECharts from 'echarts-for-react';

interface ScalingCurveProps {
  variable: string;      // swept parameter name
  metric: string;        // output metric name (e.g., "neutron_yield")
  data: Array<[number, number]>;  // [param_value, metric_value] pairs
  optimum?: [number, number];     // highlighted optimum point
}

const formatEngineering = (value: number): string => {
  if (value === 0) return '0';

  const absValue = Math.abs(value);
  const exponent = Math.floor(Math.log10(absValue));
  const mantissa = value / Math.pow(10, exponent);

  return `${mantissa.toFixed(1)}e${exponent}`;
};

export const ScalingCurve: React.FC<ScalingCurveProps> = ({
  variable,
  metric,
  data,
  optimum,
}) => {
  const option = React.useMemo(() => {
    const series: any[] = [
      {
        type: 'line',
        data: data,
        lineStyle: {
          color: '#00E5FF',
          width: 2,
        },
        itemStyle: {
          color: '#00E5FF',
        },
        symbol: 'circle',
        symbolSize: 6,
        smooth: false,
      },
    ];

    // Add optimum point as separate series
    if (optimum) {
      series.push({
        type: 'scatter',
        data: [optimum],
        itemStyle: {
          color: '#00E5FF',
          borderColor: '#00E5FF',
          borderWidth: 2,
        },
        symbol: 'diamond',
        symbolSize: 16,
        zlevel: 10,
      });
    }

    return {
      backgroundColor: 'transparent',
      title: {
        text: `${metric} vs ${variable}`,
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
        left: 70,
        right: 30,
        top: 50,
        bottom: 60,
      },
      xAxis: {
        type: 'value',
        name: variable,
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
        name: metric,
        nameLocation: 'middle',
        nameGap: 50,
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
      series,
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
          const isOptimum = optimum && point.value[0] === optimum[0] && point.value[1] === optimum[1];
          const label = isOptimum ? ' (OPTIMUM)' : '';
          return `${variable}: ${formatEngineering(point.value[0])}<br/>${metric}: ${formatEngineering(point.value[1])}${label}`;
        },
      },
    };
  }, [variable, metric, data, optimum]);

  return (
    <div className="border border-gray-700 rounded-lg bg-gray-900 p-2">
      <ReactECharts option={option} style={{ height: '350px' }} />
    </div>
  );
};
