import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';

interface TraceData {
  name: string;
  color: string;
  data: Array<[number, number]>; // [time, value] pairs
  yAxisIndex?: number;
}

interface TraceChartProps {
  title: string;
  traces: TraceData[];
  height?: number;
  yAxisLabel?: string;
  yAxisLabel2?: string; // For dual y-axis
  logScale?: boolean;
}

function formatEngineering(value: number): string {
  if (Math.abs(value) === 0) return '0';
  const absVal = Math.abs(value);
  if (absVal >= 1e12) return (value / 1e12).toFixed(1) + 'T';
  if (absVal >= 1e9) return (value / 1e9).toFixed(1) + 'G';
  if (absVal >= 1e6) return (value / 1e6).toFixed(1) + 'M';
  if (absVal >= 1e3) return (value / 1e3).toFixed(1) + 'k';
  if (absVal >= 1) return value.toFixed(1);
  if (absVal >= 1e-3) return (value * 1e3).toFixed(1) + 'm';
  if (absVal >= 1e-6) return (value * 1e6).toFixed(1) + 'μ';
  if (absVal >= 1e-9) return (value * 1e9).toFixed(1) + 'n';
  return value.toExponential(1);
}

export const TraceChart: React.FC<TraceChartProps> = ({
  title,
  traces,
  height = 300,
  yAxisLabel,
  yAxisLabel2,
  logScale = false,
}) => {
  const option = useMemo(() => {
    const hasDualAxis = traces.some((t) => t.yAxisIndex === 1);

    return {
      backgroundColor: '#0A0A0A',
      title: {
        text: title,
        left: 'center',
        top: 10,
        textStyle: {
          color: '#888888',
          fontFamily: 'Inter, sans-serif',
          fontSize: 12,
          fontWeight: 'normal',
        },
      },
      grid: {
        left: 80,
        right: hasDualAxis ? 80 : 50,
        top: 50,
        bottom: 50,
        backgroundColor: '#0A0A0A',
        borderColor: '#1A1A1A',
      },
      xAxis: {
        type: 'value',
        name: 'Time (s)',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          color: '#888888',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 11,
        },
        axisLine: {
          lineStyle: { color: '#1A1A1A' },
        },
        axisLabel: {
          color: '#888888',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 10,
          formatter: formatEngineering,
        },
        splitLine: {
          lineStyle: { color: '#1A1A1A', width: 1 },
        },
      },
      yAxis: hasDualAxis
        ? [
            {
              type: logScale ? 'log' : 'value',
              name: yAxisLabel || '',
              nameLocation: 'middle',
              nameGap: 50,
              nameTextStyle: {
                color: '#888888',
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: 11,
              },
              axisLine: {
                lineStyle: { color: '#1A1A1A' },
              },
              axisLabel: {
                color: '#888888',
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: 10,
                formatter: formatEngineering,
              },
              splitLine: {
                lineStyle: { color: '#1A1A1A', width: 1 },
              },
            },
            {
              type: logScale ? 'log' : 'value',
              name: yAxisLabel2 || '',
              nameLocation: 'middle',
              nameGap: 50,
              nameTextStyle: {
                color: '#888888',
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: 11,
              },
              axisLine: {
                lineStyle: { color: '#1A1A1A' },
              },
              axisLabel: {
                color: '#888888',
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: 10,
                formatter: formatEngineering,
              },
              splitLine: {
                show: false,
              },
            },
          ]
        : {
            type: logScale ? 'log' : 'value',
            name: yAxisLabel || '',
            nameLocation: 'middle',
            nameGap: 50,
            nameTextStyle: {
              color: '#888888',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 11,
            },
            axisLine: {
              lineStyle: { color: '#1A1A1A' },
            },
            axisLabel: {
              color: '#888888',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 10,
              formatter: formatEngineering,
            },
            splitLine: {
              lineStyle: { color: '#1A1A1A', width: 1 },
            },
          },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        borderColor: '#003300',
        textStyle: {
          color: '#00E5FF',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 11,
        },
        axisPointer: {
          type: 'cross',
          lineStyle: {
            color: '#003300',
            width: 1,
          },
        },
        formatter: (params: any) => {
          if (!Array.isArray(params)) return '';
          let result = `Time: ${formatEngineering(params[0].value[0])}s<br/>`;
          params.forEach((p: any) => {
            result += `${p.seriesName}: ${formatEngineering(p.value[1])}<br/>`;
          });
          return result;
        },
      },
      series: traces.map((trace) => ({
        name: trace.name,
        type: 'line',
        data: trace.data,
        yAxisIndex: trace.yAxisIndex || 0,
        lineStyle: {
          color: trace.color,
          width: 2,
        },
        itemStyle: {
          color: trace.color,
        },
        showSymbol: false,
        animation: false,
      })),
    };
  }, [title, traces, yAxisLabel, yAxisLabel2, logScale]);

  return (
    <div
      className="scope-container"
      style={{
        filter: `drop-shadow(0 0 2px ${traces[0]?.color || '#00E5FF'})`,
      }}
    >
      <ReactECharts option={option} style={{ height: `${height}px`, width: '100%' }} />
    </div>
  );
};
