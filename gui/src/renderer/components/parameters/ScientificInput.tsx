import React from 'react';

interface ScientificInputProps {
  label: string;
  value: number;
  unit: string;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  validation?: 'valid' | 'warning' | 'error' | 'none';
  disabled?: boolean;
}

export const ScientificInput: React.FC<ScientificInputProps> = ({
  label,
  value,
  unit,
  onChange,
  min,
  max,
  validation = 'none',
  disabled = false,
}) => {
  const formatScientific = (num: number): string => {
    if (num === 0) return '0';
    const exp = Math.floor(Math.log10(Math.abs(num)));
    if (exp >= -2 && exp <= 3) {
      return num.toString();
    }
    return num.toExponential(2);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const parsed = parseFloat(e.target.value);
    if (!isNaN(parsed)) {
      const clamped =
        min !== undefined && max !== undefined
          ? Math.max(min, Math.min(max, parsed))
          : min !== undefined
          ? Math.max(min, parsed)
          : max !== undefined
          ? Math.min(max, parsed)
          : parsed;
      onChange(clamped);
    }
  };

  const getBorderColor = () => {
    switch (validation) {
      case 'valid':
        return 'border-[#4CAF50]';
      case 'warning':
        return 'border-[#FFC107]';
      case 'error':
        return 'border-[#FF5252]';
      default:
        return 'border-[#333333]';
    }
  };

  return (
    <div>
      <label className="dpf-label text-xs mb-1 block">{label}</label>
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={formatScientific(value)}
          onChange={handleChange}
          disabled={disabled}
          className={`dpf-input flex-1 font-mono text-sm ${getBorderColor()} ${
            disabled ? 'opacity-50 cursor-not-allowed' : ''
          }`}
          min={min}
          max={max}
        />
        <span className="dpf-label text-xs text-[#999999] w-12 text-right">{unit}</span>
      </div>
    </div>
  );
};
