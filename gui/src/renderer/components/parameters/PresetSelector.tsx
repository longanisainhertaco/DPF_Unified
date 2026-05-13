import React from 'react';

interface PresetSelectorProps {
  presets: Array<{
    name: string;
    description: string;
    device: string;
    geometry: string;
    source_scope_status?: string;
    source_scope_note?: string;
    validation_scope?: string;
  }>;
  selectedPreset: string | null;
  onSelect: (name: string) => void;
  loading?: boolean;
}

export const PresetSelector: React.FC<PresetSelectorProps> = ({
  presets,
  selectedPreset,
  onSelect,
  loading = false,
}) => {
  const selected = presets.find((p) => p.name === selectedPreset);
  const scopeLabel = selected?.validation_scope
    ? `${selected.source_scope_status ?? 'not_validation_evidence'} · ${selected.validation_scope}`
    : selected?.source_scope_status;

  return (
    <div>
      <label className="dpf-label text-xs mb-2 block">Device Preset</label>
      <select
        value={selectedPreset || 'custom'}
        onChange={(e) => { if (e.target.value !== 'custom') onSelect(e.target.value); }}
        disabled={loading}
        className={`dpf-input w-full font-mono text-sm ${
          loading ? 'opacity-50 cursor-wait' : ''
        }`}
      >
        <option value="custom">Custom Configuration</option>
        <optgroup label="Available Presets">
          {presets.map((preset) => (
            <option key={preset.name} value={preset.name}>
              {preset.name} — {preset.device} ({preset.geometry})
            </option>
          ))}
        </optgroup>
      </select>
      {selectedPreset && (
        <div className="mt-2 px-3 py-2 bg-[#1E1E1E] rounded border border-[#333333]">
          <p className="dpf-label text-xs text-[#999999]">
            {selected?.description || ''}
          </p>
          {scopeLabel && (
            <p
              className="dpf-label mt-2 text-xs text-accent-amber"
              title={selected?.source_scope_note || scopeLabel}
            >
              {scopeLabel}
            </p>
          )}
        </div>
      )}
    </div>
  );
};
