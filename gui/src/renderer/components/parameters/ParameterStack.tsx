import React, { useState } from 'react';
import { ScientificInput } from './ScientificInput';
import { CommitButton } from './CommitButton';
import { PresetSelector } from './PresetSelector';

export const ParameterStack: React.FC = () => {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    preset: true,
    bank: true,
    geometry: true,
    gas: true,
    grid: true,
    physics: true,
  });

  const [isArmed, setIsArmed] = useState(false);
  const [isRunning, setIsRunning] = useState(false);

  // Placeholder state for parameters
  const [params, setParams] = useState({
    preset: null,
    C: 2.7e-5,
    V0: 3.5e4,
    L0: 5.0e-8,
    R0: 1.0e-3,
    ESR: 5.0e-3,
    ESL: 1.0e-8,
    anodeRadius: 0.025,
    cathodeRadius: 0.05,
    geometryType: 'cylindrical',
    rho0: 1.67e-5,
    T0: 3.0e2,
    nx: 64,
    ny: 64,
    nz: 64,
    dx: 0.001,
    simTime: 1.0e-6,
    enableResistive: true,
    enableViscosity: false,
    enablePowell: true,
  });

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const handleArm = () => setIsArmed(true);
  const handleFire = () => {
    setIsRunning(true);
    setIsArmed(false);
    // TODO: Trigger simulation via store
  };
  const handleStop = () => {
    setIsRunning(false);
    setIsArmed(false);
    // TODO: Stop simulation via store
  };

  const updateParam = (key: string, value: any) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  const presetData = [
    {
      name: 'PF-400J',
      description: 'Mather-type DPF, 400J bank',
      device: 'Mather',
      geometry: 'cylindrical',
    },
    {
      name: 'NX2',
      description: 'Filippov-type DPF, 2kJ bank',
      device: 'Filippov',
      geometry: 'cylindrical',
    },
  ];

  return (
    <div className="h-full flex flex-col bg-[#1E1E1E]">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[#333333]">
        <h2 className="dpf-label text-sm font-semibold tracking-wide">PARAMETERS</h2>
      </div>

      {/* Scrollable Parameter Sections */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {/* Preset Selector */}
        <Section
          title="Preset"
          expanded={expandedSections.preset}
          onToggle={() => toggleSection('preset')}
        >
          <PresetSelector
            presets={presetData}
            selectedPreset={params.preset}
            onSelect={(name) => updateParam('preset', name)}
          />
        </Section>

        {/* Bank */}
        <Section
          title="Bank"
          expanded={expandedSections.bank}
          onToggle={() => toggleSection('bank')}
        >
          <div className="space-y-2">
            <ScientificInput
              label="Capacitance"
              value={params.C}
              unit="F"
              onChange={(v) => updateParam('C', v)}
              validation="valid"
            />
            <ScientificInput
              label="Charge Voltage"
              value={params.V0}
              unit="V"
              onChange={(v) => updateParam('V0', v)}
              validation="valid"
            />
            <ScientificInput
              label="Inductance"
              value={params.L0}
              unit="H"
              onChange={(v) => updateParam('L0', v)}
              validation="valid"
            />
            <ScientificInput
              label="Resistance"
              value={params.R0}
              unit="Ω"
              onChange={(v) => updateParam('R0', v)}
              validation="valid"
            />
            <ScientificInput
              label="ESR"
              value={params.ESR}
              unit="Ω"
              onChange={(v) => updateParam('ESR', v)}
              validation="none"
            />
            <ScientificInput
              label="ESL"
              value={params.ESL}
              unit="H"
              onChange={(v) => updateParam('ESL', v)}
              validation="none"
            />
          </div>
        </Section>

        {/* Geometry */}
        <Section
          title="Geometry"
          expanded={expandedSections.geometry}
          onToggle={() => toggleSection('geometry')}
        >
          <div className="space-y-2">
            <div className="mb-3">
              <label className="dpf-label text-xs mb-2 block">Type</label>
              <div className="flex gap-2">
                <button
                  onClick={() => updateParam('geometryType', 'cylindrical')}
                  className={`flex-1 py-2 px-3 rounded text-xs font-medium transition-all ${
                    params.geometryType === 'cylindrical'
                      ? 'bg-[#00E5FF] text-[#121212]'
                      : 'bg-[#2A2A2A] text-[#999999] hover:bg-[#333333]'
                  }`}
                >
                  Cylindrical
                </button>
                <button
                  onClick={() => updateParam('geometryType', 'cartesian')}
                  className={`flex-1 py-2 px-3 rounded text-xs font-medium transition-all ${
                    params.geometryType === 'cartesian'
                      ? 'bg-[#00E5FF] text-[#121212]'
                      : 'bg-[#2A2A2A] text-[#999999] hover:bg-[#333333]'
                  }`}
                >
                  Cartesian
                </button>
              </div>
            </div>
            <ScientificInput
              label="Anode Radius"
              value={params.anodeRadius}
              unit="m"
              onChange={(v) => updateParam('anodeRadius', v)}
              validation="valid"
            />
            <ScientificInput
              label="Cathode Radius"
              value={params.cathodeRadius}
              unit="m"
              onChange={(v) => updateParam('cathodeRadius', v)}
              validation="valid"
            />
          </div>
        </Section>

        {/* Gas */}
        <Section
          title="Gas"
          expanded={expandedSections.gas}
          onToggle={() => toggleSection('gas')}
        >
          <div className="space-y-2">
            <ScientificInput
              label="Initial Density"
              value={params.rho0}
              unit="kg/m³"
              onChange={(v) => updateParam('rho0', v)}
              validation="valid"
            />
            <ScientificInput
              label="Initial Temperature"
              value={params.T0}
              unit="K"
              onChange={(v) => updateParam('T0', v)}
              validation="valid"
            />
          </div>
        </Section>

        {/* Grid */}
        <Section
          title="Grid"
          expanded={expandedSections.grid}
          onToggle={() => toggleSection('grid')}
        >
          <div className="space-y-2">
            <div>
              <label className="dpf-label text-xs mb-2 block">Grid Shape [nx, ny, nz]</label>
              <div className="flex gap-2">
                <input
                  type="number"
                  value={params.nx}
                  onChange={(e) => updateParam('nx', parseInt(e.target.value))}
                  className="dpf-input flex-1 text-center font-mono"
                  min="8"
                />
                <input
                  type="number"
                  value={params.ny}
                  onChange={(e) => updateParam('ny', parseInt(e.target.value))}
                  className="dpf-input flex-1 text-center font-mono"
                  min="8"
                />
                <input
                  type="number"
                  value={params.nz}
                  onChange={(e) => updateParam('nz', parseInt(e.target.value))}
                  className="dpf-input flex-1 text-center font-mono"
                  min="8"
                />
              </div>
            </div>
            <ScientificInput
              label="Grid Spacing"
              value={params.dx}
              unit="m"
              onChange={(v) => updateParam('dx', v)}
              validation="valid"
            />
            <ScientificInput
              label="Simulation Time"
              value={params.simTime}
              unit="s"
              onChange={(v) => updateParam('simTime', v)}
              validation="valid"
            />
          </div>
        </Section>

        {/* Physics */}
        <Section
          title="Physics"
          expanded={expandedSections.physics}
          onToggle={() => toggleSection('physics')}
        >
          <div className="space-y-2">
            <ToggleSwitch
              label="Resistive MHD"
              checked={params.enableResistive}
              onChange={(checked) => updateParam('enableResistive', checked)}
            />
            <ToggleSwitch
              label="Viscosity"
              checked={params.enableViscosity}
              onChange={(checked) => updateParam('enableViscosity', checked)}
            />
            <ToggleSwitch
              label="Powell Divergence Cleaning"
              checked={params.enablePowell}
              onChange={(checked) => updateParam('enablePowell', checked)}
            />
          </div>
        </Section>
      </div>

      {/* Footer: Commit Button */}
      <div className="px-4 py-4 border-t border-[#333333]">
        <CommitButton
          isValid={true}
          isArmed={isArmed}
          isRunning={isRunning}
          onArm={handleArm}
          onFire={handleFire}
          onStop={handleStop}
        />
      </div>
    </div>
  );
};

// Section Header Component
interface SectionProps {
  title: string;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

const Section: React.FC<SectionProps> = ({ title, expanded, onToggle, children }) => (
  <div className="bg-[#2A2A2A] rounded-lg border border-[#333333] overflow-hidden">
    <button
      onClick={onToggle}
      className="w-full px-3 py-2 flex items-center justify-between hover:bg-[#333333] transition-colors"
    >
      <span className="dpf-label text-xs font-semibold tracking-wide">{title}</span>
      <svg
        className={`w-4 h-4 text-[#666666] transition-transform ${
          expanded ? 'rotate-180' : ''
        }`}
        fill="currentColor"
        viewBox="0 0 20 20"
      >
        <path
          fillRule="evenodd"
          d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
          clipRule="evenodd"
        />
      </svg>
    </button>
    {expanded && <div className="px-3 pb-3">{children}</div>}
  </div>
);

// Toggle Switch Component
interface ToggleSwitchProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ label, checked, onChange }) => (
  <div className="flex items-center justify-between">
    <span className="dpf-label text-xs">{label}</span>
    <button
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
        checked ? 'bg-[#00E5FF]' : 'bg-[#333333]'
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          checked ? 'translate-x-5' : 'translate-x-1'
        }`}
      />
    </button>
  </div>
);
