import React from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { ParameterStack } from '../parameters/ParameterStack';
import { OscilloscopePanel } from '../scope/OscilloscopePanel';
import { PostShotPanel } from '../analysis/PostShotPanel';
import { CoPilotSidebar } from './CoPilotSidebar';
import { useSimulationStore } from '../../stores/simulation';
import { useConfigStore } from '../../stores/config';

interface DashboardShellProps {
  coPilotOpen: boolean;
}

export const DashboardShell: React.FC<DashboardShellProps> = ({ coPilotOpen }) => {
  const simStatus = useSimulationStore((s) => s.status);
  const scalarHistory = useSimulationStore((s) => s.scalarHistory);
  const config = useConfigStore((s) => s.config);

  const showPostShot = simStatus === 'finished' && scalarHistory.length > 0;

  return (
    <div className="flex-1 overflow-hidden">
      <PanelGroup direction="horizontal" className="h-full">
        {/* Left Panel: Parameters */}
        <Panel
          defaultSize={20}
          minSize={15}
          maxSize={30}
        >
          <div className="h-full overflow-y-auto border-r border-dpf-border bg-dpf-panel">
            <ParameterStack />
          </div>
        </Panel>

        <PanelResizeHandle className="w-1 bg-dpf-panel hover:bg-dpf-border transition-colors cursor-col-resize" />

        {/* Center Panel: Oscilloscope + Post-Shot */}
        <Panel defaultSize={coPilotOpen ? 60 : 80} minSize={40}>
          <div className="h-full flex flex-col bg-dpf-bg">
            {showPostShot ? (
              <div className="flex-1 overflow-y-auto p-4">
                <PostShotPanel
                  scalarHistory={scalarHistory}
                  capacitance={config.circuit?.C ?? 1e-6}
                  voltage={config.circuit?.V0 ?? 1e3}
                />
              </div>
            ) : (
              <OscilloscopePanel />
            )}
          </div>
        </Panel>

        {/* Right Panel: AI Co-Pilot (Conditional) */}
        {coPilotOpen && (
          <>
            <PanelResizeHandle className="w-1 bg-dpf-panel hover:bg-dpf-border transition-colors cursor-col-resize" />
            <Panel
              defaultSize={20}
              minSize={15}
              maxSize={30}
            >
              <div className="h-full overflow-y-auto border-l border-dpf-border bg-dpf-panel">
                <CoPilotSidebar />
              </div>
            </Panel>
          </>
        )}
      </PanelGroup>
    </div>
  );
};
