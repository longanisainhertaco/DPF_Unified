import React from 'react';
import { useAIStore } from '../../stores/ai';

const formatRelativeTime = (timestamp: number): string => {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);

  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
};

const SeverityIcon: React.FC<{ severity: 'info' | 'warning' | 'critical' }> = ({ severity }) => {
  if (severity === 'info') {
    return <div className="w-2 h-2 rounded-full bg-cyan-400" title="Info" />;
  }

  if (severity === 'warning') {
    return (
      <div className="w-4 h-4 flex items-center justify-center text-amber-400" title="Warning">
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
        </svg>
      </div>
    );
  }

  // critical
  return (
    <div className="w-4 h-4 flex items-center justify-center text-red-500" title="Critical">
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
      </svg>
    </div>
  );
};

export const AdvisoryPanel: React.FC = () => {
  const advisories = useAIStore(state => state.advisories);
  const scrollRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new advisories arrive
  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [advisories]);

  return (
    <div className="dpf-panel h-full flex flex-col">
      <div className="dpf-label text-sm mb-3">ADVISORIES</div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto space-y-2 pr-2"
        style={{ maxHeight: 'calc(100% - 2rem)' }}
      >
        {advisories.length === 0 ? (
          <div className="text-gray-500 text-sm text-center py-8">
            No advisories yet
          </div>
        ) : (
          advisories.map((advisory) => (
            <div
              key={advisory.id}
              className="border border-gray-700 rounded p-3 bg-gray-900 space-y-2"
            >
              <div className="flex items-start gap-2">
                <SeverityIcon severity={advisory.severity} />
                <div className="flex-1 text-gray-300 text-sm leading-relaxed">
                  {advisory.message}
                </div>
              </div>
              <div className="text-gray-600 text-xs font-mono">
                {formatRelativeTime(advisory.timestamp)}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
