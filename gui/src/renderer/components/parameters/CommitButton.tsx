import React from 'react';

interface CommitButtonProps {
  isValid: boolean;
  isArmed: boolean;
  isRunning: boolean;
  finished: boolean;
  onArm: () => void;
  onFire: () => void;
  onStop: () => void;
  onReset: () => void;
}

export const CommitButton: React.FC<CommitButtonProps> = ({
  isValid,
  isArmed,
  isRunning,
  finished,
  onArm,
  onFire,
  onStop,
  onReset,
}) => {
  if (finished) {
    return (
      <button
        onClick={onReset}
        className="w-full py-3 rounded-lg font-mono font-bold text-sm tracking-wider border-2 border-[#00E5FF] text-[#00E5FF] hover:bg-[#00E5FF] hover:text-[#121212] transition-all"
      >
        RESET
      </button>
    );
  }

  if (isRunning) {
    return (
      <button
        onClick={onStop}
        className="w-full py-3 rounded-lg font-mono font-bold text-sm tracking-wider bg-[#FF5252] hover:bg-[#FF6B6B] text-white transition-all shadow-lg"
      >
        STOP
      </button>
    );
  }

  if (isArmed) {
    return (
      <button
        onClick={onFire}
        className="w-full py-3 rounded-lg font-mono font-bold text-sm tracking-wider bg-[#00E5FF] hover:bg-[#00D4E8] text-[#121212] transition-all animate-pulse shadow-[0_0_20px_rgba(0,229,255,0.5)]"
      >
        FIRE
      </button>
    );
  }

  return (
    <button
      onClick={onArm}
      className="w-full py-3 rounded-lg font-mono font-bold text-sm tracking-wider border-2 border-[#666666] text-[#CCCCCC] hover:border-[#999999] hover:bg-[#2A2A2A] transition-all"
    >
      ARM
    </button>
  );
};
