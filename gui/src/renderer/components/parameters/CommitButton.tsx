import React from 'react';

interface CommitButtonProps {
  isValid: boolean;
  isArmed: boolean;
  isRunning: boolean;
  onArm: () => void;
  onFire: () => void;
  onStop: () => void;
}

export const CommitButton: React.FC<CommitButtonProps> = ({
  isValid,
  isArmed,
  isRunning,
  onArm,
  onFire,
  onStop,
}) => {
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
      disabled={!isValid}
      className={`w-full py-3 rounded-lg font-mono font-bold text-sm tracking-wider border-2 transition-all ${
        isValid
          ? 'border-[#666666] text-[#CCCCCC] hover:border-[#999999] hover:bg-[#2A2A2A]'
          : 'border-[#333333] text-[#555555] cursor-not-allowed opacity-50'
      }`}
    >
      ARM
    </button>
  );
};
