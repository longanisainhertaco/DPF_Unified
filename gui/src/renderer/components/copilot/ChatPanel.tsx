import React, { useState, useRef, useEffect } from 'react';
import { useAIStore } from '../../stores/ai';

export const ChatPanel: React.FC = () => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const chatMessages = useAIStore((s) => s.chatMessages);
  const chatStatus = useAIStore((s) => s.chatStatus);
  const sendChat = useAIStore((s) => s.sendChat);
  const clearChat = useAIStore((s) => s.clearChat);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleSend = async () => {
    const question = input.trim();
    if (!question || chatStatus === 'sending') return;
    setInput('');
    await sendChat(question);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
    inputRef.current?.focus();
  };

  return (
    <div className="dpf-panel flex flex-col" style={{ minHeight: '280px' }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="dpf-label text-xs">ASK WALRUS</div>
        {chatMessages.length > 0 && (
          <button
            onClick={clearChat}
            className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-3 mb-3 max-h-64">
        {chatMessages.length === 0 && (
          <div className="text-xs text-gray-500 italic text-center py-4">
            Ask about physics, sweeps, predictions, or optimization.
            <br />
            Try: &quot;help&quot; or &quot;what is bremsstrahlung?&quot;
          </div>
        )}

        {chatMessages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[85%] rounded-lg px-3 py-2 text-xs ${
                msg.role === 'user'
                  ? 'bg-[#003344] text-cyan-300 border border-[#005566]'
                  : 'bg-[#2A2A2A] text-gray-300 border border-[#333333]'
              }`}
            >
              {/* Intent badge for assistant messages */}
              {msg.role === 'assistant' && msg.intent && msg.intent !== 'unknown' && (
                <div className="mb-1">
                  <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-mono uppercase bg-[#1a1a1a] text-gray-500 border border-[#444]">
                    {msg.intent}
                  </span>
                </div>
              )}

              {/* Message content — preserve line breaks */}
              <div className="whitespace-pre-wrap font-mono leading-relaxed">
                {msg.content}
              </div>

              {/* Suggestion chips */}
              {msg.role === 'assistant' && msg.suggestions && msg.suggestions.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {msg.suggestions.map((suggestion, i) => (
                    <button
                      key={i}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className="text-[10px] px-2 py-0.5 rounded-full border border-[#444] text-gray-400 hover:text-cyan-400 hover:border-cyan-700 transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Typing indicator */}
        {chatStatus === 'sending' && (
          <div className="flex justify-start">
            <div className="bg-[#2A2A2A] border border-[#333333] rounded-lg px-3 py-2">
              <div className="flex gap-1">
                <span className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask WALRUS..."
          className="dpf-input flex-1 text-xs"
          disabled={chatStatus === 'sending'}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || chatStatus === 'sending'}
          className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
            input.trim() && chatStatus !== 'sending'
              ? 'bg-[#00E5FF] text-[#121212] hover:brightness-110'
              : 'bg-[#2A2A2A] text-[#666666] cursor-not-allowed'
          }`}
        >
          Send
        </button>
      </div>
    </div>
  );
};
