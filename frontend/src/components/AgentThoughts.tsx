import React, { useState, useRef, useEffect } from 'react';

export interface Thought {
  type: string;
  content: string;
}

interface AgentThoughtsProps {
  steps: Thought[] | null;
  defaultCollapsed: boolean;
  isFinalizing?: boolean;
}

const AgentThoughts: React.FC<AgentThoughtsProps> = ({ steps, defaultCollapsed, isFinalizing }) => {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIsCollapsed(defaultCollapsed);
  }, [defaultCollapsed]);

  useEffect(() => {
    const container = contentRef.current;
    if (container && !isCollapsed) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [steps, isCollapsed]);

  if (!steps || steps.length === 0) {
    return null;
  }

  const getPrefix = (type: string) => {
    const typeMap: { [key: string]: string } = {
      thought: '[Анализ]',
      tool_call: '[Действие]',
      tool_result: '[Результат]',
      error: '[Ошибка]',
    };
    return typeMap[type] || '➡️ [Шаг]';
  };

  return (
    <div className={`agent-thoughts-container ${isFinalizing ? 'collapsing' : ''}`}>
      <div className="agent-thoughts-header" onClick={() => setIsCollapsed(!isCollapsed)}>
        <h5>Мыслительный процесс</h5>
        <button>{isCollapsed ? 'Развернуть' : 'Свернуть'}</button>
      </div>

      <div 
        className={`agent-thoughts-content ${isCollapsed ? 'collapsed' : ''}`} 
        ref={contentRef}
        style={{
          maxHeight: '300px', 
          overflowY: 'auto',
          display: isCollapsed ? 'none' : 'block',
          backgroundColor: '#2d2d2d',
          padding: '10px',
          borderRadius: '8px',
        }}
      >
        {steps.map((step, index) => (
          <div key={index} className="thought-step">
            <span style={{ whiteSpace: 'pre-wrap' }}>{`${getPrefix(step.type)} ${step.content}`}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentThoughts;
