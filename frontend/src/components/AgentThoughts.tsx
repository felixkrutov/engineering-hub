import React, { useState, useRef, useEffect } from 'react';

interface Step {
  type: string;
  content: string;
}

interface AgentThoughtsProps {
  steps: Step[] | null;
  defaultCollapsed: boolean;
  isFinalizing?: boolean;
}

const AgentThoughts: React.FC<AgentThoughtsProps> = ({ steps, defaultCollapsed, isFinalizing }) => {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (contentRef.current && !isCollapsed) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
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
    return typeMap[type] || '[Шаг]';
  };

  return (
    <div className={`agent-thoughts-container ${isFinalizing ? 'collapsing' : ''}`}>
      <div className="agent-thoughts-header" onClick={() => setIsCollapsed(!isCollapsed)}>
        <h5>Мыслительный процесс</h5>
        <button>{isCollapsed ? 'Развернуть' : 'Свернуть'}</button>
      </div>
      <div className={`agent-thoughts-content ${isCollapsed ? 'collapsed' : ''}`} ref={contentRef}>
        {steps.map((step, index) => (
          <div key={index} className="thought-step">
            {`${getPrefix(step.type)} ${step.content}`}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentThoughts;
