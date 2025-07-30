import React, { useState, useRef, useEffect } from 'react';

interface Step {
  type: string;
  content: string;
}

interface AgentThoughtsProps {
  steps: Step[] | null;
}

const AgentThoughts: React.FC<AgentThoughtsProps> = ({ steps }) => {
  const [isCollapsed, setIsCollapsed] = useState(true);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [steps]);

  if (!steps || steps.length === 0) {
    return null;
  }

  const getPrefix = (type: string) => {
    const typeMap: { [key: string]: string } = {
      thought: '[Анализ]',
      tool_call: '[Действие]',
      tool_result: '[Результат]',
    };
    return typeMap[type] || '[Шаг]';
  };

  return (
    <div className="agent-thoughts-container">
      <div className="agent-thoughts-header" onClick={() => setIsCollapsed(!isCollapsed)}>
        <h5>Мыслительный процесс</h5>
        <button>{isCollapsed ? 'Развернуть' : 'Свернуть'}</button>
      </div>
      {!isCollapsed && (
        <div className="agent-thoughts-content" ref={contentRef}>
          {steps.map((step, index) => (
            <div key={index}>
              {`${getPrefix(step.type)} ${step.content}`}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AgentThoughts;
