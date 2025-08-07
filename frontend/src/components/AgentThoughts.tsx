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

  // 1. Улучшаем эффект для плавной прокрутки вниз
  useEffect(() => {
    const container = contentRef.current;
    if (container && !isCollapsed) {
      // Используем scrollTo с опцией 'smooth' для плавной анимации
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [steps, isCollapsed]); // Эффект срабатывает при добавлении нового шага

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

      {/* 2. Применяем стили напрямую к контейнеру с контентом */}
      <div 
        className={`agent-thoughts-content ${isCollapsed ? 'collapsed' : ''}`} 
        ref={contentRef}
        style={{
          // Задаем максимальную высоту. Блок не будет расти больше этого значения.
          maxHeight: '300px', 
          // Добавляем вертикальную прокрутку, если содержимое не помещается.
          overflowY: 'auto',
          // Скрываем контент, если блок свернут (для плавности анимации).
          // Обрати внимание: этот стиль работает вместе с твоим классом `collapsed`.
          // Если твой CSS для `collapsed` уже делает это, то можно убрать. Но так надежнее.
          display: isCollapsed ? 'none' : 'block',
          // Дополнительные стили для красоты
          backgroundColor: '#2d2d2d',
          padding: '10px',
          borderRadius: '8px',
        }}
      >
        {steps.map((step, index) => (
          <div key={index} className="thought-step">
            {/* Используем getPrefix для отображения иконки и текста */}
            <span style={{ whiteSpace: 'pre-wrap' }}>{`${getPrefix(step.type)} ${step.content}`}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentThoughts;
