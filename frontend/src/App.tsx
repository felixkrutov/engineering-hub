import React, { useState, useEffect, useRef } from 'react';
import { ClipLoader } from 'react-spinners';
import { FaPaperPlane, FaBars, FaTimes, FaPencilAlt, FaTrashAlt, FaSun, FaMoon, FaCog } from 'react-icons/fa';
import { v4 as uuidv4 } from 'uuid';
import AgentThoughts, { Thought } from './components/AgentThoughts';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/hub/api';

interface Chat {
  id: string;
  title: string;
}

interface Message {
  id:string;
  jobId?: string; // Unique ID for the currently running job
  role: 'user' | 'model' | 'error';
  content: string;
  displayedContent: string;
  thinking_steps?: Thought[];
  sources?: string[];
}

interface ModalState {
  visible: boolean;
  title: string;
  message: string;
  showInput: boolean;
  inputValue: string;
  confirmText: string;
  onConfirm: (value: string | boolean | null) => void;
}

interface KnowledgeBaseFile {
  id: string;
  name: string;
}

// --- New Nested Configuration Types ---
interface AgentSettings {
  model_name: string;
  system_prompt: string;
}

interface AppConfig {
  executor: AgentSettings;
  controller: AgentSettings;
}

const user = { username: 'Engineer', theme: 'dark' };

function App() {
  const [theme, setTheme] = useState(user.theme);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [isAgentMode, setIsAgentMode] = useState(false);
  
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const [activeSettingsTab, setActiveSettingsTab] = useState('ai');

  // --- Refactored State for Nested Config ---
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [dirtyConfig, setDirtyConfig] = useState<AppConfig | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  
  const [kbFiles, setKbFiles] = useState<KnowledgeBaseFile[]>([]);
  const [isKbFilesLoading, setIsKbFilesLoading] = useState(false);
  const [kbFilesError, setKbFilesError] = useState<string | null>(null);

  const [activeFileId, setActiveFileId] = useState<string | null>(null);

  const [modalState, setModalState] = useState<ModalState>({
    visible: false, title: '', message: '', showInput: false, inputValue: '', confirmText: 'OK', onConfirm: () => {},
  });

  const chatContainerRef = useRef<HTMLDivElement>(null);
  const userInputRef = useRef<HTMLTextAreaElement>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);


  useEffect(() => {
    if (isSettingsModalOpen && config) {
        setDirtyConfig(JSON.parse(JSON.stringify(config))); // Deep copy
    }
  }, [isSettingsModalOpen, config]);

  useEffect(() => {
    const fetchFiles = async () => {
      setIsKbFilesLoading(true);
      setKbFilesError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/kb/files`);
        if (!response.ok) throw new Error('Failed to fetch file list');
        const data = await response.json();
        setKbFiles(data.map((file: any) => ({ id: file.id, name: file.name })));
      } catch (error) {
        console.error("Failed to fetch KB files:", error);
        setKbFilesError("Не удалось загрузить список файлов.");
      } finally {
        setIsKbFilesLoading(false);
      }
    };

    if (isSettingsModalOpen && activeSettingsTab === 'db') {
      fetchFiles();
    }
  }, [isSettingsModalOpen, activeSettingsTab]);

  const loadConfig = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/v1/config`);
      if (!response.ok) throw new Error('Failed to load config');
      const data: AppConfig = await response.json();
      setConfig(data);
      setDirtyConfig(JSON.parse(JSON.stringify(data)));
    } catch (error) {
      console.error("Could not load config:", error);
    }
  };

  const handleSaveSettings = async () => {
    if (!dirtyConfig) return;
    setIsSaving(true);
    try {
      const response = await fetch(`${API_BASE_URL}/v1/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(dirtyConfig),
      });
      if (!response.ok) throw new Error('Failed to save settings');
      setConfig(JSON.parse(JSON.stringify(dirtyConfig)));
    } catch(error) {
      console.error("Save settings failed:", error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleUseFile = (fileId: string, fileName: string) => {
    setActiveFileId(fileId);
    setUserInput(`Проанализируй файл "${fileName}" по запросу: `);
    setIsSettingsModalOpen(false);
    userInputRef.current?.focus();
  };
  
  const loadChats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/v1/chats`);
      if (!response.ok) throw new Error('Failed to fetch chats');
      const data = await response.json();
      setChats(data);
    } catch (error) {
      console.error("Ошибка загрузки чатов:", error);
    }
  };

  const startPolling = (jobId: string, isNewChat: boolean) => {
      if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
      }

      const poll = async () => {
          try {
              const statusResponse = await fetch(`${API_BASE_URL}/v1/jobs/${jobId}/status`);
              if (!statusResponse.ok) {
                  throw new Error(`Polling failed with status: ${statusResponse.status}`);
              }
              const jobStatus = await statusResponse.json();

              setMessages(currentMessages => currentMessages.map(msg => {
                  if (msg.jobId === jobId) { // KEY CHANGE: Find by job_id
                      const updatedMsg: Message = { 
                          ...msg, 
                          thinking_steps: jobStatus.thoughts,
                          content: (jobStatus.status === 'complete') ? (jobStatus.final_answer || '') : msg.content,
                          role: (jobStatus.status === 'failed') ? 'error' : msg.role,
                          jobId: msg.jobId // Keep jobId by default
                      };
                      // If the job is done, remove the jobId to make it a static message
                      if (['complete', 'failed', 'cancelled'].includes(jobStatus.status)) {
                          delete updatedMsg.jobId; 
                          if (jobStatus.status === 'failed') {
                              updatedMsg.content = 'Обработка задачи завершилась с ошибкой.';
                          }
                      }
                      return updatedMsg;
                  }
                  return msg;
              }));

              if (['complete', 'failed', 'cancelled'].includes(jobStatus.status)) {
                  if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
                  setIsLoading(false);
                  setCurrentJobId(null);
              }
          } catch (error) {
              console.error('Polling error:', error);
              if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
              setIsLoading(false);
              setCurrentJobId(null);
              setMessages(currentMessages => currentMessages.map(msg =>
                  msg.jobId === jobId ? { ...msg, role: 'error', content: 'Ошибка при получении статуса задачи.', displayedContent: 'Ошибка при получении статуса задачи.' } : msg
              ));
          }
      };

      pollIntervalRef.current = setInterval(poll, 2000);
  };

  const selectChat = async (chatId: string) => {
    if (isLoading && chatId !== currentChatId) return;
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    setIsLoading(true);
    setCurrentChatId(chatId);
    setMessages([]); // Clear previous messages

    try {
        const historyRes = await fetch(`${API_BASE_URL}/v1/chats/${chatId}`);
        if (!historyRes.ok) throw new Error(`Failed to fetch chat history: ${historyRes.statusText}`);
        
        const historyData = await historyRes.json();
        const historyMessages: Message[] = historyData.map((m: any, index: number) => ({
            id: `${chatId}-${index}`,
            role: m.role,
            content: m.content || '', // Use the 'content' field from the backend
            displayedContent: m.content || '',
            thinking_steps: m.thinking_steps || [],
            sources: m.sources || [] // Defensively get sources, default to empty array
        }));
        
        const activeJobRes = await fetch(`${API_BASE_URL}/v1/chats/${chatId}/active_job`);
        if (!activeJobRes.ok) throw new Error(`Failed to check for active job: ${activeJobRes.statusText}`);

        const { job_id } = await activeJobRes.json();

        if (job_id) {
            const jobStatusRes = await fetch(`${API_BASE_URL}/v1/jobs/${job_id}/status`);
            if (!jobStatusRes.ok) throw new Error(`Failed to get job status for ${job_id}: ${jobStatusRes.statusText}`);

            const jobStatus = await jobStatusRes.json();
            
            const modelMessage: Message = {
                id: uuidv4(), role: 'model', content: '', displayedContent: '',
                thinking_steps: jobStatus.thoughts,
                jobId: job_id
            };

            setMessages([...historyMessages, modelMessage]);
            setCurrentJobId(job_id);

            if (!['complete', 'failed', 'cancelled'].includes(jobStatus.status)) {
                setIsLoading(true);
                startPolling(job_id, false);
            } else {
                setIsLoading(false);
            }
        } else {
            setMessages(historyMessages);
            setIsLoading(false);
            setCurrentJobId(null);
        }
    } catch (error) {
        console.error("Failed to select chat:", error);
        setMessages([{ id: uuidv4(), role: 'error', content: 'Не удалось загрузить этот чат.', displayedContent: 'Не удалось загрузить этот чат.' }]);
        setIsLoading(false);
    }
  };


  const startNewChat = () => {
    setCurrentChatId(null);
    setCurrentJobId(null);
    setMessages([]);
    setActiveFileId(null);
  };
  
  const handleRenameChat = async (chatId: string, currentTitle: string) => {
    const newTitle = await showModal({ title: 'Переименовать чат', message: 'Введите новое название для этого чата.', showInput: true, inputValue: currentTitle, confirmText: 'Сохранить' });
    if (typeof newTitle === 'string' && newTitle.trim() && newTitle.trim() !== currentTitle) {
        try {
            await fetch(`${API_BASE_URL}/v1/chats/${chatId}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ new_title: newTitle.trim() }) });
            await loadChats();
        } catch (error) { console.error("Error renaming chat:", error); }
    }
  };

  const handleDeleteChat = async (chatId: string) => {
    const confirmed = await showModal({ title: 'Удалить чат?', message: 'Вы уверены, что хотите удалить этот чат? Это действие необратимо.', confirmText: 'Удалить' });
    if (confirmed) {
        try {
            await fetch(`${API_BASE_URL}/v1/chats/${chatId}`, { method: 'DELETE' });
            if (currentChatId === chatId) startNewChat();
            await loadChats();
        } catch (error) { console.error("Error deleting chat:", error); }
    }
  };

  useEffect(() => {
    loadChats();
    loadConfig();
  }, []);

  const scrollToBottom = () => {
    chatContainerRef.current?.scrollTo({ top: chatContainerRef.current.scrollHeight, behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    const messageText = userInput.trim();
    if (!messageText || isLoading) return;

    // --- Optimistic UI Update ---
    const userMessage: Message = {
      id: `local-${uuidv4()}`,
      role: 'user',
      content: messageText,
      displayedContent: messageText,
    };
    // Add user message to UI immediately and clear input
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setUserInput('');
    setIsLoading(true);
    // -----------------------------

    if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
    }

    let conversationId = currentChatId;
    const isNewChat = !conversationId;

    try {
        if (isNewChat) {
            const chatResponse = await fetch(`${API_BASE_URL}/v1/chats`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: messageText.substring(0, 50) || "Новый чат" }),
            });
            if (!chatResponse.ok) throw new Error('Failed to create a new chat.');
            
            const newChatInfo: Chat = await chatResponse.json();
            conversationId = newChatInfo.id;
            setCurrentChatId(conversationId); // Set the new chat as active
            await loadChats(); // Refresh the sidebar
        }

        if (!conversationId) throw new Error("Missing conversation ID to create a job.");

        // Backend job creation
        const jobResponse = await fetch(`${API_BASE_URL}/v1/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: messageText, conversation_id: conversationId,
                file_id: activeFileId, use_agent_mode: isAgentMode,
            }),
        });
        setActiveFileId(null);
        if (!jobResponse.ok) throw new Error(`Failed to create job: ${await jobResponse.text()}`);

        const { job_id } = await jobResponse.json();
        setCurrentJobId(job_id);

        // --- Add Model Placeholder and Start Polling ---
        const modelPlaceholder: Message = {
            id: `model-${job_id}`,
            role: 'model',
            content: '',
            displayedContent: '',
            thinking_steps: [{ type: 'log', content: 'Задача поставлена в очередь...' }],
            jobId: job_id,
        };
        setMessages(prevMessages => [...prevMessages, modelPlaceholder]);
        startPolling(job_id, isNewChat);
        // ---------------------------------------------

    } catch (error) {
        console.error('Error during message sending process:', error);
        // Revert optimistic update on failure
        setMessages(prev => prev.filter(m => m.id !== userMessage.id)); 
        setUserInput(messageText); // Restore user input
        setIsLoading(false);
        // Optionally add a temporary error message to the UI
    }
    // NOTE: The 'finally' block is removed as setIsLoading is now handled by the polling logic.
};

  const handleCancelJob = async () => {
    if (!currentJobId) return;

    try {
      await fetch(`${API_BASE_URL}/v1/jobs/${currentJobId}/cancel`, {
        method: 'POST',
      });

      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
      setIsLoading(false);
      setCurrentJobId(null);

      // Reload the chat to reflect the cancelled state from history
      if (currentChatId) {
        selectChat(currentChatId);
      }

    } catch (error) {
      console.error("Failed to cancel job:", error);
    }
  };

  useEffect(() => {
    const messageToType = messages.find(m => m.role === 'model' && m.content.length > m.displayedContent.length);
    if (messageToType) {
      const interval = setInterval(() => {
        setMessages(currentMessages => currentMessages.map(m => {
            if (m.id === messageToType.id) {
              const nextCharIndex = m.displayedContent.length;
              if (nextCharIndex >= m.content.length) {
                clearInterval(interval);
                return m;
              }
              return { ...m, displayedContent: m.content.substring(0, nextCharIndex + 1) };
            }
            return m;
          }));
      }, 20);
      return () => clearInterval(interval);
    }
  }, [messages]);

  const adjustTextareaHeight = () => {
    if (userInputRef.current) {
        userInputRef.current.style.height = 'auto';
        userInputRef.current.style.height = `${userInputRef.current.scrollHeight}px`;
    }
  };

  const showModal = (props: Partial<Omit<ModalState, 'visible' | 'onConfirm'>>) => {
    return new Promise<string | boolean | null>((resolve) => {
      setModalState({
        visible: true, title: props.title || '', message: props.message || '',
        showInput: props.showInput || false, inputValue: props.inputValue || '',
        confirmText: props.confirmText || 'OK',
        onConfirm: (value) => { setModalState(prev => ({...prev, visible: false})); resolve(value); },
      });
    });
  };

  useEffect(scrollToBottom, [messages]);
  useEffect(adjustTextareaHeight, [userInput]);

  const handleThemeToggle = () => setTheme(theme === 'dark' ? 'light' : 'dark');
  const hasChanges = config && dirtyConfig ? JSON.stringify(config) !== JSON.stringify(dirtyConfig) : false;

  return (
    <div className={`app-wrapper ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`} data-theme={theme}>
        {sidebarCollapsed && (<button className="sidebar-reopen-btn" onClick={() => setSidebarCollapsed(false)}><FaBars /></button>)}
        <aside className="sidebar">
          <div className="sidebar-header">
            <button className="new-chat-btn" onClick={startNewChat}><i className="bi bi-plus-lg"></i> Новый чат</button>
            <button className="hide-sidebar-btn" onClick={() => setSidebarCollapsed(true)}><FaTimes /></button>
          </div>
          <ul className="chat-list">
              {chats.map(chat => (
                  <li key={chat.id} className={`chat-list-item ${chat.id === currentChatId ? 'active' : ''}`} onClick={() => selectChat(chat.id)}>
                      <span className="chat-title">{chat.title}</span>
                      <div className="chat-actions">
                          <button title="Переименовать" onClick={(e) => { e.stopPropagation(); handleRenameChat(chat.id, chat.title); }}><FaPencilAlt /></button>
                          <button title="Удалить" onClick={(e) => { e.stopPropagation(); handleDeleteChat(chat.id); }}><FaTrashAlt /></button>
                      </div>
                  </li>
              ))}
          </ul>
          <div className="sidebar-footer">
            <div className="user-info">
              <div className="user-icon">{user.username[0].toUpperCase()}</div>
              <span>{user.username}</span>
            </div>
            <div>
              <button className="theme-toggle-btn" title="Сменить тему" onClick={handleThemeToggle}>{theme === 'dark' ? <FaSun /> : <FaMoon />}</button>
              <button className="settings-btn" title="Настройки" onClick={() => setIsSettingsModalOpen(true)}><FaCog /></button>
            </div>
          </div>
        </aside>
        <main className="main-content">
          <div className="chat-area">
            <div className="chat-container" ref={chatContainerRef}>
              {messages.length === 0 && !isLoading ? (
                  <div className="welcome-screen"><h1>Mossa AI</h1><p>Начните новый диалог или выберите существующий</p></div>
              ) : (
                  messages.map((msg, index) => (
                      <div key={msg.id} className={`message-block ${msg.role} ${msg.content.length > 0 && msg.content === msg.displayedContent ? 'done' : ''}`}>
                          <div className="message-content">
                              {msg.role === 'model' && msg.thinking_steps && msg.thinking_steps.length > 0 && (
                                <AgentThoughts
                                  steps={msg.thinking_steps}
                                  defaultCollapsed={!msg.jobId}
                                />
                              )}
                              <p className="content">{msg.displayedContent}</p>
                              {/* Defensive rendering of sources */}
                              {msg.sources && msg.sources.length > 0 && (
                                <div className="message-sources">
                                  <strong>Источники: </strong>
                                  <span>{msg.sources.join(', ')}</span>
                                </div>
                              )}
                          </div>
                      </div>
                  ))
              )}
               {isLoading && messages.length > 0 && messages[messages.length-1].role !== 'model' && <div className="spinner-container"><ClipLoader color="#888" size={30} /></div>}
            </div>
            <div className="input-area-wrapper">
              <div className="input-area">
                  <div className="input-top-row">
                      <textarea
                        ref={userInputRef}
                        className="user-input"
                        placeholder="Спросите что-нибудь..."
                        rows={1}
                        value={userInput}
                        onChange={(e) => setUserInput(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); }}}
                        disabled={isLoading}
                      />
                      {isLoading ? (
                          <button className="cancel-btn" onClick={handleCancelJob} title="Отменить">
                              <FaTimes />
                          </button>
                      ) : (
                          <button className="send-btn" onClick={handleSendMessage} disabled={userInput.trim() === ''}>
                              <FaPaperPlane />
                          </button>
                      )}
                  </div>
                  <div className="input-bottom-toolbar">
                      <button
                          className={`mode-toggle-btn ${isAgentMode ? 'active' : ''}`}
                          onClick={() => setIsAgentMode(!isAgentMode)}
                      >
                          Режим агента
                      </button>
                  </div>
              </div>
            </div>
          </div>
        </main>
        {modalState.visible && (
          <div className={`modal-overlay visible`} onClick={() => modalState.onConfirm(null)}>
              <div className="modal-box" onClick={(e) => e.stopPropagation()}>
                  <h3>{modalState.title}</h3><p>{modalState.message}</p>
                  {modalState.showInput && (<input type="text" className="modal-input" value={modalState.inputValue} onChange={(e) => setModalState(prev => ({...prev, inputValue: e.target.value }))} autoFocus />)}
                  <div className="modal-actions">
                      <button className="modal-btn-cancel" onClick={() => modalState.onConfirm(null)}>Отмена</button>
                      <button className="modal-btn-confirm" onClick={() => modalState.onConfirm(modalState.showInput ? modalState.inputValue : true)}>{modalState.confirmText}</button>
                  </div>
              </div>
          </div>
        )}
        {isSettingsModalOpen && (
          <div className="modal-overlay visible">
            <div className="modal-box settings-modal">
              <div className="modal-header"><h2>Настройки</h2><button className="modal-close-btn" onClick={() => setIsSettingsModalOpen(false)}>×</button></div>
              <div className="modal-content">
                <div className="tabs">
                  <button className={`tab-btn ${activeSettingsTab === 'ai' ? 'active' : ''}`} onClick={() => setActiveSettingsTab('ai')}>Настройки ИИ</button>
                  <button className={`tab-btn ${activeSettingsTab === 'db' ? 'active' : ''}`} onClick={() => setActiveSettingsTab('db')}>База Знаний</button>
                </div>
                <div className="tab-content">
                  {activeSettingsTab === 'ai' && dirtyConfig && (
                    <div className="ai-settings">
                      <div className="settings-group">
                        <h4>Агент-Исполнитель (Gemini)</h4>
                        <label htmlFor="executor-model-name">Модель</label>
                        <input id="executor-model-name" type="text" value={dirtyConfig.executor.model_name} onChange={(e) => setDirtyConfig({...dirtyConfig, executor: {...dirtyConfig.executor, model_name: e.target.value}})} />
                        <label htmlFor="executor-system-prompt">Системный промпт</label>
                        <textarea id="executor-system-prompt" rows={6} value={dirtyConfig.executor.system_prompt} onChange={(e) => setDirtyConfig({...dirtyConfig, executor: {...dirtyConfig.executor, system_prompt: e.target.value}})} />
                      </div>
                      <div className="settings-group">
                        <h4>Агент-Контролёр (OpenAI / OpenRouter)</h4>
                        <label htmlFor="controller-model-name">Модель</label>
                        <input id="controller-model-name" type="text" value={dirtyConfig.controller.model_name} onChange={(e) => setDirtyConfig({...dirtyConfig, controller: {...dirtyConfig.controller, model_name: e.target.value}})} />
                        <label htmlFor="controller-system-prompt">Системный промпт</label>
                        <textarea id="controller-system-prompt" rows={6} value={dirtyConfig.controller.system_prompt} onChange={(e) => setDirtyConfig({...dirtyConfig, controller: {...dirtyConfig.controller, system_prompt: e.target.value}})} />
                      </div>
                    </div>
                  )}
                  {activeSettingsTab === 'db' && (
                    <div className="db-settings file-manager">
                      {kbFilesError && <p className="error-message">{kbFilesError}</p>}
                      {isKbFilesLoading ? (<div className="spinner-container"><ClipLoader color="#888" size={30} /></div>) : (
                        kbFiles.length > 0 ? (
                          <div className="kb-file-list-container">
                            <ul className="kb-file-list">
                              {kbFiles.map((file) => (
                                <li key={file.id} className="kb-file-item">
                                  <span className="kb-file-name">{file.name}</span>
                                  <button className="modal-btn-confirm kb-use-btn" onClick={() => handleUseFile(file.id, file.name)}>Использовать</button>
                                </li>
                              ))}
                            </ul>
                          </div>
                        ) : (<p>Файлы в базе знаний не найдены.</p>)
                      )}
                    </div>
                  )}
                </div>
              </div>
              <div className="modal-footer"><button className={`modal-btn-confirm ${!hasChanges || isSaving ? 'disabled' : ''}`} onClick={handleSaveSettings} disabled={!hasChanges || isSaving}>{isSaving ? <ClipLoader color="#ffffff" size={16} /> : 'Сохранить'}</button></div>
            </div>
          </div>
        )}
    </div>
  );
}

export default App;
