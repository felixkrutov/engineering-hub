import React, { useState, useEffect, useRef } from 'react';
import { ClipLoader } from 'react-spinners';
import { FaPaperPlane, FaBars, FaTimes, FaPencilAlt, FaTrashAlt, FaSun, FaMoon, FaCog } from 'react-icons/fa';
import { v4 as uuidv4 } from 'uuid';
import AgentThoughts from './components/AgentThoughts';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/hub/api';

interface Chat {
  id: string;
  title: string;
}

interface ThinkingStep {
  type: string;
  content: string;
}

interface Message {
  id: string;
  role: 'user' | 'model' | 'error';
  content: string;
  displayedContent: string;
  thinking_steps?: ThinkingStep[];
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

// New interface for files from the knowledge base
interface KnowledgeBaseFile {
  id: string;
  name: string;
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
  
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const [activeSettingsTab, setActiveSettingsTab] = useState('ai');

  const [dirtyModelName, setDirtyModelName] = useState('');
  const [dirtySystemPrompt, setDirtySystemPrompt] = useState('');
  const [savedModelName, setSavedModelName] = useState('');
  const [savedSystemPrompt, setSavedSystemPrompt] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  
  // State for the new file manager
  const [kbFiles, setKbFiles] = useState<KnowledgeBaseFile[]>([]);
  const [isKbFilesLoading, setIsKbFilesLoading] = useState(false);
  const [kbFilesError, setKbFilesError] = useState<string | null>(null);

  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[] | null>(null);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [isFinalizing, setIsFinalizing] = useState<string | null>(null);

  const [modalState, setModalState] = useState<ModalState>({
    visible: false,
    title: '',
    message: '',
    showInput: false,
    inputValue: '',
    confirmText: 'OK',
    onConfirm: () => {},
  });

  const chatContainerRef = useRef<HTMLDivElement>(null);
  const userInputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (isSettingsModalOpen) {
        setDirtyModelName(savedModelName);
        setDirtySystemPrompt(savedSystemPrompt);
    }
  }, [isSettingsModalOpen, savedModelName, savedSystemPrompt]);

  // Effect to load files when the "База данных" tab is active
  useEffect(() => {
    const fetchFiles = async () => {
      setIsKbFilesLoading(true);
      setKbFilesError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/kb/files`);
        if (!response.ok) throw new Error('Failed to fetch file list');
        const data = await response.json();
        // Assuming the backend returns an array of objects with 'id' and 'name'
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
      const config = await response.json();
      setSavedModelName(config.model_name);
      setDirtyModelName(config.model_name);
      setSavedSystemPrompt(config.system_prompt);
      setDirtySystemPrompt(config.system_prompt);
    } catch (error) {
      console.error("Could not load config:", error);
    }
  };

  const handleSaveSettings = async () => {
    setIsSaving(true);
    try {
      const response = await fetch(`${API_BASE_URL}/v1/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: dirtyModelName,
          system_prompt: dirtySystemPrompt,
        }),
      });
      if (!response.ok) throw new Error('Failed to save settings');
      setSavedModelName(dirtyModelName);
      setSavedSystemPrompt(dirtySystemPrompt);
    } catch(error) {
      console.error("Save settings failed:", error);
    } finally {
      setIsSaving(false);
    }
  };

  // Updated handleUseFile to set a prompt template and activeFileId
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
  
  const selectChat = async (chatId: string) => {
    if (isLoading) return;
    setIsLoading(true);
    setCurrentChatId(chatId);
    try {
      const response = await fetch(`${API_BASE_URL}/v1/chats/${chatId}`);
      if (!response.ok) throw new Error("Chat history not found");
      const rawMessages: any[] = await response.json();
      const formattedMessages: Message[] = rawMessages.map(m => ({
          id: uuidv4(),
          role: m.role,
          content: m.content,
          displayedContent: m.content,
          thinking_steps: m.thinking_steps
      }));
      setMessages(formattedMessages);
    } catch(error) {
      console.error("Failed to select chat:", error);
      setMessages([{ id: uuidv4(), role: 'error', content: 'Could not load this chat.', displayedContent: 'Could not load this chat.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const startNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
    setActiveFileId(null); // Ensure active file is cleared for a new chat
  };
  
  const handleRenameChat = async (chatId: string, currentTitle: string) => {
    const newTitle = await showModal({
        title: 'Переименовать чат',
        message: 'Введите новое название для этого чата.',
        showInput: true,
        inputValue: currentTitle,
        confirmText: 'Сохранить'
    });
    if (typeof newTitle === 'string' && newTitle.trim() && newTitle.trim() !== currentTitle) {
        try {
            await fetch(`${API_BASE_URL}/v1/chats/${chatId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_title: newTitle.trim() })
            });
            await loadChats();
        } catch (error) {
            console.error("Error renaming chat:", error);
        }
    }
  };

  const handleDeleteChat = async (chatId: string) => {
    const confirmed = await showModal({
        title: 'Удалить чат?',
        message: 'Вы уверены, что хотите удалить этот чат? Это действие необратимо.',
        confirmText: 'Удалить'
    });
    if (confirmed) {
        try {
            await fetch(`${API_BASE_URL}/v1/chats/${chatId}`, { method: 'DELETE' });
            if (currentChatId === chatId) startNewChat();
            await loadChats();
        } catch (error) {
            console.error("Error deleting chat:", error);
        }
    }
  };

  useEffect(() => {
    loadChats();
    loadConfig();
  }, []);

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  const handleSendMessage = async () => {
    const messageText = userInput.trim();
    if (!messageText || isLoading) return;

    setIsLoading(true);
    setUserInput('');

    const userMessage: Message = { id: uuidv4(), role: 'user', content: messageText, displayedContent: messageText };
    const isNewChat = currentChatId === null;
    const conversationId = currentChatId || uuidv4();
    
    const tempModelMessageId = uuidv4();
    const tempModelMessage: Message = { id: tempModelMessageId, role: 'model', content: '', displayedContent: '' };
    setMessages(prev => [...prev, userMessage, tempModelMessage]);
    setStreamingMessageId(tempModelMessageId);
    
    const liveThinkingSteps: ThinkingStep[] = [];
    setThinkingSteps(liveThinkingSteps);

    try {
      const response = await fetch(`${API_BASE_URL}/v1/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageText,
          conversation_id: conversationId,
          file_id: activeFileId, // Pass the active file ID to the backend
        }),
      });
      setActiveFileId(null); // Reset after sending

      if (!response.ok || !response.body) throw new Error(`Streaming failed: ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          if (part.startsWith('data: ')) {
            const jsonString = part.substring(6);
            try {
              const data = JSON.parse(jsonString);
              if (data.type === 'final_answer') {
                setIsFinalizing(tempModelMessageId);
                setMessages(currentMessages => 
                    currentMessages.map(m => 
                        m.id === tempModelMessageId 
                        ? { ...m, content: data.content, thinking_steps: liveThinkingSteps } 
                        : m
                    )
                );
                setTimeout(() => {
                    setThinkingSteps(null);
                    setStreamingMessageId(null);
                    setIsFinalizing(null);
                }, 500);
                if (isNewChat) {
                    setCurrentChatId(conversationId);
                    await loadChats();
                }
              } else {
                liveThinkingSteps.push(data);
                setThinkingSteps([...liveThinkingSteps]);
              }
            } catch (e) { console.error("JSON parse error:", e, "Payload:", jsonString); }
          }
        }
      }
    } catch (error) {
        console.error("Streaming chat failed:", error);
        setMessages(prev => prev.map(msg =>
            msg.id === tempModelMessageId ? { ...msg, role: 'error', content: 'Ошибка потоковой передачи ответа.', displayedContent: 'Ошибка потоковой передачи ответа.' } : msg
        ));
    } finally {
        setIsLoading(false);
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
          })
        );
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
        onConfirm: (value) => {
          setModalState(prev => ({...prev, visible: false}));
          resolve(value);
        },
      });
    });
  };

  useEffect(scrollToBottom, [messages, thinkingSteps, isFinalizing]);
  useEffect(adjustTextareaHeight, [userInput]);

  const handleThemeToggle = () => setTheme(theme === 'dark' ? 'light' : 'dark');
  const hasChanges = dirtyModelName !== savedModelName || dirtySystemPrompt !== savedSystemPrompt;

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
            {messages.length === 0 ? (
                <div className="welcome-screen"><h1>Mossa AI</h1><p>Начните новый диалог или выберите существующий</p></div>
            ) : (
                messages.map(msg => (
                    <div key={msg.id} className={`message-block ${msg.role} ${msg.content.length > 0 && msg.content === msg.displayedContent ? 'done' : ''}`}>
                        <div className="message-content">
                            {msg.thinking_steps && msg.thinking_steps.length > 0 && <AgentThoughts steps={msg.thinking_steps} defaultCollapsed={true} />}
                            <p className="content">{msg.displayedContent}</p>
                            {msg.id === streamingMessageId && thinkingSteps && <AgentThoughts steps={thinkingSteps} defaultCollapsed={false} isFinalizing={isFinalizing === msg.id} />}
                        </div>
                    </div>
                ))
            )}
          </div>
          <div className="input-area-wrapper">
            <div className="input-area">
              <textarea
                ref={userInputRef} className="user-input" placeholder="Спросите что-нибудь..." rows={1}
                value={userInput} onChange={(e) => setUserInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); }}}
              />
              <button className="send-btn" onClick={handleSendMessage} disabled={userInput.trim() === '' || isLoading}>
                {isLoading ? <ClipLoader color="#ffffff" size={20} /> : <FaPaperPlane />}
              </button>
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
                {activeSettingsTab === 'ai' && (
                  <div className="ai-settings">
                    <label htmlFor="model-name">Модель</label><input id="model-name" type="text" value={dirtyModelName} onChange={(e) => setDirtyModelName(e.target.value)} />
                    <label htmlFor="system-prompt">Системный промпт</label><textarea id="system-prompt" rows={10} value={dirtySystemPrompt} onChange={(e) => setDirtySystemPrompt(e.target.value)} />
                  </div>
                )}
                {activeSettingsTab === 'db' && (
                  <div className="db-settings file-manager">
                    {kbFilesError && <p className="error-message">{kbFilesError}</p>}
                    {isKbFilesLoading ? (<div className="spinner-container"><ClipLoader color="#888" size={30} /></div>) : (
                      kbFiles.length > 0 ? (
                        <ul className="kb-file-list">
                          {kbFiles.map((file) => (
                            <li key={file.id} className="kb-file-item">
                              <span className="kb-file-name">{file.name}</span>
                              <button className="kb-file-use-btn" onClick={() => handleUseFile(file.id, file.name)}>
                                Использовать
                              </button>
                            </li>
                          ))}
                        </ul>
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
