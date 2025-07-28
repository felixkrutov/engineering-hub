import React, { useState, useEffect, useRef } from 'react';
import { ClipLoader } from 'react-spinners';
import { FaPaperPlane, FaBars, FaTimes, FaPencilAlt, FaTrashAlt, FaSun, FaMoon, FaCog } from 'react-icons/fa';
import { v4 as uuidv4 } from 'uuid';
import './App.css';

interface Chat {
  id: string;
  title: string;
}

interface Message {
  role: 'user' | 'model' | 'error';
  content: string;
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

  const loadConfig = async () => {
    try {
      const response = await fetch('/api/v1/config');
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
      const response = await fetch('/api/v1/config', {
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
  
  const loadChats = async () => {
    try {
      const response = await fetch('/api/v1/chats');
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
      const response = await fetch(`/api/v1/chats/${chatId}`);
      if (!response.ok) throw new Error("Chat history not found");
      const data: Message[] = await response.json();
      setMessages(data);
    } catch(error) {
      console.error("Failed to select chat:", error);
      setMessages([{ role: 'error', content: 'Could not load this chat.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const startNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
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
            const response = await fetch(`/api/v1/chats/${chatId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_title: newTitle.trim() })
            });
            if (!response.ok) throw new Error('Failed to rename chat');
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
            const response = await fetch(`/api/v1/chats/${chatId}`, { method: 'DELETE' });
            if (!response.ok) throw new Error('Failed to delete chat');
            
            if (currentChatId === chatId) {
                startNewChat();
            }
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

  const updateTheme = (newTheme: string) => {
    console.log(`Theme updated to ${newTheme}`);
  };

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  const handleSendMessage = async () => {
    const message = userInput.trim();
    if (!message || isLoading) return;

    setIsLoading(true);
    const userMessage: Message = { role: 'user', content: message };
    setMessages(prev => [...prev, userMessage]);
    setUserInput('');
    
    const isNewChat = currentChatId === null;
    const conversationId = currentChatId || uuidv4();

    try {
        const response = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, conversation_id: conversationId })
        });

        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        const modelMessage: Message = { role: 'model', content: data.reply };
        setMessages(prev => [...prev, modelMessage]);

        if (isNewChat) {
          setCurrentChatId(conversationId);
          await loadChats();
        }

    } catch (error) {
        const errorMessage: Message = { role: 'error', content: 'Sorry, something went wrong. Please try again.' };
        setMessages(prev => [...prev, errorMessage]);
    } finally {
        setIsLoading(false);
    }
  };

  const adjustTextareaHeight = () => {
    if (userInputRef.current) {
        userInputRef.current.style.height = 'auto';
        userInputRef.current.style.height = `${userInputRef.current.scrollHeight}px`;
    }
  };

  const showModal = (props: Partial<Omit<ModalState, 'visible' | 'onConfirm'>>) => {
    return new Promise<string | boolean | null>((resolve) => {
      setModalState({
        visible: true,
        title: props.title || '',
        message: props.message || '',
        showInput: props.showInput || false,
        inputValue: props.inputValue || '',
        confirmText: props.confirmText || 'OK',
        onConfirm: (value) => {
          setModalState(prev => ({...prev, visible: false}));
          resolve(value);
        },
      });
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    adjustTextareaHeight();
  }, [userInput]);

  const handleThemeToggle = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    updateTheme(newTheme);
  };
  
  const hasChanges = dirtyModelName !== savedModelName || dirtySystemPrompt !== savedSystemPrompt;

  return (
    <div className={`app-wrapper ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`} data-theme={theme}>
        {sidebarCollapsed && (
          <button className="sidebar-reopen-btn" onClick={() => setSidebarCollapsed(false)}>
            <FaBars />
          </button>
        )}

      <aside className="sidebar">
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={startNewChat}>
            <i className="bi bi-plus-lg"></i> Новый чат
          </button>
           <button className="hide-sidebar-btn" onClick={() => setSidebarCollapsed(true)}>
            <FaTimes />
          </button>
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
            <button className="theme-toggle-btn" title="Сменить тему" onClick={handleThemeToggle}>
              {theme === 'dark' ? <FaSun /> : <FaMoon />}
            </button>
            <button className="settings-btn" title="Настройки" onClick={() => setIsSettingsModalOpen(true)}>
              <FaCog />
            </button>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <div className="chat-area">
          <div className="chat-container" ref={chatContainerRef}>
            {messages.length === 0 ? (
                <div className="welcome-screen">
                    <h1>Mossa AI</h1>
                    <p>Начните новый диалог или выберите существующий</p>
                </div>
            ) : (
                messages.map((msg, index) => (
                    <div key={index} className={`message-block ${msg.role}`}>
                        <div className="message-content">
                          <p>{msg.content}</p>
                        </div>
                    </div>
                ))
            )}
          </div>
          <div className="input-area-wrapper">
            <div className="input-area">
              <textarea
                ref={userInputRef}
                className="user-input"
                placeholder="Спросите что-нибудь..."
                rows={1}
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                    }
                }}
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
                <h3>{modalState.title}</h3>
                <p>{modalState.message}</p>
                {modalState.showInput && (
                    <input
                        type="text"
                        className="modal-input"
                        value={modalState.inputValue}
                        onChange={(e) => setModalState(prev => ({...prev, inputValue: e.target.value }))}
                        autoFocus
                    />
                )}
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
            <div className="modal-header">
              <h2>Настройки</h2>
              <button className="modal-close-btn" onClick={() => setIsSettingsModalOpen(false)}>×</button>
            </div>
            <div className="modal-content">
              <div className="tabs">
                <button className={`tab-btn ${activeSettingsTab === 'ai' ? 'active' : ''}`} onClick={() => setActiveSettingsTab('ai')}>Настройки ИИ</button>
                <button className={`tab-btn ${activeSettingsTab === 'db' ? 'active' : ''}`} onClick={() => setActiveSettingsTab('db')}>База данных</button>
              </div>
              <div className="tab-content">
                {activeSettingsTab === 'ai' && (
                  <div className="ai-settings">
                    <label htmlFor="model-name">Модель</label>
                    <input id="model-name" type="text" value={dirtyModelName} onChange={(e) => setDirtyModelName(e.target.value)} />
                    <label htmlFor="system-prompt">Системный промпт</label>
                    <textarea id="system-prompt" rows={10} value={dirtySystemPrompt} onChange={(e) => setDirtySystemPrompt(e.target.value)} />
                  </div>
                )}
                {activeSettingsTab === 'db' && (
                  <div className="db-settings">
                    <p>Управление базой данных будет доступно здесь.</p>
                  </div>
                )}
              </div>
            </div>
            <div className="modal-footer">
               <button
                className={`modal-btn-confirm ${!hasChanges || isSaving ? 'disabled' : ''}`}
                onClick={handleSaveSettings}
                disabled={!hasChanges || isSaving}
              >
                {isSaving ? <ClipLoader color="#ffffff" size={16} /> : 'Сохранить'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
