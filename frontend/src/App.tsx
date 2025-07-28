import React, { useState, useEffect, useRef } from 'react';
import { ClipLoader } from 'react-spinners';
import { FaPaperPlane, FaBars, FaTimes, FaPencilAlt, FaTrashAlt } from 'react-icons/fa';
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
  }, []);

  const updateTheme = (newTheme: string) => fetch('/mossaassistant/api/user/theme', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ theme: newTheme })
  });

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
    updateTheme(newTheme).catch(err => console.error("Ошибка сохранения темы:", err));
  };

  const handleLogout = () => {
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = '/mossaassistant/logout';
    document.body.appendChild(form);
    form.submit();
  };

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
              {theme === 'dark' ? <i className="bi bi-sun-fill"></i> : <i className="bi bi-moon-fill"></i>}
            </button>
            <button className="logout-btn" title="Выйти" onClick={handleLogout}>
              <i className="bi bi-box-arrow-right"></i>
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
    </div>
  );
}

export default App;
