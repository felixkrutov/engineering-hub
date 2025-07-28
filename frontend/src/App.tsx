import React, { useState, useEffect, useRef } from 'react';
import { ClipLoader } from 'react-spinners';
import { FaPaperPlane, FaBars, FaTimes } from 'react-icons/fa';
import { v4 as uuidv4 } from 'uuid';
import './App.css';

interface Chat {
  id: number;
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
  const [conversationId, setConversationId] = useState<string>('');
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

  useEffect(() => {
    setConversationId(uuidv4());
  }, []);

  const getChats = async () => fetch('/mossaassistant/api/chats').then(res => res.json());
  const renameChat = (chatId: number, newTitle: string) => fetch(`/mossaassistant/api/chats/${chatId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_title: newTitle })
  });
  const deleteChat = (chatId: number) => fetch(`/mossaassistant/api/chats/${chatId}`, { method: 'DELETE' });
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

  const loadChats = async () => {
    try {
      const data = await getChats();
      setChats(data.chats || []);
    } catch (error) {
      console.error("Ошибка загрузки чатов:", error);
    }
  };

  const handleSendMessage = async () => {
    const message = userInput.trim();
    if (!message || isLoading) return;

    setIsLoading(true);
    const userMessage: Message = { role: 'user', content: message };
    setMessages(prev => [...prev, userMessage]);
    setUserInput('');

    try {
        const response = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, conversation_id: conversationId })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        const modelMessage: Message = { role: 'model', content: data.reply };
        setMessages(prev => [...prev, modelMessage]);

    } catch (error) {
        const errorMessage: Message = { role: 'error', content: 'Sorry, something went wrong. Please try again.' };
        setMessages(prev => [...prev, errorMessage]);
    } finally {
        setIsLoading(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setConversationId(uuidv4());
  };

  const adjustTextareaHeight = () => {
    if (userInputRef.current) {
        userInputRef.current.style.height = 'auto';
        userInputRef.current.style.height = `${userInputRef.current.scrollHeight}px`;
    }
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

  const handleRenameChat = async (chatId: number, currentTitle: string) => {
    const newTitle = await showModal({
        title: 'Переименовать чат',
        message: 'Введите новое название для этого чата.',
        showInput: true,
        inputValue: currentTitle,
        confirmText: 'Сохранить'
    });
    if (typeof newTitle === 'string' && newTitle.trim() && newTitle.trim() !== currentTitle) {
        renameChat(chatId, newTitle.trim()).then(res => {
            if (res.ok) loadChats();
        });
    }
  };

  const handleDeleteChat = async (chatId: number) => {
    const confirmed = await showModal({
        title: 'Удалить чат?',
        message: 'Вы уверены, что хотите удалить этот чат? Это действие необратимо.',
        confirmText: 'Удалить'
    });
    if (confirmed) {
        deleteChat(chatId).then(res => {
            if (res.ok) {
                startNewChat();
                loadChats();
            }
        });
    }
  };

  return (
    <div className={`app-wrapper ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`} data-theme={theme}>
      <button className="sidebar-toggle" onClick={() => setSidebarCollapsed(!sidebarCollapsed)}>
        {sidebarCollapsed ? <FaBars /> : <FaTimes />}
      </button>

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
                <li key={chat.id} className={`chat-list-item`} onClick={() => {}}>
                    <span className="chat-title">{chat.title}</span>
                    <div className="chat-actions">
                        <button title="Переименовать" onClick={(e) => { e.stopPropagation(); handleRenameChat(chat.id, chat.title); }}><i className="bi bi-pencil"></i></button>
                        <button title="Удалить" onClick={(e) => { e.stopPropagation(); handleDeleteChat(chat.id); }}><i className="bi bi-trash"></i></button>
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
                    <p>Начните новый диалог</p>
                </div>
            ) : (
                messages.map((msg, index) => (
                    <div key={index} className={`message-block ${msg.role}`}>
                        <p>{msg.content}</p>
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
