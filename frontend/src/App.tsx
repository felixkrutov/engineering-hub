import React, { useState, useEffect, useRef } from 'react';
import './App.css';

interface Chat {
  id: number;
  title: string;
}

interface Message {
  role: 'user' | 'ai' | 'ai-loading';
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
  const [currentChatId, setCurrentChatId] = useState<number | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [userInput, setUserInput] = useState('');

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

  const getChats = async () => fetch('/mossaassistant/api/chats').then(res => res.json());
  const getMessages = (chatId: number) => fetch(`/mossaassistant/api/chats/${chatId}/messages`).then(res => res.json());
  const sendMessage = (message: string, chatId: number | null) => fetch('/mossaassistant/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_message: message, chat_id: chatId })
  }).then(res => res.json());
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

  const selectChat = async (chatId: number) => {
    if (isSending) return;
    setCurrentChatId(chatId);
    setMessages([]);
    try {
      const data = await getMessages(chatId);
      setMessages(data.messages || []);
    } catch (error) {
      console.error("Ошибка загрузки сообщений:", error);
      setMessages([{ role: 'ai', content: 'Ошибка загрузки чата.' }]);
    }
  };

  const handleSendMessage = async () => {
    const message = userInput.trim();
    if (!message || isSending) return;

    setIsSending(true);
    
    const lastMessage = messages[messages.length - 1];
    const isConsecutiveUserMessage = lastMessage && lastMessage.role === 'user';
    
    setMessages(prev => [...prev, { role: 'user', content: message }]);
    setUserInput('');

    setTimeout(() => {
        setMessages(prev => [...prev, { role: 'ai-loading', content: 'Думаю...' }]);
    }, 10);
    
    try {
        const response = await sendMessage(message, currentChatId);
        if (response.detail) throw new Error(response.detail);

        setMessages(prev => prev.filter(m => m.role !== 'ai-loading'));
        
        const lastMessageAfterLoading = messages[messages.length-1];
        const isConsecutiveAIMessage = lastMessageAfterLoading && lastMessageAfterLoading.role === 'ai';
        
        setMessages(prev => [...prev.filter(m => m.role !== 'ai-loading'), { role: 'ai', content: response.ai_response }]);

        if (response.is_new_chat) {
            setCurrentChatId(response.chat_id);
            await loadChats();
        }
    } catch (error) {
        setMessages(prev => [...prev.filter(m => m.role !== 'ai-loading'), { role: 'ai', content: `Ошибка: ${(error as Error).message}` }]);
    } finally {
        setIsSending(false);
    }
  };

  const startNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
  };

  const adjustTextareaHeight = () => {
    if (userInputRef.current) {
        userInputRef.current.style.height = 'auto';
        userInputRef.current.style.height = `${userInputRef.current.scrollHeight}px`;
    }
  };

  useEffect(() => {
    loadChats();
    startNewChat();
  }, []);
  
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
                if (currentChatId === chatId) startNewChat();
                loadChats();
            }
        });
    }
  };

  return (
    <div className={`app-wrapper ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`} data-theme={theme}>
      <button className="sidebar-toggle" onClick={() => setSidebarCollapsed(false)}>
        <i className="bi bi-layout-sidebar-inset"></i>
      </button>

      <aside className="sidebar">
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={startNewChat}>
            <i className="bi bi-plus-lg"></i> Новый чат
          </button>
          <button className="hide-sidebar-btn" onClick={() => setSidebarCollapsed(true)}>
            <i className="bi bi-layout-sidebar-inset-reverse"></i>
          </button>
        </div>
        <ul className="chat-list">
            {chats.map(chat => (
                <li key={chat.id} className={`chat-list-item ${chat.id === currentChatId ? 'active' : ''}`} onClick={() => selectChat(chat.id)}>
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
            {currentChatId === null && messages.length === 0 ? (
                <div className="welcome-screen">
                    <h1>Mossa AI</h1>
                    <p>Начните новый диалог или выберите существующий</p>
                </div>
            ) : (
                messages.map((msg, index) => {
                    const prevMessage = index > 0 ? messages[index - 1] : null;
                    const isConsecutive = prevMessage ? prevMessage.role === msg.role : false;
                    return (
                        <div key={index} className={`message-block ${msg.role} ${isConsecutive ? 'consecutive' : ''}`}>
                            {msg.role === 'user' || msg.role === 'ai-loading' ? (
                                msg.content
                            ) : (
                                msg.content.split(/\n\s*\n/).filter(p => p.trim() !== '').map((para, pIndex) => (
                                    <p key={pIndex}>{para}</p>
                                ))
                            )}
                        </div>
                    );
                })
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
              <button className="send-btn" onClick={handleSendMessage} disabled={userInput.trim() === '' || isSending}>
                <i className="bi bi-send"></i>
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
