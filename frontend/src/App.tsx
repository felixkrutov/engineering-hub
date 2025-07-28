import React, { useState, useEffect, useRef } from 'react';
import { ClipLoader } from 'react-spinners';
import { FaPaperPlane, FaBars, FaTimes } from 'react-icons/fa';
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

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const loadChats = async () => {
    try {
      const response = await fetch('/api/v1/chats');
      const data = await response.json();
      setChats(data);
    } catch (error) {
      console.error("Failed to load chats:", error);
    }
  };

  const selectChat = async (chatId: string) => {
    if (isLoading) return;
    setIsLoading(true);
    try {
      const response = await fetch(`/api/v1/chats/${chatId}`);
      if (!response.ok) throw new Error("Chat not found");
      const historyData = await response.json();
      const formattedMessages: Message[] = historyData.map((item: any) => ({
        role: item.role,
        content: item.parts[0],
      }));
      setMessages(formattedMessages);
      setCurrentChatId(chatId);
    } catch (error) {
      console.error("Failed to load chat history:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const startNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
  };

  useEffect(() => {
    loadChats();
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

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
        body: JSON.stringify({ message: message, conversation_id: conversationId }),
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
      const errorMessage: Message = { role: 'error', content: 'Sorry, something went wrong.' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`app-wrapper ${!isSidebarOpen ? 'sidebar-closed' : ''}`}>
      <aside className="sidebar">
        <div className="sidebar-header">
          <button className="new-chat-button" onClick={startNewChat}>New Chat</button>
        </div>
        <ul className="chat-list">
          {chats.map(chat => (
            <li
              key={chat.id}
              className={`chat-list-item ${chat.id === currentChatId ? 'active' : ''}`}
              onClick={() => selectChat(chat.id)}
            >
              {chat.title}
            </li>
          ))}
        </ul>
      </aside>

      <main className="main-content">
        <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="sidebar-toggle">
          {isSidebarOpen ? <FaTimes /> : <FaBars />}
        </button>

        <div className="chat-area">
          <div className="chat-container" ref={chatContainerRef}>
            {messages.length === 0 ? (
              <div className="welcome-screen">
                <h1>Gemini Chat</h1>
                <p>Select a chat or start a new one.</p>
              </div>
            ) : (
              messages.map((msg, index) => (
                <div key={index} className={`message-block ${msg.role}`}>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))
            )}
          </div>
          <div className="input-area-wrapper">
            <div className="input-area">
              <textarea
                className="user-input"
                placeholder="Ask me anything..."
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
              <button className="send-btn" onClick={handleSendMessage} disabled={!userInput.trim() || isLoading}>
                {isLoading ? <ClipLoader color="#ffffff" size={20} /> : <FaPaperPlane />}
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
