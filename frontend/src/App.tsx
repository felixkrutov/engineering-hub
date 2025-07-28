import React, { useState, useEffect, useRef } from 'react';
import { ClipLoader } from 'react-spinners';
import { FaPaperPlane, FaBars, FaTimes } from 'react-icons/fa';
import { v4 as uuidv4 } from 'uuid';
import './App.css';

interface Message {
  role: 'user' | 'model' | 'error';
  content: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string>('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setConversationId(uuidv4());
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = async () => {
    const trimmedInput = userInput.trim();
    if (!trimmedInput || isLoading || !conversationId) return;

    const userMessage: Message = { role: 'user', content: trimmedInput };
    setMessages(prev => [...prev, userMessage]);
    setUserInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: trimmedInput,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
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

  return (
    <div className={`app-wrapper ${!isSidebarOpen ? 'sidebar-closed' : ''}`}>
      <aside className="sidebar">
        <div className="sidebar-header">
          <button className="new-chat-button">Новый чат</button>
        </div>
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
                <p>Start a conversation</p>
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
