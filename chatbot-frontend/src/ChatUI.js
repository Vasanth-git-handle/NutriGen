import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import ScrollToBottom from "react-scroll-to-bottom";
import MessageBubble from "./MessageBubble";
import Sidebar from "./Sidebar";

const API_URL = "http://127.0.0.1:8000/ask"; // FastAPI endpoint

const ChatUI = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await axios.post(API_URL, {
        user_id: "default_user",
        question: input,
      });

      // Simulate typing delay
      setTimeout(() => {
        setIsTyping(false);
        const botMessage = {
          sender: "bot",
          text: response.data.answer,
        };
        setMessages((prev) => [...prev, botMessage]);
      }, 1000);
    } catch (error) {
      setIsTyping(false);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Server Error. Try again." },
      ]);
    }
  };

  return (
    <div className="chat-wrapper">
      <Sidebar />
      <div className="chat-box">
        <ScrollToBottom className="messages-container">
          {messages.map((msg, i) => (
            <MessageBubble key={i} sender={msg.sender} text={msg.text} />
          ))}

          {isTyping && (
            <div className="typing-indicator">
              <span>Bot is typing...</span>
            </div>
          )}
        </ScrollToBottom>

        <div className="input-area">
          <input
            type="text"
            placeholder="What's in your mind?"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button onClick={sendMessage}>➤</button>
        </div>
      </div>
    </div>
  );
};

export default ChatUI;
