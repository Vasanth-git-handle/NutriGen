import React from "react";

const MessageBubble = ({ sender, text }) => {
  return (
    <div className={`bubble ${sender === "user" ? "user" : "bot"}`}>
      {text}
    </div>
  );
};

export default MessageBubble;
