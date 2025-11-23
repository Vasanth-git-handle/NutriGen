import React from "react";

const Sidebar = () => {
  return (
    <div className="sidebar">
      <h2>NUTRIGEN</h2>
      <button className="new-chat-btn">+ New chat</button>

      <div className="history">
        <p>Your conversations</p>
      </div>

      <div className="profile">
        <p>Settings</p>
        <p>Vasanth</p>
      </div>
    </div>
  );
};

export default Sidebar;
