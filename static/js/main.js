const API = "http://localhost:8000";
const chat = document.getElementById("chat");
const msgInput = document.getElementById("msg");
const sendBtn = document.getElementById("sendBtn");

let messageCount = 0;

// Initial Jarvis greeting
window.addEventListener("DOMContentLoaded", () => {
    addMessage("Jarvis", "Hello! I'm Jarvis, your AI assistant. How can I help you today?", "jarvis");
});

// Send message on button click
sendBtn.addEventListener("click", sendMessage);

// Send message on Enter key
msgInput.addEventListener("keypress", function(e) {
    if (e.key === "Enter") sendMessage();
});

// Add message to chat
function addMessage(sender, text, type) {
    const div = document.createElement("div");
    div.className = `message ${type}`;
    div.innerHTML = `<b>${sender}:</b> ${text}`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    messageCount++;
}

// Send message function
async function sendMessage() {
    const text = msgInput.value.trim();
    if (!text) return;

    addMessage("You", text, "user");
    msgInput.value = "";

    // Show "thinking..." placeholder
    const thinkingDiv = document.createElement("div");
    thinkingDiv.className = "message jarvis";
    thinkingDiv.innerHTML = "<b>Jarvis:</b> Thinking...";
    chat.appendChild(thinkingDiv);
    chat.scrollTop = chat.scrollHeight;

    try {
        const res = await fetch(`${API}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        const data = await res.json();

        thinkingDiv.remove();
        addMessage("Jarvis", data.response, "jarvis");
    } catch (err) {
        thinkingDiv.remove();
        addMessage("Jarvis", `Error: ${err.message}`, "jarvis");
    }
}
