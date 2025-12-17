const API = "http://localhost:8000";

const chat = document.getElementById("chat");
const msgInput = document.getElementById("msg");
const sendBtn = document.getElementById("sendBtn");

// Add message to chat
function addMessage(sender, text, type) {
    const div = document.createElement("div");
    div.className = `message ${type}-message`;
    div.innerHTML = `
        <div class="message-sender"><strong>${sender}</strong></div>
        <div class="message-content">${text}</div>
    `;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight; // Auto-scroll
    return div;
}

// Send message
async function sendMessage() {
    const text = msgInput.value.trim();
    if (!text) return;
    
    // Add User message and clear input
    addMessage("You", text, "user");
    msgInput.value = "";

    // Show "Thinking..." placeholder
    const thinkingDiv = addMessage("Jarvis", "Thinking...", "jarvis");

    try {
        const res = await fetch(`${API}/query`, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            // Matches your QueryRequest Pydantic model in api.py
            body: JSON.stringify({ 
                text: text,
                context: {}, 
                stream: false,
                model: null 
            })
        });

        // Parse response
        const data = await res.json();

        // Check if the server returned an error (like 500 or 503)
        if (!res.ok) {
            throw new Error(data.detail || "Server error occurred");
        }

        // Remove "Thinking..." and show actual response
        thinkingDiv.remove();
        // Access 'response' key from your QueryResponse model
        addMessage("Jarvis", data.response, "jarvis");

    } catch (err) {
        thinkingDiv.remove();
        console.error("Communication Error:", err);
        addMessage("Jarvis", `⚠️ Error: ${err.message}`, "jarvis");
    }
}

// Event Listeners
sendBtn.addEventListener("click", sendMessage);

msgInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        e.preventDefault(); // Prevent line breaks in input
        sendMessage();
    }
});

// Optional: Health Check on load to verify connection
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch(`${API}/health`);
        if (res.ok) {
            console.log("✅ Jarvis API is online and healthy");
        }
    } catch (e) {
        addMessage("System", "Warning: Cannot connect to Jarvis API. Ensure the server is running on port 8000.", "jarvis");
    }
});