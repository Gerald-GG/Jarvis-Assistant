const API = "http://localhost:8000";

const chat = document.getElementById("chat");
const msgInput = document.getElementById("msg");
const sendBtn = document.getElementById("sendBtn");

// Helper: Typewriter Effect
function typeWriter(element, text, speed = 20) {
    let i = 0;
    element.innerHTML = ""; // Clear the "Thinking..." text
    
    return new Promise((resolve) => {
        function type() {
            if (i < text.length) {
                element.innerHTML += text.charAt(i);
                i++;
                chat.scrollTop = chat.scrollHeight; // Keep scrolling as text grows
                setTimeout(type, speed);
            } else {
                resolve();
            }
        }
        type();
    });
}

// Add message to chat
function addMessage(sender, text, type) {
    const div = document.createElement("div");
    div.className = `message ${type}-message`;
    div.innerHTML = `
        <div class="message-sender"><strong>${sender}</strong></div>
        <div class="message-content">${text}</div>
    `;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div.querySelector(".message-content"); // Return the content div for typewriter
}

// Send message
async function sendMessage() {
    const text = msgInput.value.trim();
    if (!text) return;
    
    // Add User message and clear input
    addMessage("You", text, "user");
    msgInput.value = "";

    // Add Jarvis bubble with "..." initial state
    const jarvisContentArea = addMessage("Jarvis", "Generating transmission...", "jarvis");

    try {
        const res = await fetch(`${API}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                text: text,
                context: {}, 
                stream: false
            })
        });

        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || "Server error");

        // Execute typewriter effect on the returned text
        await typeWriter(jarvisContentArea, data.response);

    } catch (err) {
        jarvisContentArea.innerHTML = `⚠️ Error: ${err.message}`;
        console.error("Communication Error:", err);
    }
}

// Event Listeners
sendBtn.addEventListener("click", sendMessage);
msgInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        sendMessage();
    }
});