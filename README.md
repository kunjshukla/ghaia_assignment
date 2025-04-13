
# 🤖 Multi-Agent AI Help Desk System

This project is a prototype AI Help Desk system designed to simulate how multiple autonomous agents can collaborate to resolve IT support tickets. The system is built using [Microsoft AutoGen](https://github.com/microsoft/autogen), [LangChain](https://www.langchain.com), and Google's [Gemini Pro API](https://ai.google.dev/), or optionally OpenRouter/Groq.

---

## 🧠 Architecture Overview

```
User → Master Agent → User Intake Agent → Resolution Agent → (Optional: Escalation Agent) → Response
                      ↑                         ↓                     ↓
                 System Memory         Knowledge Base         Ticket Generator
```

Each agent performs a specific function:
- **Master Agent**: Routes and coordinates all tasks.
- **User Intake Agent**: Extracts keywords and classifies the request.
- **Resolution Agent**: Attempts automated resolution via a knowledge base.
- **Escalation Agent**: Handles unresolved issues by creating support tickets.

---

## 🚀 Features

- Multi-agent orchestration using AutoGen
- Built-in memory to track and reuse past resolutions
- Keyword-based request classification
- Escalation with ticket summary generation
- Gemini/OpenRouter LLM support
- Streamlit-based front-end (optional)

---

## 💻 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-help-desk.git
cd ai-help-desk
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` File

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 4. Run the System

```bash
streamlit run help_desk_dashboard.py
```

---

## 🧪 Example Use Case

**Input:** "My VPN is not working, keeps failing with a timeout error"  
**Output:**  
> "Try restarting your VPN client. Check your internet connection and credentials..."

---

## 📁 File Structure

```
├── help_desk.py                # Core logic and agents
├── help_desk_dashboard.py      # Streamlit app
├── requirements.txt
├── .env.example
└── architecture.png            # Visual flow diagram (add your draw.io export here)
```

---

## 📌 Future Improvements

- Add persistence using TinyDB or SQLite
- Integrate with real IT support tools (Jira, Slack, etc.)
- Advanced NLP for smarter keyword detection
- Role-based access and user authentication

---

## 📄 License

MIT License © 2025 YourName
