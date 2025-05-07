# 🧠 Multi-Agent AI Chatbot

This is a **multi-agent AI assistant** built using [LangGraph](https://docs.langgraph.dev/), [LangChain](https://www.langchain.com/), and [Streamlit](https://streamlit.io/). It uses **dynamic tool invocation** to fetch information from **Arxiv**, **Wikipedia**, and **Tavily Search** — all orchestrated behind the scenes by an intelligent, graph-based reasoning engine.

> ✅ Ask *any* question. The assistant automatically decides if it should search the web, fetch scientific papers, or reference Wikipedia — all without the user needing to specify a tool.

---

## ✨ What Can It Do?

This chatbot is not just another LLM wrapper — it acts like an autonomous **multi-agent system**, where tools (agents) are triggered dynamically based on the user's query.

### 🔍 Auto-Triggers Web Search and Knowledge Tools

- Automatically **routes questions** to the most relevant tool:
  - **Tavily Search** for real-time web data
  - **Arxiv API** for research-based or scientific questions
  - **Wikipedia API** for general knowledge
- Uses **LangGraph’s conditional logic** to analyze the query and select tools at runtime
- All tools run in the background — no tool selection is needed from the user

### 🧠 Multi-Agent Behavior (Behind the Scenes)

- Built with **LangGraph**, which defines a *state machine* or *graph of logic* for how the assistant should operate
- Each node (LLM, Tool, Response Generator) is a functional "agent"
- Nodes are **connected intelligently**:
  - First, the system interprets the question
  - Then, it decides *which tools to run*
  - Finally, it collects tool outputs and generates a final, human-readable answer

### 💬 Clean Chat UI via Streamlit

- Interactive chat interface to ask anything naturally
- Maintains conversation history with context-aware responses
- Fully compatible with real-time use cases (classroom help, research assistant, etc.)
