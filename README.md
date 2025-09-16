# 🕵️ DevAgent - Demystify your codebase
**Built for TiDB AgentX Hackathon 2025: Forge Agentic AI for Real-World Impact**

DevAgent transforms any GitHub repository into an interactive AI assistant. Instead of navigating thousands of files, simply provide a repository URL and start asking questions about code architecture, security vulnerabilities, deployment strategies, and implementation details.

DevAgent combines TiDB's distributed vector database for lightning-fast semantic code search with Kimi LLM's web search capabilities to provide both deep code analysis and up-to-date deployment insights. Common questions are cached for sub-100ms responses, while complex queries leverage real-time web search.

# 🚀 Quick Setup

## Prerequisites
- **TiDB Account** – free tier available at [tidbcloud.com](https://tidbcloud.com)  
- **Kimi API Key** – get yours at [moonshot.cn](https://moonshot.cn)  
- **Modal Account** – serverless compute at [modal.com](https://modal.com)  

## Installation & Deployment
1. Clone the repository and install dependencies:
    ```bash
    git clone https://github.com/yourusername/devagent.git
    cd devagent
    pip install -r requirements.txt
    ```

2. Configure environment variables:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` with your credentials:
    ```env
    DATABASE_URL=your_tidb_connection_string
    KIMI_API_KEY=your_kimi_api_key
    ```

3. Deploy backend and run frontend:
    - Setup Modal secrets:
        ```bash
        modal secret create tidb-secret DATABASE_URL=your_tidb_url
        modal secret create kimi-secret KIMI_API_KEY=your_kimi_key
        ```
    - Deploy serverless backend:
        ```bash
        modal deploy agent_core.py
        ```
    - Run Streamlit frontend:
        ```bash
        streamlit run app.py
        ```

The system automatically handles repository ingestion, vector embedding, and intelligent caching. Visit the Streamlit URL to start analyzing repositories with AI.

# 🎯 Key Features
- **🧠 Smart Code Analysis** – Deep understanding of architecture and patterns  
- **🌐 Real-Time Web Search** – Current deployment guides and best practices  
- **⚡ Ultra-Fast Responses** – Sub-100ms cached answers for common questions  
- **🛡️ Security Auditing** – Automated vulnerability detection with suggested fixes  
- **🔧 Function Explorer** – Detailed explanations of specific code components  

# 🔮 Future Goals
- Private repository support with GitHub authentication  
- VS Code extension for seamless IDE workflow  
- Team collaboration features with shared insights and annotations  
- Multi-language expansion beyond current supported programming languages  

# 📂 Project Structure
```devagent/
├── .env.example # Environment variables template
├── agent_core.py # Modal serverless backend with AI agents
├── app.py # Streamlit frontend application
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

# 📖 License
MIT License © 2025 Pratik Shah