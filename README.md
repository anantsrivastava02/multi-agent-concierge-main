# Multi-Agent Conversational AI System

## Overview
This project implements a sophisticated multi-agent conversational AI system with authentication, information retrieval, and conversation memory. The system uses a modular architecture with specialized agents that can handle different types of user requests.

## Features
- **User Authentication**: Secure login system with session management.
- **Vector Search**: Pinecone-based retrieval for finding relevant information.
- **Conversation Memory**: Remembers chat history across sessions.
- **Multi-Agent Architecture**: Different specialized agents handle different types of queries.
- **Orchestration**: Smart routing of user requests to the appropriate agent.
- **Tool Integration**: Agents can use tools to perform actions like database queries.
- **Conversation Summarization**: Automatic summarization of conversations for future reference.

## Getting Started

### Prerequisites
- Python 3.9+
- PostgreSQL database
- Redis server
- OpenAI API key
- Pinecone API key

### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-agent-ai.git
   cd multi-agent-ai
   ```
## export OPENAI_API_KEY="your-openai-api-key"
## export PINECONE_API_KEY="your-pinecone-api-key"
## export POSTGRES_CONNECTION_STRING="postgresql://username:password@localhost:5432/dbname"
## export REDIS_URL="redis://localhost:6379"


## System Architecture

### Components
- **Main Application (main.py):** Entry point that handles user interaction and authentication.
- **Workflow Engine (workflow.py):** Manages the flow between different agents.
- **Embedding Service (embed_summaries.py):** Processes and embeds conversation summaries.
- **Utility Functions (utils.py):** Common utilities used across the system.

## Agents
- **Authentication Agent:** Handles user login and session management.
- **Retriever Agent:** Searches for information using vector search.
- **Concierge Agent:** Orchestrates between different specialized agents.

## Data Flow
1. User sends a message.
2. System checks authentication status.
3. If authenticated, the message is routed to the appropriate agent.
4. Agent processes the message, potentially using tools.
5. Response is returned to the user.
6. Conversation is stored and periodically summarized.

## Database Schema
- **users:** Stores user information and session tokens.
- **conversations:** Stores raw conversation data.
- **conversation_summaries:** Stores summarized conversations with embeddings.

## Environment Variables
Set the following environment variables before running the application:
```sh
export OPENAI_API_KEY="your-openai-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
export POSTGRES_CONNECTION_STRING="postgresql://username:password@localhost:5432/dbname"
export REDIS_URL="redis://localhost:6379"
```

## Initializing the Database
Run the following command to initialize the database:
```sh
python init_db.py
```

## Extending the System
### Adding New Agents
Define a new agent configuration in `main.py` to extend the system.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- **LlamaIndex** for the workflow and agent framework.
- **OpenAI** for the language models.
- **Pinecone** for vector search capabilities.

