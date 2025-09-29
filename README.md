# O!Store Mobile Phone Agent

A sophisticated ReAct agent for mobile phone retail assistance in Kyrgyzstan, powered by OpenAI and Qdrant vector database. The agent provides intelligent recommendations and answers questions about mobile phones, store services, and policies.

## 🌟 Features

- **Intelligent Mobile Phone Recommendations**: Search and compare 931+ mobile phones from Apple, Samsung, Xiaomi, and more
- **Store Information**: Get details about delivery, warranty, installment plans, and store locations
- **Ollama-Compatible API**: Works with OpenWebUI and Ollama CLI
- **Multilingual Support**: Responds in Russian and Kyrgyz
- **Beautiful Markdown Formatting**: Professional responses with proper formatting
- **Vector Search**: Uses Qdrant for semantic search across phone specifications and store documents

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   OpenWebUI     │    │     API      │    │   O!Store   │
│   (Frontend)    │◄───┤   Wrapper    │◄───┤   Agent     │
│                 │    │  (Ollama)    │    │  (ReAct)    │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │                    │
                              │                    ▼
                              │            ┌─────────────┐
                              │            │   Qdrant    │
                              │            │  Database   │
                              │            └─────────────┘
                              ▼
                       ┌─────────────┐
                       │   OpenAI    │
                       │    API      │
                       └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- OpenAI API key

### 1. Clone and Setup

```bash
git clone <your-repo>
cd ostore

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Start Services

```bash
# Start Qdrant database
docker-compose up -d qdrant

# Setup vector store (one-time)
source .venv/bin/activate
python scripts/setup_vectorstore.py
python scripts/upload_mobile_specs.py

# Start the API
python -m uvicorn api.ollama_wrapper:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Use with OpenWebUI (Optional)

```bash
# Start OpenWebUI
docker-compose up -d openwebui

# Access at http://localhost:3000
# Select model: ostore-agent
```

## 📊 Data

The agent includes:

- **931 Mobile Phone Specifications**: Complete database of phones with prices, specs, and features
- **Store Documentation**: FAQ, delivery info, warranty terms, installment plans, store locations
- **Qdrant Collections**:
  - `mobiles_specs`: Phone specifications and pricing
  - `mobile_docs`: Store policies and information

## 🎯 Usage Examples

### CLI Interface
```bash
source .venv/bin/activate
python main.py
```

### API Endpoints
```bash
# Chat with the agent
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ostore-agent",
    "messages": [{"role": "user", "content": "покажи iPhone 16 Pro Max"}],
    "stream": false
  }'

# List available models
curl http://localhost:8000/api/tags
curl http://localhost:8000/v1/models
```

### Ollama CLI
```bash
export OLLAMA_HOST=http://localhost:8000
ollama run ostore-agent "покажи Samsung Galaxy S24"
```

## 🔧 Development

### Project Structure
```
ostore/
├── api/                    # Ollama-compatible API wrapper
│   ├── ollama_wrapper.py   # Main API endpoints
│   ├── models.py          # Pydantic models
│   └── streaming.py       # Streaming response helpers
├── app/                   # Core application
│   ├── agent/            # ReAct agent implementation
│   ├── core/             # Configuration and vector store
│   └── utils/            # Logging and utilities
├── data/                 # Sample data
│   ├── docs_dataset/     # Store documents (.docx)
│   └── mobiles_dataset/  # Phone specifications (.csv)
├── scripts/              # Setup and utility scripts
├── docker-compose.yml    # Docker services
└── requirements.txt      # Python dependencies
```

### Adding New Data

**Mobile Phones:**
```bash
# Add to data/mobiles_dataset/mobiles.csv
python scripts/upload_mobile_specs.py
```

**Store Documents:**
```bash
# Add .docx files to data/docs_dataset/
python scripts/setup_vectorstore.py
```

## 🌐 OpenWebUI Integration

The agent is fully compatible with OpenWebUI:

1. **Models**: Appears as "ostore-agent" and "ostore-agent:latest"
2. **Streaming**: Real-time response streaming
3. **Markdown**: Beautiful formatted responses
4. **Conversations**: Maintains context across messages

## 📝 API Reference

### Ollama-Compatible Endpoints

- `GET /api/tags` - List available models
- `GET /api/ps` - Show running models
- `POST /api/chat` - Chat with agent (streaming/non-streaming)
- `POST /api/generate` - Generate responses
- `POST /api/show` - Show model information

### OpenAI-Compatible Endpoints

- `GET /v1/models` - List models (OpenWebUI compatibility)

### Health Check

- `GET /health` - Service health and statistics

## 🔍 Monitoring

The agent includes built-in analytics:

- Query logging and phone mention tracking
- Session statistics
- Health monitoring
- Performance metrics

Access analytics at `/health` endpoint.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📋 TODO

- [ ] Add more mobile brands and models
- [ ] Implement user preference learning
- [ ] Add price comparison features
- [ ] Support for English language
- [ ] Integration with real inventory systems

## 📄 License

This project is for educational purposes. Ensure you have proper licenses for all data and API usage.

## 🆘 Troubleshooting

### Common Issues

**Docker not starting:**
```bash
docker-compose down
docker-compose up -d
```

**Qdrant connection issues:**
```bash
curl http://localhost:6333/health
```

**OpenAI API errors:**
```bash
# Check your API key in .env
echo $OPENAI_API_KEY
```

### Logs
```bash
# API logs
docker-compose logs -f ostore-api

# OpenWebUI logs
docker-compose logs -f openwebui

# Qdrant logs
docker-compose logs -f qdrant
```

---

**Built with ❤️ for O!Store Kyrgyzstan**