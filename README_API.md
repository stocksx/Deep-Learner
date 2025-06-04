# Pattern-Based Learning & Chunking API

A FastAPI server providing REST endpoints for training entity patterns and performing intelligent text chunking using configurable pattern-based learning.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python3 api_server.py
```

Or with custom settings:

```bash
python3 api_server.py --host 0.0.0.0 --port 8080 --reload
```

### 3. Access API Documentation

- **Interactive Docs**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **Health Check**: http://127.0.0.1:8000/health

## 📊 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check |
| `GET` | `/info` | System information |

### Training & Chunking

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/train` | Train entity patterns from text |
| `POST` | `/chunk` | Chunk text using learned patterns |
| `POST` | `/chunk/semantic` | Semantic chunking using transformers |

### Configuration Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/configure` | Configure new entity type |
| `GET` | `/entity-types` | List available entity types |
| `GET` | `/entity-types/{type}/info` | Get entity type details |
| `DELETE` | `/entity-types/{type}` | Delete entity type |
| `POST` | `/clear-cache` | Clear all cached data |

## 🔧 Usage Examples

### Training Patterns

Train patterns for broker names:

```bash
curl -X POST "http://127.0.0.1:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "broker",
    "text": "The client opened an account with ZERODHA for trading.",
    "entities": {"ZERODHA": "ZERODHA"}
  }'
```

Response:
```json
{
  "success": true,
  "entity_type": "broker",
  "examples_learned": 1,
  "patterns_learned": {
    "word_patterns": 1,
    "position_patterns": 1,
    "context_patterns": 2
  },
  "message": "Successfully trained broker patterns from provided text"
}
```

### Text Chunking

Chunk text using learned patterns:

```bash
curl -X POST "http://127.0.0.1:8000/chunk" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "SBI SECURITIES managed the portfolio. ZERODHA provided trading services.",
    "entity_types": ["broker"],
    "max_chunk_size": 100
  }'
```

Response:
```json
{
  "success": true,
  "total_chunks": 2,
  "chunks": [
    {
      "content": "SBI SECURITIES managed the portfolio.",
      "total_patterns": 1,
      "pattern_summary": {
        "broker": {
          "count": 1,
          "avg_confidence": 0.85,
          "entities": ["SBI SECURITIES"]
        }
      }
    }
  ],
  "statistics": {
    "total_entities_found": 2,
    "avg_entities_per_chunk": 1.0
  }
}
```

### Configure Entity Type

Add a new entity type:

```bash
curl -X POST "http://127.0.0.1:8000/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "drug",
    "examples": [
      {"original_text": "Aspirin", "normalized_form": "ASPIRIN"},
      {"original_text": "Paracetamol", "normalized_form": "PARACETAMOL"}
    ],
    "heuristics": {
      "indicators": ["TABLET", "CAPSULE", "MG"],
      "suffixes": ["MG", "ML"],
      "known_entities": ["ASPIRIN", "PARACETAMOL"],
      "weights": {
        "indicators": 0.3,
        "suffixes": 0.2,
        "known_entities": 0.5
      }
    }
  }'
```

## 🐍 Python Client

Use the provided Python client for easy integration:

```python
from api_client_example import PatternLearningClient

client = PatternLearningClient("http://127.0.0.1:8000")

# Train patterns
result = client.train_patterns(
    entity_type="broker",
    text="ZERODHA offers discount brokerage services.",
    entities={"ZERODHA": "ZERODHA"}
)

# Chunk text
chunks = client.chunk_text(
    text="The portfolio was managed by SBI SECURITIES.",
    entity_types=["broker"],
    max_chunk_size=200
)
```

## 📋 Request/Response Models

### TrainingRequest

```json
{
  "entity_type": "string",
  "text": "string",
  "entities": {
    "original_text": "normalized_form"
  },
  "config_dir": "configs"  // optional
}
```

### ChunkingRequest

```json
{
  "text": "string",
  "entity_types": ["string"],
  "max_chunk_size": 512,     // optional
  "overlap_size": 50,        // optional
  "config_dir": "configs"    // optional
}
```

### ConfigurationRequest

```json
{
  "entity_type": "string",
  "examples": [              // optional
    {
      "original_text": "string",
      "normalized_form": "string"
    }
  ],
  "heuristics": {            // optional
    "indicators": ["string"],
    "suffixes": ["string"],
    "known_entities": ["string"],
    "weights": {
      "indicators": 0.3,
      "suffixes": 0.2,
      "known_entities": 0.5
    }
  }
}
```

## 🗂️ Configuration Structure

The API uses the same configuration structure as the core system:

```
configs/
├── broker_examples.txt       # Entity examples
├── broker_heuristics.json    # Detection rules
├── broker_patterns.json      # Learned patterns (auto-generated)
├── company_examples.txt
├── company_heuristics.json
└── company_patterns.json
```

## ⚡ Features

### Intelligent Caching
- Learners and chunkers are cached for performance
- Automatic cache invalidation when configurations change
- Memory-efficient pattern storage

### Pattern Persistence
- Learned patterns automatically saved to configuration files
- Patterns loaded on server startup
- Hot-reloading of configurations

### Multi-Entity Support
- Train multiple entity types simultaneously
- Combine entity types in chunking requests
- Cross-entity pattern analysis

### Quality Metrics
- Chunking boundary quality scores
- Entity detection confidence levels
- Comprehensive statistics reporting

## 🔍 Monitoring & Debugging

### Health Checks

```bash
curl http://127.0.0.1:8000/health
```

### System Information

```bash
curl http://127.0.0.1:8000/info
```

Returns:
- Available entity types
- Configuration file status
- Loaded learners count
- System operational status

### Entity Type Details

```bash
curl http://127.0.0.1:8000/entity-types/broker/info
```

Returns detailed information about a specific entity type including:
- Number of examples
- Pattern counts
- Heuristics configuration
- Sample examples

## 🚨 Error Handling

The API provides detailed error responses:

```json
{
  "detail": "Training failed: Entity type 'unknown' not configured"
}
```

Common error codes:
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (entity type doesn't exist)
- `500`: Internal Server Error (system failure)
- `503`: Service Unavailable (missing dependencies)

## 🧪 Testing

Run the comprehensive test suite:

```bash
python3 api_client_example.py
```

This tests all major endpoints and demonstrates the complete workflow.

## 🔧 Development

### Auto-Reload Mode

For development, start the server with auto-reload:

```bash
python3 api_server.py --reload
```

### Custom Configuration Directory

Use a custom configuration directory:

```bash
python3 api_server.py --config-dir /path/to/configs
```

### Environment Variables

Set configuration via environment:

```bash
export CONFIG_DIR=/path/to/configs
python3 api_server.py
```

## 📦 Dependencies

Core dependencies:
- `fastapi>=0.68.0` - Web framework
- `uvicorn>=0.15.0` - ASGI server
- `pydantic>=1.8.0` - Data validation

Optional dependencies:
- `sentence-transformers>=2.0.0` - For semantic chunking
- `nltk>=3.6.0` - For sentence tokenization

## 🤝 Integration

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python3", "api_server.py", "--host", "0.0.0.0"]
```

### Production Deployment

For production, use a production ASGI server:

```bash
pip install gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📈 Performance

### Benchmarks

Typical performance on modern hardware:
- Training: ~100ms per paragraph
- Chunking: ~50ms per 1KB text
- Pattern matching: ~10ms per sentence

### Optimization Tips

1. **Batch Training**: Train multiple examples together
2. **Cache Warming**: Pre-load frequently used entity types
3. **Chunk Size Tuning**: Optimize chunk size for your use case
4. **Memory Management**: Clear cache periodically for long-running instances

## 🛠️ Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check port availability
lsof -i :8000

# Install missing dependencies
pip install -r requirements.txt
```

**Semantic chunking fails:**
```bash
# Install optional dependencies
pip install sentence-transformers nltk
```

**Configuration errors:**
```bash
# Check configuration directory permissions
ls -la configs/

# Validate JSON files
python3 -m json.tool configs/broker_heuristics.json
```

## 📝 License

This project is part of the Deep-Learner pattern-based learning system.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

**Happy Pattern Learning! 🎯** 