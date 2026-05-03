# Cloud Deploy

`serve()` wraps any `GantryEngine` in a production-ready FastAPI server
with async job queues, real-time SSE event streaming, and HITL resume.

## Minimal server

```bash
pip install 'gantrygraph[cloud]'
```

```python
# server.py
from gantrygraph import GantryEngine
from gantrygraph.cloud import serve
from gantrygraph.presets import qa_agent
from langchain_anthropic import ChatAnthropic

def make_agent() -> GantryEngine:
    return qa_agent(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        workspace="/workspace",
    )

serve(make_agent, host="0.0.0.0", port=8080)
```

```bash
python server.py
# → Serving on http://0.0.0.0:8080
```

## REST API

### Submit a task

```bash
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{"task": "List the 10 largest files in /workspace"}'
# → {"job_id": "abc123", "status": "queued"}
```

### Poll status

```bash
curl http://localhost:8080/status/abc123
# → {"job_id": "abc123", "status": "completed", "result": "..."}
```

### Stream events (SSE)

```bash
curl -N http://localhost:8080/stream/abc123
# data: {"type": "observe", "step": 0, "data": {...}}
# data: {"type": "think",   "step": 0, "data": {...}}
# data: {"type": "act",     "step": 0, "data": {"tools_executed": ["shell_run"]}}
# event: done
# data: {}
```

### Resume a suspended job

```bash
curl -X POST http://localhost:8080/resume/abc123 \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```

## Docker

The included `Dockerfile` installs **Xvfb** so GUI automation works headless:

```dockerfile
# In your project:
# cp $(python -c "import gantrygraph; import os; print(os.path.dirname(gantrygraph.__file__))")/cloud/templates/Dockerfile .
```

```bash
docker build -t my-agent .
docker run -p 8080:8080 -v $(pwd)/workspace:/workspace my-agent
```

## Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gantry-agent
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: agent
        image: my-agent:latest
        ports:
        - containerPort: 8080
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: anthropic-secret
              key: api-key
```

## Config-driven deployment

Use `GantryConfig` so all settings come from environment variables:

```python
from gantrygraph import GantryConfig
from gantrygraph.cloud import serve
from langchain_anthropic import ChatAnthropic

def make_agent():
    cfg = GantryConfig.from_env()   # reads GANTRY_* env vars
    return cfg.build(llm=ChatAnthropic(model="claude-sonnet-4-6"))

serve(make_agent)
```

```bash
export GANTRY_WORKSPACE=/workspace
export GANTRY_MAX_STEPS=30
export GANTRY_MAX_WALL_SECONDS=120
export GANTRY_MEMORY=in_memory
python server.py
```
