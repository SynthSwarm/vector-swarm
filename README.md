# Vector Swarm 🐝

Multi-agent AI system where LLM-powered agents coordinate through vector space to solve complex tasks. Inspired by [SynthSwarm](hhttps://synthswarm.com).

## Concept

- **The Queen**: Your mission embedded as a fixed point in 768D space
- **The Drones**: LLM agents that generate actions, embed them, and navigate toward the Queen
- **Swarm Physics**: Agents coordinate using flocking dynamics (cohesion, alignment, separation)
- **Visualization**: Real-time 3D view using PCA dimensionality reduction

## Requirements

- NVIDIA GPU with 8GB+ VRAM
- Python 3.10+
- Docker & Docker Compose

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/vector-swarm.git
cd vector-swarm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start vLLM
docker compose up -d

# 3. Run swarm
python app.py
```

Open [http://localhost:5000](http://localhost:5000) and click **INITIALIZE SWARM**.

## Architecture

```
Browser (3D Viz) → Flask API → Swarm Manager → Agent Processes
                                      ↓
                              Shared Memory ← vLLM + FastEmbed
```

**Core Components:**
- `app.py` - Flask API (port 5000)
- `swarm_manager.py` - Orchestration & PCA
- `swarm_agent.py` - Agent logic & decision-making
- `swarm_physics.py` - Flocking dynamics
- `agent_memory.py` - Lock-free shared memory
- `embedding_service.py` - FastEmbed (nomic-embed-text-v1.5)

## How It Works

1. Mission text → 768D embedding (the Queen)
2. Each agent:
   - Generates 3-5 action candidates via LLM
   - Embeds each action to 768D vectors
   - Calculates alignment with Queen + swarm
   - Selects highest-scoring action
   - Updates position in shared memory
3. Visualization: PCA reduces 768D → 3D for plotting

## Configuration

**Models** (`swarm_manager.py`):
```python
VLLM_API = "http://localhost:8000/v1"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
MAX_AGENTS = 20
```

**Flocking weights** (`swarm_physics.py:46`):
```python
w_c=1.0  # Cohesion (attraction to Queen)
w_a=0.5  # Alignment (match swarm direction)
w_s=0.8  # Separation (avoid crowding)
```

## Performance Tips

**GPU Contention (Windows):**
- Settings → Display → Graphics settings
- Add your browser → Set to "Power saving" (uses Intel GPU)
- Lets vLLM use NVIDIA while browser uses Intel

**Tuning:**
- Start with 3-5 agents
- Adjust poll interval: `index.html:545` (default 500ms)
- Reduce vLLM GPU usage: `--gpu-memory-utilization 0.8`

## Troubleshooting

| Issue                              | Fix                                                       |
| ---------------------------------- | --------------------------------------------------------- |
| "Waiting for vector entropy" stuck | Counter not incrementing - check `swarm_physics.py:32-34` |
| Slow visualization                 | GPU contention - force browser to Intel GPU               |
| vLLM fails to start                | Check GPU memory: `docker compose logs vllm`              |

## Acknowledgments

- [SynthSwarm](https://synthswarm.com) - Original inspiration
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference
- [FastEmbed](https://github.com/qdrant/fastembed) - Fast embeddings
- [Plotly](https://plotly.com/) - 3D visualization

## License

MIT
