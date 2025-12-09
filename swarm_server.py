import time
import numpy as np
import atexit
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from multiprocessing import Process
from sklearn.decomposition import PCA
from openai import OpenAI

# Import our custom modules
from swarm_physics import SwarmPhysics
from swarm_agent import VectorAgent  # We will add a wrapper function here later

# --- Logging Helper ---
def log(message, level="INFO"):
    """Timestamped logging helper"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}", flush=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for your JS frontend

# --- Configuration ---
OLLAMA_API = "http://localhost:11434/v1"
EMBED_MODEL = "nomic-embed-text"

# --- Global System State ---
# We keep these global so Flask routes can access them
physics_engine = None
agent_processes = []
pca_model = PCA(n_components=3)
client = OpenAI(base_url=OLLAMA_API, api_key="ollama")


# --- Helper: Embedding Wrapper ---
def get_embedding(text):
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(model=EMBED_MODEL, input=[text])
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Embedding Error: {e}")
        return np.zeros(768, dtype=np.float32)


# --- Configuration Constants ---
MAX_AGENTS = 20
VECTOR_DIM = 768

# --- The Agent Wrapper (Must be top-level for pickling) ---
def run_agent_process(agent_id):
    """
    The entry point for a child process.
    Re-instantiates the Physics engine (create=False) to attach to shared memory.
    """
    log(f"Agent {agent_id} starting up", "AGENT")

    # 1. Attach to Physics Engine
    # Note: We create a fresh instance which attaches to the existing shared memory block
    # IMPORTANT: Must use same parameters as the parent process!
    physics = SwarmPhysics(max_agents=MAX_AGENTS, dim=VECTOR_DIM, create=False)

    # 2. Setup Local LLM Client (Each process needs its own connection)
    local_client = OpenAI(base_url=OLLAMA_API, api_key="ollama")

    # 3. Create Agent Instance
    agent = VectorAgent(
        agent_id=agent_id,
        physics_engine=physics,
        llm_client=local_client,
        embed_func=get_embedding,  # Pass our helper function
    )

    # 4. The Infinite Loop
    try:
        log(f"Agent {agent_id} entering main loop", "AGENT")
        while True:
            agent.step()
            # Jitter the sleep to prevent lock-step behavior
            time.sleep(np.random.uniform(1.0, 2.0))
    except KeyboardInterrupt:
        log(f"Agent {agent_id} shutting down", "AGENT")


# --- Server Initialization ---
def init_system():
    global physics_engine
    # Initialize the Physics Engine (Allocates Shared RAM)
    physics_engine = SwarmPhysics(max_agents=MAX_AGENTS, dim=VECTOR_DIM, create=True)
    log("🐝 Swarm Physics Engine initialized", "SYSTEM")


# --- API Endpoints ---


@app.route("/api/swarm/start", methods=["POST"])
def start_swarm():
    """
    Payload: { "mission": "Fix the DB deadlock", "agent_count": 3 }
    """
    global agent_processes
    data = request.json
    mission = data.get("mission", "Idling")
    count = int(data.get("agent_count", 3))

    log(f"⚡ START command received: '{mission}' with {count} agents", "API")

    # 1. Stop existing agents if any
    stop_swarm()

    # 2. Inject the Queen Vector (The Mission)
    assert physics_engine is not None, "Physics engine not initialized"
    log("Embedding queen mission signal...", "API")
    queen_vec = get_embedding(mission)
    physics_engine.set_queen_signal(queen_vec)
    log("✓ Queen signal injected into shared memory", "API")

    # 3. Spawn Agents
    log(f"Spawning {count} agent processes...", "API")
    for i in range(count):
        p = Process(target=run_agent_process, args=(i,))
        p.start()
        agent_processes.append(p)
    log(f"✓ All {count} agents spawned", "API")

    return jsonify({"status": "started", "mission": mission, "agents": count})


@app.route("/api/swarm/stop", methods=["POST"])
def stop_swarm_route():
    stop_swarm()
    return jsonify({"status": "stopped"})


def stop_swarm():
    global agent_processes
    if not agent_processes:
        log("No active agents to stop", "API")
        return

    log(f"🛑 STOP command: terminating {len(agent_processes)} agents...", "API")
    for i, p in enumerate(agent_processes):
        if p.is_alive():
            p.terminate()
            p.join()
            log(f"✓ Agent {i} terminated", "API")
    agent_processes = []
    log("All agents stopped", "API")


@app.route("/api/swarm/state", methods=["GET"])
def get_state():
    """
    Returns 3D coordinates for visualization.
    """
    assert physics_engine is not None, "Physics engine not initialized"

    # 1. Read Raw Vectors from Shared Memory
    count = int(physics_engine.space.counter_view[0])  # Convert numpy int64 to Python int

    # Need at least 3 points for 3D PCA to make sense
    # (Queen + 2 Agents, or just Queen + dummies if empty)
    if count < 3:
        return jsonify({"status": "waiting_for_entropy", "data": []})

    raw_vectors = physics_engine.space.vector_view[:count]

    # 2. Dimensionality Reduction (Dynamic PCA)
    # We wrap in try/except because PCA can fail if vectors are identical (zero variance)
    try:
        reduced = pca_model.fit_transform(raw_vectors)
    except Exception as e:
        # Fallback if system is perfectly static/aligned
        return jsonify({"status": "calculating", "error": str(e)})

    snapshot = []

    # 3. Format Data
    # Index 0 is QUEEN
    queen_x, queen_y, queen_z = float(reduced[0][0]), float(reduced[0][1]), float(reduced[0][2])

    # Log Queen position for debugging
    log(f"Queen position: ({queen_x:.3f}, {queen_y:.3f}, {queen_z:.3f})", "VIZ")

    snapshot.append(
        {
            "id": "QUEEN",
            "type": "queen",
            "x": queen_x,
            "y": queen_y,
            "z": queen_z,
            "label": "MISSION COMMAND",
        }
    )

    # Indices 1..N are AGENTS
    for i in range(1, count):
        snapshot.append(
            {
                "id": f"Agent-{i-1}",
                "type": "drone",
                "x": float(reduced[i][0]),
                "y": float(reduced[i][1]),
                "z": float(reduced[i][2]),
                "label": f"Drone-{i-1}",
            }
        )

    return jsonify({"status": "active", "count": count, "data": snapshot})


# --- Cleanup Handlers ---
def cleanup():
    log("Server shutdown initiated", "SYSTEM")
    stop_swarm()
    if physics_engine:
        physics_engine.destroy()
        log("Shared memory destroyed", "SYSTEM")


atexit.register(cleanup)

if __name__ == "__main__":
    # Initialize Shared Memory
    init_system()
    # Run Flask
    app.run(host="0.0.0.0", port=5000, debug=False)
