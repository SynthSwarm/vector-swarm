import time
import numpy as np
import atexit
import threading
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from multiprocessing import Process, Queue, Lock
from sklearn.decomposition import PCA
from openai import OpenAI

# Import our custom modules
from swarm_physics import SwarmPhysics
from swarm_agent import VectorAgent
from logged_llm_client import LoggedLLMClient


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
pca_fitted = False  # Track if PCA has been fitted
expected_agent_count = 0  # How many agents we're waiting for
client = OpenAI(base_url=OLLAMA_API, api_key="ollama")

# --- Chat Logging Infrastructure ---
chat_log_queue = None  # Multiprocessing Queue for agent logs
chat_logs = []  # In-memory buffer (last 50 entries)
MAX_CHAT_LOGS = 50
log_drain_thread = None  # Background thread to drain queue

# --- Embedding Lock ---
embedding_lock = None  # Multiprocessing Lock to serialize embedding calls


# --- Helper: Embedding Wrapper (Thread-Safe) ---
def get_embedding(text, lock=None):
    """
    Get embedding from Ollama API with optional locking for multiprocessing safety.

    Args:
        text: Text to embed
        lock: Optional multiprocessing.Lock to serialize calls
    """
    text = text.replace("\n", " ")

    # If lock provided, use it to serialize access
    if lock:
        with lock:
            try:
                response = client.embeddings.create(model=EMBED_MODEL, input=[text])
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                log(f"Embedding Error: {e}", "EMBED")
                return np.zeros(768, dtype=np.float32)
    else:
        # No lock (main process usage)
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=[text])
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            log(f"Embedding Error: {e}", "EMBED")
            return np.zeros(768, dtype=np.float32)


# --- Configuration Constants ---
MAX_AGENTS = 20
VECTOR_DIM = 768


# --- Background Thread: Drain Log Queue ---
def drain_log_queue():
    """
    Background thread that continuously drains the chat_log_queue
    and appends entries to the in-memory chat_logs buffer.
    Keeps only the last MAX_CHAT_LOGS entries.
    """
    global chat_logs, chat_log_queue
    while True:
        try:
            # Safety check: ensure queue is initialized
            if chat_log_queue is None:
                time.sleep(0.1)
                continue

            # Block until we get a log entry (or timeout after 1 second)
            entry = chat_log_queue.get(timeout=1.0)
            chat_logs.append(entry)

            # Trim to last N entries
            if len(chat_logs) > MAX_CHAT_LOGS:
                chat_logs = chat_logs[-MAX_CHAT_LOGS:]

        except Exception:
            # Queue.Empty exception or other errors - just continue
            time.sleep(0.1)


# --- The Agent Wrapper (Must be top-level for pickling) ---
def run_agent_process(agent_id, mission, log_queue, embed_lock):
    """
    The entry point for a child process.
    Re-instantiates the Physics engine (create=False) to attach to shared memory.

    Args:
        agent_id: ID of this agent
        log_queue: multiprocessing.Queue for chat logs
        embed_lock: multiprocessing.Lock for embedding calls
    """
    log(f"Agent {agent_id} starting up", "AGENT")

    # 1. Attach to Physics Engine
    # Note: We create a fresh instance which attaches to the existing shared memory block
    # IMPORTANT: Must use same parameters as the parent process!
    physics = SwarmPhysics(max_agents=MAX_AGENTS, dim=VECTOR_DIM, create=False)

    # 2. Setup Local LLM Client with Logging Wrapper
    base_client = OpenAI(base_url=OLLAMA_API, api_key="ollama")
    logged_client = LoggedLLMClient(base_client, agent_id, log_queue)

    # 3. Create locked embedding function
    def locked_embed(text):
        return get_embedding(text, lock=embed_lock)

    # 4. Create Agent Instance
    agent = VectorAgent(
        agent_id=agent_id,
        physics_engine=physics,
        llm_client=logged_client,
        embed_func=locked_embed,  # Pass locked embedding function
        starting_task=mission,
        log_queue=log_queue,  # Pass queue for selection logging
    )

    # 5. The Infinite Loop
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
    global physics_engine, chat_log_queue, log_drain_thread, embedding_lock
    # Initialize the Physics Engine (Allocates Shared RAM)
    physics_engine = SwarmPhysics(max_agents=MAX_AGENTS, dim=VECTOR_DIM, create=True)
    log("🐝 Swarm Physics Engine initialized", "SYSTEM")

    # Initialize Chat Log Queue
    chat_log_queue = Queue(maxsize=200)  # Buffer up to 200 messages
    log("📝 Chat log queue initialized", "SYSTEM")

    # Initialize Embedding Lock (to prevent deadlocks)
    embedding_lock = Lock()
    log("🔒 Embedding lock initialized", "SYSTEM")

    # Start background thread to drain log queue
    log_drain_thread = threading.Thread(target=drain_log_queue, daemon=True)
    log_drain_thread.start()
    log("🔄 Log drain thread started", "SYSTEM")


# --- API Endpoints ---


@app.route("/api/swarm/start", methods=["POST"])
def start_swarm():
    """
    Payload: { "mission": "Fix the DB deadlock", "agent_count": 3 }
    """
    global agent_processes, expected_agent_count, embedding_lock
    data = request.json
    mission = data.get("mission", "Idling")
    count = int(data.get("agent_count", 3))

    # Store expected count for PCA fitting logic
    expected_agent_count = count

    log(f"⚡ START command received: '{mission}' with {count} agents", "API")

    # 1. Stop existing agents if any
    stop_swarm()

    # 2. Inject the Queen Vector (The Mission)
    assert physics_engine is not None, "Physics engine not initialized"
    log("Embedding queen mission signal...", "API")
    queen_vec = get_embedding(mission, embedding_lock)
    physics_engine.set_queen_signal(queen_vec)
    log("✓ Queen signal injected into shared memory", "API")

    # 3. Spawn Agents
    log(f"Spawning {count} agent processes...", "API")
    for i in range(count):
        p = Process(
            target=run_agent_process, args=(i, mission, chat_log_queue, embedding_lock)
        )
        p.start()
        agent_processes.append(p)
    log(f"✓ All {count} agents spawned", "API")

    return jsonify({"status": "started", "mission": mission, "agents": count})


@app.route("/api/swarm/stop", methods=["POST"])
def stop_swarm_route():
    stop_swarm()
    return jsonify({"status": "stopped"})


def stop_swarm():
    global agent_processes, pca_fitted, expected_agent_count, chat_logs
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

    # Reset PCA and agent count for next run
    pca_fitted = False
    expected_agent_count = 0

    # Clear chat logs for next run
    chat_logs = []

    log("All agents stopped, PCA and chat logs reset for next mission", "API")


@app.route("/api/swarm/state", methods=["GET"])
def get_state():
    """
    Returns 3D coordinates for visualization.
    """
    global pca_fitted
    assert physics_engine is not None, "Physics engine not initialized"

    # 1. Read Raw Vectors from Shared Memory
    count = int(
        physics_engine.space.counter_view[0]
    )  # Convert numpy int64 to Python int

    # Need at least 3 points for 3D PCA to make sense
    # (Queen + 2 Agents, or just Queen + dummies if empty)
    if count < 3:
        return jsonify({"status": "waiting_for_entropy", "data": []})

    # Check if all agents have reported in (count = Queen + all agents)
    expected_total = expected_agent_count + 1  # +1 for Queen
    if not pca_fitted and count < expected_total:
        log(f"Waiting for all agents to report... ({count}/{expected_total})", "VIZ")
        return jsonify(
            {
                "status": "waiting_for_agents",
                "data": [],
                "count": count,
                "expected": expected_total,
            }
        )

    raw_vectors = physics_engine.space.vector_view[:count]

    # 2. Dimensionality Reduction (Fixed PCA - fit once, reuse)
    # We wrap in try/except because PCA can fail if vectors are identical (zero variance)
    try:
        if not pca_fitted:
            # First time: fit the PCA and transform (after all agents have reported)
            log(
                f"All {expected_agent_count} agents ready! Fitting PCA model (locking camera angle)...",
                "VIZ",
            )
            reduced = pca_model.fit_transform(raw_vectors)
            pca_fitted = True
            log("✓ PCA locked - Queen position fixed", "VIZ")
        else:
            # Subsequent times: just transform using the fitted model
            reduced = pca_model.transform(raw_vectors)
    except Exception as e:
        # Fallback if system is perfectly static/aligned
        return jsonify({"status": "calculating", "error": str(e)})

    snapshot = []

    # 3. Format Data
    # Index 0 is QUEEN
    snapshot.append(
        {
            "id": "QUEEN",
            "type": "queen",
            "x": float(reduced[0][0]),
            "y": float(reduced[0][1]),
            "z": float(reduced[0][2]),
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

    return jsonify(
        {
            "status": "active",
            "count": count,
            "data": snapshot,
            "chat_logs": chat_logs,  # Include chat logs from all agents
        }
    )


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
