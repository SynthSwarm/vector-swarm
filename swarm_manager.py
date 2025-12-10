"""
Vector Swarm Manager - Core Business Logic
Handles swarm orchestration, physics engine, and agent lifecycle.
Separated from Flask API layer for cleaner architecture.
"""

import time
import numpy as np
import threading
from datetime import datetime
from multiprocessing import Process, Queue
from sklearn.decomposition import PCA
from openai import OpenAI

# Import our custom modules
from swarm_physics import SwarmPhysics
from swarm_agent import VectorAgent
from logged_llm_client import LoggedLLMClient
from embedding_service import create_embedding_service


# --- Logging Helper ---
def log(message, level="INFO"):
    """Timestamped logging helper"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}", flush=True)


# --- Configuration Constants ---
MAX_AGENTS = 20
VECTOR_DIM = 768
VLLM_API = "http://localhost:8000/v1"  # vLLM OpenAI-compatible endpoint
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # FastEmbed model
MAX_CHAT_LOGS = 50


class SwarmManager:
    """
    Manages the entire swarm system:
    - Physics engine (shared memory)
    - Embedding service (multiprocessing)
    - Agent processes
    - PCA model for visualization
    - Chat log collection
    """

    def __init__(self):
        # Core components
        self.physics_engine = None
        self.embedding_service = None
        self.llm_client = OpenAI(base_url=VLLM_API, api_key="dummy")

        # Agent management
        self.agent_processes = []
        self.expected_agent_count = 0

        # Visualization
        self.pca_model = PCA(n_components=3)
        self.pca_fitted = False

        # Chat logging
        self.chat_log_queue = None
        self.chat_logs = []
        self.log_drain_thread = None

    def initialize(self):
        """Initialize all system components"""
        # Initialize the Physics Engine (Allocates Shared RAM)
        self.physics_engine = SwarmPhysics(
            max_agents=MAX_AGENTS, dim=VECTOR_DIM, create=True
        )
        log("🐝 Swarm Physics Engine initialized", "SYSTEM")

        # Initialize Embedding Service (runs in separate process)
        log("📊 Initializing Embedding Service...", "SYSTEM")
        self.embedding_service = create_embedding_service(
            backend_type="fastembed", model_name=EMBED_MODEL
        )
        log("✓ Embedding Service ready", "SYSTEM")

        # Initialize Chat Log Queue
        self.chat_log_queue = Queue(maxsize=200)  # Buffer up to 200 messages
        log("📝 Chat log queue initialized", "SYSTEM")

        # Start background thread to drain log queue
        self.log_drain_thread = threading.Thread(
            target=self._drain_log_queue, daemon=True
        )
        self.log_drain_thread.start()
        log("🔄 Log drain thread started", "SYSTEM")

    def get_embedding(self, text):
        """
        Get embedding using the embedding service.

        Args:
            text: Text to embed

        Returns:
            numpy array of embeddings
        """
        if self.embedding_service is None:
            log("Embedding service not initialized!", "ERROR")
            return np.zeros(VECTOR_DIM, dtype=np.float32)

        return self.embedding_service.embed(text)

    def _drain_log_queue(self):
        """
        Background thread that continuously drains the chat_log_queue
        and appends entries to the in-memory chat_logs buffer.
        """
        while True:
            try:
                # Safety check: ensure queue is initialized
                if self.chat_log_queue is None:
                    time.sleep(0.1)
                    continue

                # Block until we get a log entry (or timeout after 1 second)
                entry = self.chat_log_queue.get(timeout=1.0)
                self.chat_logs.append(entry)

                # Trim to last N entries
                if len(self.chat_logs) > MAX_CHAT_LOGS:
                    self.chat_logs = self.chat_logs[-MAX_CHAT_LOGS:]

            except Exception:
                # Queue.Empty exception or other errors - just continue
                time.sleep(0.1)

    def start_swarm(self, mission, agent_count):
        """
        Start the swarm with a given mission and number of agents.

        Args:
            mission: The mission/task description
            agent_count: Number of agents to spawn

        Returns:
            dict with status, mission, and agent count
        """
        # Store expected count for PCA fitting logic
        self.expected_agent_count = agent_count

        log(f"⚡ START command: '{mission}' with {agent_count} agents", "API")

        # 1. Stop existing agents if any
        self.stop_swarm()

        # 2. Inject the Queen Vector (The Mission)
        log("Embedding queen mission signal...", "API")
        queen_vec = self.get_embedding(mission)
        self.physics_engine.set_queen_signal(queen_vec)
        log("✓ Queen signal injected into shared memory", "API")

        # 3. Spawn Agents
        log(f"Spawning {agent_count} agent processes...", "API")
        for i in range(agent_count):
            p = Process(
                target=_run_agent_process,
                args=(i, mission, self.chat_log_queue, self.embedding_service),
            )
            p.start()
            self.agent_processes.append(p)
        log(f"✓ All {agent_count} agents spawned", "API")

        return {"status": "started", "mission": mission, "agents": agent_count}

    def stop_swarm(self):
        """Stop all running agents and reset state"""
        if not self.agent_processes:
            log("No active agents to stop", "API")
            return

        log(f"🛑 STOP command: terminating {len(self.agent_processes)} agents...", "API")
        for i, p in enumerate(self.agent_processes):
            if p.is_alive():
                p.terminate()
                p.join()
                log(f"✓ Agent {i} terminated", "API")
        self.agent_processes = []

        # Reset PCA and agent count for next run
        self.pca_fitted = False
        self.expected_agent_count = 0

        # Clear chat logs for next run
        self.chat_logs = []

        log("All agents stopped, PCA and chat logs reset for next mission", "API")

    def get_state(self):
        """
        Get current swarm state for visualization.

        Returns:
            dict with status, count, data (3D coordinates), and chat logs
        """
        # 1. Read Raw Vectors from Shared Memory
        count = int(self.physics_engine.space.counter_view[0])

        # Need at least 3 points for 3D PCA to make sense
        if count < 3:
            return {"status": "waiting_for_entropy", "data": []}

        # Check if all agents have reported in (count = Queen + all agents)
        expected_total = self.expected_agent_count + 1  # +1 for Queen
        if not self.pca_fitted and count < expected_total:
            log(f"Waiting for all agents to report... ({count}/{expected_total})", "VIZ")
            return {
                "status": "waiting_for_agents",
                "data": [],
                "count": count,
                "expected": expected_total,
            }

        raw_vectors = self.physics_engine.space.vector_view[:count]

        # 2. Dimensionality Reduction (Fixed PCA - fit once, reuse)
        try:
            if not self.pca_fitted:
                # First time: fit the PCA and transform (after all agents have reported)
                log(
                    f"All {self.expected_agent_count} agents ready! Fitting PCA model (locking camera angle)...",
                    "VIZ",
                )
                reduced = self.pca_model.fit_transform(raw_vectors)
                self.pca_fitted = True
                log("✓ PCA locked - Queen position fixed", "VIZ")
            else:
                # Subsequent times: just transform using the fitted model
                reduced = self.pca_model.transform(raw_vectors)
        except Exception as e:
            # Fallback if system is perfectly static/aligned
            return {"status": "calculating", "error": str(e)}

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

        return {
            "status": "active",
            "count": count,
            "data": snapshot,
            "chat_logs": self.chat_logs,
        }

    def cleanup(self):
        """Cleanup resources on shutdown"""
        log("Swarm manager cleanup initiated", "SYSTEM")
        self.stop_swarm()
        if self.physics_engine:
            self.physics_engine.destroy()
            log("Shared memory destroyed", "SYSTEM")
        if self.embedding_service:
            self.embedding_service.shutdown()
            log("Embedding service stopped", "SYSTEM")


# --- Agent Process Entry Point (Must be top-level for pickling) ---
def _run_agent_process(agent_id, mission, log_queue, embedding_service):
    """
    The entry point for a child process.
    Re-instantiates the Physics engine (create=False) to attach to shared memory.

    Args:
        agent_id: ID of this agent
        mission: The mission/task for this agent
        log_queue: multiprocessing.Queue for chat logs
        embedding_service: Shared embedding service instance
    """
    log(f"Agent {agent_id} starting up", "AGENT")

    # 1. Attach to Physics Engine
    physics = SwarmPhysics(max_agents=MAX_AGENTS, dim=VECTOR_DIM, create=False)

    # 2. Setup Local LLM Client with Logging Wrapper
    base_client = OpenAI(base_url=VLLM_API, api_key="dummy")
    logged_client = LoggedLLMClient(base_client, agent_id, log_queue)

    # 3. Helper function to get embeddings
    def get_embedding(text):
        if embedding_service is None:
            log("Embedding service not initialized!", "ERROR")
            return np.zeros(VECTOR_DIM, dtype=np.float32)
        return embedding_service.embed(text)

    # 4. Create Agent Instance
    agent = VectorAgent(
        agent_id=agent_id,
        physics_engine=physics,
        llm_client=logged_client,
        embed_func=get_embedding,
        starting_task=mission,
        log_queue=log_queue,
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
