"""
Vector Swarm Manager - Core Business Logic
Handles swarm orchestration, physics engine, and agent lifecycle.
Separated from Flask API layer for cleaner architecture.
"""

import time
import uuid
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
from vector_db_service import VectorDBService
from agent_memory import QdrantMemoryStore


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
    - Vector database (Qdrant)
    - Physics engine (Qdrant-backed)
    - Embedding service (multiprocessing)
    - Agent processes
    - PCA model for visualization
    - Chat log collection
    """

    def __init__(self):
        # Core components
        self.vector_db = None  # VectorDBService instance
        self.physics_engine = None
        self.memory_store = None  # QdrantMemoryStore instance
        self.embedding_service = None
        self.llm_client = OpenAI(base_url=VLLM_API, api_key="dummy")

        # Run management
        self.run_id = None  # Current run identifier
        self.mission_text = None  # Current mission

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
        log("🚀 Initializing Swarm Manager...", "SYSTEM")

        # Initialize Vector Database Service
        log("📊 Connecting to Qdrant...", "SYSTEM")
        self.vector_db = VectorDBService(use_grpc=True)

        # Health check
        if not self.vector_db.health_check():
            log("❌ Qdrant health check failed! Is the docker container running?", "ERROR")
            raise ConnectionError("Cannot connect to Qdrant at localhost:6334")

        log("✓ Qdrant connection established", "SYSTEM")

        # Initialize global collections (missions, snapshots)
        log("Creating global collections...", "SYSTEM")
        if not self.vector_db.initialize_global_collections():
            log("❌ Failed to initialize global collections", "ERROR")
            raise RuntimeError("Global collection initialization failed")

        log("✓ Global collections ready", "SYSTEM")

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

        log("✅ Swarm Manager initialization complete", "SYSTEM")

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
            dict with status, mission, agent count, and run_id
        """
        # Store expected count for PCA fitting logic
        self.expected_agent_count = agent_count
        self.mission_text = mission

        log(f"⚡ START command: '{mission}' with {agent_count} agents", "API")

        # 1. Stop existing agents if any
        self.stop_swarm()

        # 2. Generate new run ID
        self.run_id = str(uuid.uuid4())
        log(f"📋 Generated run_id: {self.run_id}", "API")

        # 3. Create per-run collections
        log("Creating per-run collections...", "API")
        if not self.vector_db.initialize_run_collections(self.run_id):
            log("❌ Failed to create per-run collections", "ERROR")
            return {"status": "error", "message": "Failed to create collections"}

        log("✓ Per-run collections created", "API")

        # 3b. Verify collections are ready before spawning agents
        log("Verifying collections are accessible...", "API")
        from vector_db_config import get_current_collection_name
        import time
        max_retries = 10
        for i in range(max_retries):
            try:
                # Try to get collection info to verify it's ready
                self.vector_db.client.get_collection(get_current_collection_name(self.run_id))
                log("✓ Collections verified and ready", "API")
                break
            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(0.1)  # Wait 100ms before retry
                else:
                    log(f"⚠️ Collection verification timeout: {e}", "WARN")
                    # Continue anyway, agents will retry

        # 4. Initialize memory store for this run
        self.memory_store = QdrantMemoryStore(
            run_id=self.run_id,
            vector_db_service=self.vector_db,
            vector_dim=VECTOR_DIM
        )

        # 5. Initialize physics engine
        self.physics_engine = SwarmPhysics(
            memory_store=self.memory_store,
            vector_dim=VECTOR_DIM
        )

        # 6. Embed and set mission vector (V_queen)
        log("Embedding mission...", "API")
        queen_vec = self.get_embedding(mission)
        self.physics_engine.set_queen_signal(queen_vec, mission, agent_count)
        log("✓ Mission vector set", "API")

        # 7. Persist mission to global missions collection (for analytics)
        log("Persisting mission to global collection...", "API")
        from vector_db_config import MISSIONS_COLLECTION

        mission_payload = {
            "text": mission,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "agent_count": agent_count,
            "status": "running"
        }

        self.vector_db.upsert_vector(
            MISSIONS_COLLECTION,
            self.run_id,  # point_id = run_id
            queen_vec.tolist(),
            mission_payload
        )
        log("✓ Mission persisted to analytics collection", "API")

        # 8. Spawn Agents
        log(f"Spawning {agent_count} agent processes...", "API")
        for i in range(agent_count):
            p = Process(
                target=_run_agent_process,
                args=(
                    i,
                    self.run_id,
                    mission,
                    self.chat_log_queue,
                    # Don't pass embedding_service - it contains unpicklable multiprocessing objects
                    # Each agent will create its own embedding client
                ),
            )
            p.start()
            self.agent_processes.append(p)
        log(f"✓ All {agent_count} agents spawned", "API")

        return {
            "status": "started",
            "mission": mission,
            "agents": agent_count,
            "run_id": self.run_id
        }

    def stop_swarm(self, cleanup=None):
        """
        Stop all running agents and optionally cleanup collections.

        Args:
            cleanup: If True, delete per-run collections. If False, preserve them.
                    If None, use QDRANT_DELETE_ON_COMPLETE config (default behavior).
        """
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

        # Update mission status in global collection
        if self.run_id:
            from vector_db_config import MISSIONS_COLLECTION, QDRANT_DELETE_ON_COMPLETE

            # Get current mission data
            result = self.vector_db.get_by_id(MISSIONS_COLLECTION, self.run_id)
            if result:
                vector, payload = result
                payload["status"] = "completed"
                # Note: would need to calculate duration, but we don't track start time here
                self.vector_db.upsert_vector(
                    MISSIONS_COLLECTION,
                    self.run_id,
                    vector,
                    payload
                )
                log("✓ Mission marked as completed in global collection", "API")

            # Determine if cleanup should happen
            # Priority: explicit cleanup parameter > config default
            should_cleanup = cleanup if cleanup is not None else QDRANT_DELETE_ON_COMPLETE

            # Cleanup per-run collections if requested
            if should_cleanup and self.memory_store:
                log("Cleaning up per-run collections...", "API")
                if self.memory_store.cleanup():
                    log("✓ Per-run collections deleted", "API")
                else:
                    log("⚠️  Some collections failed to delete", "WARN")
            elif self.memory_store:
                log("Per-run collections preserved for inspection", "API")

        # Reset state for next run
        self.pca_fitted = False
        self.expected_agent_count = 0
        self.chat_logs = []
        self.run_id = None
        self.mission_text = None
        self.physics_engine = None
        self.memory_store = None

        log("All agents stopped, state reset for next mission", "API")

    def get_state(self):
        """
        Get current swarm state for visualization.

        Returns:
            dict with status, count, data (3D coordinates), and chat logs
        """
        if not self.memory_store:
            return {"status": "idle", "data": []}

        # 1. Get agent count from Qdrant
        count = self.memory_store.count_agents()

        # Add 1 for queen
        total_count = count + 1

        # Need at least 3 points for 3D PCA
        if total_count < 3:
            return {"status": "waiting_for_entropy", "data": []}

        # Check if all agents have reported in
        expected_total = self.expected_agent_count + 1  # +1 for Queen
        if not self.pca_fitted and total_count < expected_total:
            log(f"Waiting for all agents to report... ({total_count}/{expected_total})", "VIZ")
            return {
                "status": "waiting_for_agents",
                "data": [],
                "count": total_count,
                "expected": expected_total,
            }

        # 2. Collect vectors for PCA
        vectors = []

        # Get queen vector
        queen_vec = self.memory_store.get_mission()
        if queen_vec is not None:
            vectors.append(queen_vec)

        # Get all agent vectors
        agents = self.memory_store.get_all_agents(limit=MAX_AGENTS)
        for agent_id, vec, payload in agents:
            vectors.append(vec)

        if len(vectors) < 3:
            return {"status": "calculating", "data": []}

        raw_vectors = np.array(vectors, dtype=np.float32)

        # 3. Dimensionality Reduction (Fixed PCA - fit once, reuse)
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

        # 4. Format Data
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

        # Remaining indices are AGENTS
        for i in range(1, len(vectors)):
            agent_idx = i - 1  # Adjust for queen at index 0
            snapshot.append(
                {
                    "id": f"Agent-{agent_idx}",
                    "type": "drone",
                    "x": float(reduced[i][0]),
                    "y": float(reduced[i][1]),
                    "z": float(reduced[i][2]),
                    "label": f"Drone-{agent_idx}",
                }
            )

        return {
            "status": "active",
            "count": len(vectors),
            "run_id": self.run_id,
            "data": snapshot,
            "chat_logs": self.chat_logs,
        }

    def cleanup(self):
        """Cleanup resources on shutdown"""
        log("Swarm manager cleanup initiated", "SYSTEM")
        self.stop_swarm()

        if self.vector_db:
            self.vector_db.close()
            log("Vector database connection closed", "SYSTEM")

        if self.embedding_service:
            self.embedding_service.shutdown()
            log("Embedding service stopped", "SYSTEM")


# --- Agent Process Entry Point (Must be top-level for pickling) ---
def _run_agent_process(agent_id, run_id, mission, log_queue):
    """
    The entry point for a child process.
    Attaches to Qdrant collections for the given run_id.

    Args:
        agent_id: ID of this agent
        run_id: UUID for this swarm run
        mission: The mission/task for this agent
        log_queue: multiprocessing.Queue for chat logs
    """
    log(f"Agent {agent_id} starting up (run_id: {run_id})", "AGENT")

    # 1. Initialize embedding service for this agent
    # Each agent gets its own embedding service client
    agent_embedding_service = create_embedding_service(
        backend_type="fastembed",
        model_name=EMBED_MODEL
    )

    # 2. Initialize Vector DB Service (use HTTP to avoid gRPC fork issues)
    vector_db = VectorDBService(use_grpc=False)

    # 3. Create Memory Store (attach to existing run collections)
    memory_store = QdrantMemoryStore(
        run_id=run_id,
        vector_db_service=vector_db,
        vector_dim=VECTOR_DIM
    )

    # 4. Initialize Physics Engine
    physics = SwarmPhysics(memory_store=memory_store, vector_dim=VECTOR_DIM)

    # 5. Setup Local LLM Client with Logging Wrapper
    base_client = OpenAI(base_url=VLLM_API, api_key="dummy")
    logged_client = LoggedLLMClient(base_client, agent_id, log_queue)

    # 6. Helper function to get embeddings
    def get_embedding(text):
        return agent_embedding_service.embed(text)

    # 6. Create Agent Instance
    agent = VectorAgent(
        agent_id=agent_id,
        run_id=run_id,
        physics_engine=physics,
        llm_client=logged_client,
        embed_func=get_embedding,
        starting_task=mission,
        log_queue=log_queue,
    )

    # 7. The Infinite Loop
    try:
        log(f"Agent {agent_id} entering main loop", "AGENT")
        while True:
            agent.step()
            # Jitter the sleep to prevent lock-step behavior
            time.sleep(np.random.uniform(1.0, 2.0))
    except KeyboardInterrupt:
        log(f"Agent {agent_id} shutting down", "AGENT")
    finally:
        # Cleanup
        agent_embedding_service.shutdown()
        physics.cleanup()
        vector_db.close()
