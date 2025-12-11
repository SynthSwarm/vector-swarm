"""
Vector Database Configuration
Defines collection schemas, HNSW parameters, and metadata fields for Qdrant.
"""

import os
import hashlib
from dataclasses import dataclass
from typing import Dict, Any
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff


# Environment variable configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))

# Per-run collection settings
QDRANT_RUN_RETENTION_DAYS = int(os.getenv("QDRANT_RUN_RETENTION_DAYS", "7"))
QDRANT_AUTO_CLEANUP = os.getenv("QDRANT_AUTO_CLEANUP", "true").lower() == "true"
QDRANT_DELETE_ON_COMPLETE = os.getenv("QDRANT_DELETE_ON_COMPLETE", "false").lower() == "true"

# Global collection settings
QDRANT_SNAPSHOT_INTERVAL_SEC = int(os.getenv("QDRANT_SNAPSHOT_INTERVAL_SEC", "300"))
QDRANT_SNAPSHOT_RETENTION_DAYS = int(os.getenv("QDRANT_SNAPSHOT_RETENTION_DAYS", "30"))

# Performance tuning
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "100"))
QDRANT_ASYNC_QUEUE_SIZE = int(os.getenv("QDRANT_ASYNC_QUEUE_SIZE", "10000"))
QDRANT_HNSW_M = int(os.getenv("QDRANT_HNSW_M", "16"))
QDRANT_HNSW_EF_CONSTRUCT = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "100"))
QDRANT_QUERY_EF = int(os.getenv("QDRANT_QUERY_EF", "128"))
QDRANT_ON_DISK_PAYLOAD = os.getenv("QDRANT_ON_DISK_PAYLOAD", "false").lower() == "true"

# Vector dimension (nomic-ai/nomic-embed-text-v1.5)
VECTOR_DIM = 768


@dataclass
class CollectionConfig:
    """Configuration for a Qdrant collection."""
    name: str
    description: str
    vector_params: VectorParams
    hnsw_config: HnswConfigDiff


def get_hnsw_config() -> HnswConfigDiff:
    """Get HNSW index configuration for optimal performance."""
    return HnswConfigDiff(
        m=QDRANT_HNSW_M,
        ef_construct=QDRANT_HNSW_EF_CONSTRUCT,
        on_disk=False  # Keep index in memory for fast queries
    )


def get_vector_params() -> VectorParams:
    """Get vector parameters (dimension and distance metric)."""
    return VectorParams(
        size=VECTOR_DIM,
        distance=Distance.COSINE  # Matches current dot product similarity
    )


# Per-Run Collection Templates
# These are created dynamically for each swarm run with run_id

def _short_hash(run_id: str, length: int = 6) -> str:
    """
    Generate a short deterministic hash from a UUID.
    Used for readable collection names while maintaining uniqueness.

    Args:
        run_id: Full UUID string
        length: Number of hex characters to return (default: 6)

    Returns:
        Short hash string (e.g., "a3f2bc" for a UUID)
    """
    hash_digest = hashlib.sha256(run_id.encode('utf-8')).hexdigest()
    return hash_digest[:length]


def get_historical_collection_name(run_id: str) -> str:
    """Get collection name for historical actions (V_been)."""
    short_id = _short_hash(run_id)
    return f"run_{short_id}_historical"


def get_current_collection_name(run_id: str) -> str:
    """Get collection name for current actions (V_body + V_next)."""
    short_id = _short_hash(run_id)
    return f"run_{short_id}_current"


def get_mission_collection_name(run_id: str) -> str:
    """Get collection name for mission vector (V_queen)."""
    short_id = _short_hash(run_id)
    return f"run_{short_id}_mission"


# Global Collection Names
MISSIONS_COLLECTION = "missions"
SNAPSHOTS_COLLECTION = "snapshots"


def get_per_run_collection_configs(run_id: str) -> Dict[str, CollectionConfig]:
    """
    Get collection configurations for a specific swarm run.
    These collections are used in physics calculations.
    """
    return {
        "historical": CollectionConfig(
            name=get_historical_collection_name(run_id),
            description=f"Historical actions for run {run_id} (V_been)",
            vector_params=get_vector_params(),
            hnsw_config=get_hnsw_config()
        ),
        "current": CollectionConfig(
            name=get_current_collection_name(run_id),
            description=f"Current agent actions for run {run_id} (V_body + V_next)",
            vector_params=get_vector_params(),
            hnsw_config=get_hnsw_config()
        ),
        "mission": CollectionConfig(
            name=get_mission_collection_name(run_id),
            description=f"Mission vector for run {run_id} (V_queen)",
            vector_params=get_vector_params(),
            hnsw_config=get_hnsw_config()
        )
    }


def get_global_collection_configs() -> Dict[str, CollectionConfig]:
    """
    Get global collection configurations.
    These collections persist across all runs for analytics and observability.
    """
    return {
        "missions": CollectionConfig(
            name=MISSIONS_COLLECTION,
            description="All missions ever run (analytics only)",
            vector_params=get_vector_params(),
            hnsw_config=get_hnsw_config()
        ),
        "snapshots": CollectionConfig(
            name=SNAPSHOTS_COLLECTION,
            description="Periodic swarm state snapshots (debugging and replay)",
            vector_params=get_vector_params(),
            hnsw_config=get_hnsw_config()
        )
    }


# Payload field schemas (for documentation and validation)

HISTORICAL_PAYLOAD_SCHEMA = {
    "text": "string",           # Action description
    "agent_id": "integer",      # Which agent performed this
    "timestamp": "datetime",    # When action was taken
    "score": "float",           # Alignment score (dot product)
    "step": "integer"           # Agent step number
}

CURRENT_PAYLOAD_SCHEMA = {
    "text": "string",           # Current action description
    "agent_id": "integer",      # Which agent (point_id = agent_id for upsert)
    "timestamp": "datetime",    # Last update time
    "action_type": "string",    # "body" or "next"
    "status": "string"          # "active", "completed", "blocked"
}

MISSION_PAYLOAD_SCHEMA = {
    "text": "string",          # Mission description
    "run_id": "string",        # UUID for this run
    "timestamp": "datetime",   # When mission started
    "agent_count": "integer"   # Number of agents in swarm
}

MISSIONS_PAYLOAD_SCHEMA = {
    "text": "string",          # Mission description
    "run_id": "string",        # UUID for this run
    "timestamp": "datetime",   # When mission started
    "duration_sec": "float",   # How long swarm ran
    "agent_count": "integer",  # Number of agents
    "status": "string"         # "completed", "running", "failed"
}

SNAPSHOTS_PAYLOAD_SCHEMA = {
    "agent_id": "integer",
    "run_id": "string",         # Which run this snapshot belongs to
    "timestamp": "datetime",
    "task": "string",           # Current agent task
    "chat_log": "list[string]"  # Recent messages
}
