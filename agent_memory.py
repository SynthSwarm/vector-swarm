"""
Agent Memory Store
Provides vector storage for swarm physics calculations using Qdrant.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from vector_db_service import VectorDBService
from vector_db_config import (
    get_current_collection_name,
    get_historical_collection_name,
    get_mission_collection_name
)


logger = logging.getLogger(__name__)


class QdrantMemoryStore:
    """
    Qdrant-backed memory store for swarm physics.
    Replaces the previous SharedMemoryStore with persistent vector database storage.
    """

    def __init__(self, run_id: str, vector_db_service: VectorDBService, vector_dim: int = 768):
        """
        Initialize Qdrant-backed memory store.

        Args:
            run_id: Unique identifier for this swarm run
            vector_db_service: Initialized VectorDBService instance
            vector_dim: Dimension of embeddings (768 for nomic-embed-text-v1.5)
        """
        self.run_id = run_id
        self.db = vector_db_service
        self.vector_dim = vector_dim

        # Collection names for this run
        self.current_collection = get_current_collection_name(run_id)
        self.historical_collection = get_historical_collection_name(run_id)
        self.mission_collection = get_mission_collection_name(run_id)

        logger.info(f"QdrantMemoryStore initialized for run_id: {run_id}")

    # Mission Vector Operations (V_queen)

    def set_mission(self, vector: np.ndarray, mission_text: str, agent_count: int) -> bool:
        """
        Set the mission vector (V_queen) for this run.

        Args:
            vector: Mission embedding vector
            mission_text: Mission description text
            agent_count: Number of agents in swarm

        Returns:
            True if successful
        """
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        payload = {
            "text": mission_text,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "agent_count": agent_count
        }

        success = self.db.upsert_vector(
            self.mission_collection,
            "00000000-0000-0000-0000-000000000000",  # Well-known UUID for mission
            vector.tolist(),
            payload
        )

        if success:
            logger.info(f"Mission vector set for run {self.run_id}")
        return success

    def get_mission(self) -> Optional[np.ndarray]:
        """
        Get the mission vector (V_queen).

        Returns:
            Mission vector as numpy array, or None if not found
        """
        result = self.db.get_by_id(self.mission_collection, "00000000-0000-0000-0000-000000000000")
        if result:
            vector, payload = result
            return np.array(vector, dtype=np.float32)
        return None

    # Current Agent Operations (V_body, V_next)

    def upsert_agent(self, agent_id: int, vector: np.ndarray, payload: Dict[str, Any]) -> bool:
        """
        Upsert agent's current vector (V_body or V_next).
        Each agent has one point in the current collection, identified by agent_id.

        Args:
            agent_id: Agent identifier
            vector: Agent's current vector
            payload: Metadata (text, timestamp, action_type, status)

        Returns:
            True if successful
        """
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Ensure agent_id is in payload
        payload["agent_id"] = agent_id
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.now().isoformat()

        return self.db.upsert_vector(
            self.current_collection,
            agent_id,  # point_id = agent_id (integer)
            vector.tolist(),
            payload
        )

    def get_agent(self, agent_id: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get agent's current vector and metadata.

        Args:
            agent_id: Agent identifier

        Returns:
            (vector, payload) tuple or None if not found
        """
        result = self.db.get_by_id(self.current_collection, agent_id)
        if result:
            vector, payload = result
            return np.array(vector, dtype=np.float32), payload
        return None

    def get_all_agents(self, exclude_agent_id: Optional[int] = None, limit: int = 1000) -> List[Tuple[int, np.ndarray, Dict[str, Any]]]:
        """
        Get all agent vectors (for flock calculations).

        Args:
            exclude_agent_id: Optional agent ID to exclude
            limit: Maximum number of agents to retrieve

        Returns:
            List of (agent_id, vector, payload) tuples
        """
        all_agents = self.db.get_all_from_collection(
            self.current_collection,
            limit=limit,
            offset=0
        )

        results = []
        for point_id, vector, payload in all_agents:
            # Convert point_id to int (it might be string or int)
            try:
                agent_id = int(point_id) if isinstance(point_id, str) else point_id
            except ValueError:
                # Skip non-integer point IDs (shouldn't happen in current collection)
                continue

            # Skip excluded agent
            if exclude_agent_id is not None and agent_id == exclude_agent_id:
                continue

            results.append((
                agent_id,
                np.array(vector, dtype=np.float32),
                payload
            ))

        return results

    def count_agents(self) -> int:
        """
        Count active agents in current collection.

        Returns:
            Number of agents
        """
        agents = self.db.get_all_from_collection(
            self.current_collection,
            limit=10000  # High limit to get all
        )
        return len(agents)

    # Historical Actions (V_been)

    def append_historical(self, agent_id: int, vector: np.ndarray,
                         text: str, score: float, step: int) -> Optional[str]:
        """
        Append completed action to historical collection.

        Args:
            agent_id: Agent who performed the action
            vector: Action embedding vector
            text: Action description
            score: Alignment score (dot product with v_next)
            step: Agent step number

        Returns:
            Point ID if successful, None otherwise
        """
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        payload = {
            "text": text,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "score": float(score),
            "step": step
        }

        point_id = self.db.append_vector(
            self.historical_collection,
            vector.tolist(),
            payload
        )

        if point_id:
            logger.debug(f"Agent {agent_id} action appended to history: {text[:50]}...")
        return point_id

    def search_historical(self, query_vector: np.ndarray,
                         limit: int = 10,
                         score_threshold: Optional[float] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search historical actions by similarity.

        Args:
            query_vector: Query embedding
            limit: Max results to return
            score_threshold: Minimum similarity score

        Returns:
            List of (point_id, score, payload) tuples
        """
        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        results = self.db.search_similar(
            self.historical_collection,
            query_vector.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )

        return results

    # Physics Helper Methods

    def calculate_flock_vector(self, exclude_agent_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Calculate average vector of all other agents (V_flock).

        Args:
            exclude_agent_id: Agent to exclude from calculation

        Returns:
            Mean vector of other agents, or None if no agents
        """
        agents = self.get_all_agents(exclude_agent_id=exclude_agent_id)

        if not agents:
            return None

        vectors = np.array([vec for _, vec, _ in agents], dtype=np.float32)
        mean_vec = np.mean(vectors, axis=0)

        # Normalize
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm

        return mean_vec

    def find_closest_agent(self, my_vector: np.ndarray, my_agent_id: int) -> Optional[Tuple[int, np.ndarray, float]]:
        """
        Find the closest agent to the given vector (for separation).

        Args:
            my_vector: Current agent's vector
            my_agent_id: Current agent's ID (to exclude self)

        Returns:
            (agent_id, vector, similarity) tuple of closest agent, or None
        """
        agents = self.get_all_agents(exclude_agent_id=my_agent_id)

        if not agents:
            return None

        # Calculate similarities
        max_sim = -1.0
        closest_agent = None

        for agent_id, vector, payload in agents:
            sim = np.dot(my_vector, vector)
            if sim > max_sim:
                max_sim = sim
                closest_agent = (agent_id, vector, float(sim))

        return closest_agent

    # Lifecycle Methods

    def close(self):
        """
        Close the memory store.
        Note: Qdrant client is managed by VectorDBService, not closed here.
        """
        logger.info(f"QdrantMemoryStore closed for run {self.run_id}")

    def cleanup(self) -> bool:
        """
        Delete all per-run collections for this run.
        Call this after run completion if cleanup is desired.

        Returns:
            True if all collections deleted successfully
        """
        logger.info(f"Cleaning up collections for run {self.run_id}")
        return self.db.cleanup_run_collections(self.run_id)
