"""
Swarm Physics Engine
Implements the Resolution Equation for agent trajectory calculation using Qdrant.
"""

import numpy as np
import logging
from agent_memory import QdrantMemoryStore


logger = logging.getLogger(__name__)


class SwarmPhysics:
    """
    The Physics Engine of the Vector Swarm.
    Calculates agent trajectories based on mission alignment, flock behavior, and separation.
    """

    def __init__(self, memory_store: QdrantMemoryStore, vector_dim: int = 768):
        """
        Initialize the physics engine.

        Args:
            memory_store: QdrantMemoryStore instance for this run
            vector_dim: Dimension of embeddings (768 for nomic-embed-text-v1.5)
        """
        self.memory = memory_store
        self.dim = vector_dim
        logger.info(f"SwarmPhysics initialized for run {memory_store.run_id}")

    def set_queen_signal(self, vector: np.ndarray, mission_text: str, agent_count: int) -> bool:
        """
        Sets V_queen (Layer 1: The Mission).

        Args:
            vector: Mission embedding vector
            mission_text: Mission description
            agent_count: Number of agents in swarm

        Returns:
            True if successful
        """
        success = self.memory.set_mission(vector, mission_text, agent_count)
        if success:
            logger.info(f"Queen signal set: {mission_text[:50]}...")
        return success

    def update_body_signal(self, agent_id: int, vector: np.ndarray,
                          text: str, action_type: str = "body",
                          status: str = "active") -> bool:
        """
        Updates V_body (Layer 3: Atomic Position).
        Upserts agent's current vector to the current collection.

        Args:
            agent_id: Agent identifier
            vector: Agent's current vector
            text: Current action description
            action_type: "body" or "next"
            status: "active", "completed", or "blocked"

        Returns:
            True if successful
        """
        payload = {
            "text": text,
            "action_type": action_type,
            "status": status
        }

        success = self.memory.upsert_agent(agent_id, vector, payload)
        if success:
            logger.debug(f"Agent {agent_id} body signal updated: {text[:30]}...")
        return success

    def calculate_trajectory(self, agent_id: int, w_c: float = 1.0,
                           w_a: float = 0.5, w_s: float = 0.8) -> np.ndarray:
        """
        THE RESOLUTION EQUATION (Section 5.2)
        Returns V_next: The ideal vector direction the agent should move towards.

        Formula: V_next = (w_c * V_queen) + (w_a * V_flock) + (w_s * V_separation)

        Args:
            agent_id: Agent whose trajectory to calculate
            w_c: Cohesion weight (obedience to mission) - default 1.0
            w_a: Alignment weight (social conformity) - default 0.5
            w_s: Separation weight (personal space) - default 0.8

        Returns:
            Normalized V_next vector
        """
        # 1. Get V_queen (Cohesion)
        v_queen = self.memory.get_mission()
        if v_queen is None:
            logger.warning("Mission vector not set, using zero vector for cohesion")
            v_queen = np.zeros(self.dim, dtype=np.float32)

        # 2. Get agent's current vector
        agent_result = self.memory.get_agent(agent_id)
        if agent_result is None:
            logger.warning(f"Agent {agent_id} not found in current collection")
            # Return just the mission vector as fallback
            return v_queen

        my_vec, _ = agent_result

        # 3. Calculate V_flock (Alignment)
        # Average of all other active agents
        v_flock = self.memory.calculate_flock_vector(exclude_agent_id=agent_id)
        if v_flock is None:
            # No other agents, use zero vector
            v_flock = np.zeros(self.dim, dtype=np.float32)
            v_separation = np.zeros(self.dim, dtype=np.float32)
        else:
            # 4. Calculate V_separation (Repulsion from closest neighbor)
            closest = self.memory.find_closest_agent(my_vec, agent_id)

            if closest is None:
                # No other agents (shouldn't happen if v_flock is not None, but safety)
                v_separation = np.zeros(self.dim, dtype=np.float32)
            else:
                closest_id, closest_vec, similarity = closest

                # Repulsion vector: My position - Their position
                # "Push me away from them"
                repulsion = my_vec - closest_vec
                norm = np.linalg.norm(repulsion)

                if norm > 1e-9:
                    v_separation = repulsion / norm
                else:
                    v_separation = np.zeros(self.dim, dtype=np.float32)

                logger.debug(f"Agent {agent_id} closest neighbor: {closest_id} (sim={similarity:.3f})")

        # 5. Synthesize V_next
        # V_next = (w_c * V_queen) + (w_a * V_flock) + (w_s * V_separation)
        v_next = (w_c * v_queen) + (w_a * v_flock) + (w_s * v_separation)

        # Normalize result
        norm = np.linalg.norm(v_next)
        if norm > 1e-9:
            v_next = v_next / norm
        else:
            # Fallback to mission vector if result is zero
            logger.warning(f"Agent {agent_id} trajectory is zero, falling back to mission")
            v_next = v_queen

        return v_next

    def cleanup(self):
        """Close the memory store."""
        self.memory.close()

    def destroy(self):
        """Delete all per-run collections (if cleanup is desired)."""
        logger.info(f"Destroying per-run collections for {self.memory.run_id}")
        self.memory.cleanup()
