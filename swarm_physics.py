import numpy as np
from agent_memory import SharedMemoryStore

class SwarmPhysics:
    def __init__(self, max_agents=50, dim=768, create=False):
        """
        The Physics Engine of the Vector Swarm.
        Manages the positions of all agents and the Queen.
        """
        self.dim = dim
        
        # 1. The Queen's Signal (Immutable, Global)
        # We store this in a tiny separate shared block or just pass it in.
        # For simplicity, we assume the Queen is set once in the shared slab at index 0.
        # Indices 1 to max_agents are for the agents.
        self.space = SharedMemoryStore(
            name="swarm_physics_plane", 
            max_items=max_agents + 1, 
            vector_dim=dim, 
            create=create
        )
        
        # Reserved Index 0 is the QUEEN
        self.QUEEN_IDX = 0

    def set_queen_signal(self, vector):
        """Sets V_queen (Layer 1: The Mission)"""
        # Overwrite index 0
        # We manually access the buffer to force overwrite
        vector = vector / np.linalg.norm(vector)
        self.space.vector_view[self.QUEEN_IDX] = vector.astype(np.float32)
        # Increment counter to make Queen visible (count = 1)
        if self.space.counter_view[0] < 1:
            self.space.counter_view[0] = 1

    def update_body_signal(self, agent_id, vector):
        """Updates V_body (Layer 3: Atomic Position)"""
        # Map agent_id (0, 1, 2) to memory slots (1, 2, 3)
        slot = agent_id + 1
        
        norm = np.linalg.norm(vector)
        if norm > 0: vector = vector / norm
        
        self.space.vector_view[slot] = vector.astype(np.float32)
        # Ensure counter is high enough to make this slot "visible"
        if self.space.counter_view[0] < slot + 1:
             self.space.counter_view[0] = slot + 1

    def calculate_trajectory(self, agent_id, w_c=1.0, w_a=0.5, w_s=0.8):
        """
        THE RESOLUTION EQUATION (Section 5.2)
        Returns V_next: The ideal vector direction the agent *should* move towards.
        """
        slot = agent_id + 1
        
        # 1. Get V_queen (Cohesion)
        v_queen = self.space.vector_view[self.QUEEN_IDX]
        
        # 2. Get V_flock (Alignment)
        # Average of all *other* active agents
        # We grab the whole buffer up to current count
        limit = self.space.counter_view[0]
        # Exclude Queen (0) and Self (slot)
        others_indices = [i for i in range(1, limit) if i != slot]

        # Early exit if no other agents
        if not others_indices:
            v_flock = np.zeros(self.dim, dtype=np.float32)
            v_separation = np.zeros(self.dim, dtype=np.float32)
        else:
            # Get all other agents' vectors once
            others_vectors = self.space.vector_view[others_indices]

            # 2. Calculate V_flock (Alignment)
            v_flock = np.mean(others_vectors, axis=0)
            v_flock = v_flock / (np.linalg.norm(v_flock) + 1e-9)

            # 3. Get V_body Repulsion (Separation)
            # Find the CLOSEST neighbor and create a repulsive vector
            my_vec = self.space.vector_view[slot]
            # Calculate distances to all others
            # (Using dot product as proxy for distance in unit sphere)
            sims = np.dot(others_vectors, my_vec)
            closest_idx = np.argmax(sims)

            # Repulsion vector is: My_Pos - Their_Pos
            # "Push me away from them"
            closest_vec = others_vectors[closest_idx]
            repulsion = my_vec - closest_vec
            if np.linalg.norm(repulsion) > 0:
                v_separation = repulsion / np.linalg.norm(repulsion)
            else:
                v_separation = np.zeros(self.dim, dtype=np.float32)

        # 4. Synthesize V_next
        # V_next = (w_c * V_queen) + (w_a * V_flock) + (w_s * V_separation)
        v_next = (w_c * v_queen) + (w_a * v_flock) + (w_s * v_separation)
        
        # Normalize Result
        return v_next / (np.linalg.norm(v_next) + 1e-9)
    
    def cleanup(self):
        self.space.close()
    
    def destroy(self):
        self.space.destroy()