# swarm_agent.py
import numpy as np
from datetime import datetime
from swarm_physics import SwarmPhysics

def log(message, agent_id):
    """Timestamped logging for agents"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [Agent-{agent_id}] {message}", flush=True)


class VectorAgent:
    def __init__(self, agent_id, physics_engine: SwarmPhysics, llm_client, embed_func):
        self.id = agent_id
        self.physics = physics_engine
        self.llm = llm_client
        self.embed = embed_func
        self.current_task = "Idle"

        # Weights (Personality)
        self.w_c = 1.0  # Obedience to Queen
        self.w_a = 0.5  # Social Conformity
        self.w_s = 0.8  # Personal Space

    def step(self):
        # 1. Update my Body Vector in the shared universe
        # "I am currently analyzing logs"
        my_vec = self.embed(self.current_task)
        self.physics.update_body_signal(self.id, my_vec)

        # 2. Calculate the Desired Trajectory (V_next)
        # This uses the Resolution Equation
        v_next = self.physics.calculate_trajectory(
            self.id, self.w_c, self.w_a, self.w_s
        )

        # 3. Generate Options (The LLM Brainstorm)
        # We ask Qwen for distinct potential next actions
        prompt = f"""
        Current Task: {self.current_task}
        Generate 3 distinct, short next potential actions.
        Format: 1. [Action]
        """
        response = self.llm.chat.completions.create(
            model="qwen3:0.6b", messages=[{"role": "user", "content": prompt}]
        )
        options_text = response.choices[0].message.content.split("\n")
        # (Add simple parsing logic here to extract the 3 strings)
        candidates = [opt.strip() for opt in options_text if opt.strip()]

        # 4. Resolve: Which option aligns with V_next?
        best_score = -1
        best_action = None

        log(f"Evaluating {len(candidates)} action candidates...", self.id)

        for action in candidates:
            # Embed the candidate action
            v_cand = self.embed(action)

            # Cosine Similarity with the Ideal Trajectory
            score = np.dot(v_cand, v_next)

            # Debugging the "Forces"
            log(f"  '{action[:50]}...' → alignment={score:.3f}", self.id)

            if score > best_score:
                best_score = score
                best_action = action

        # 5. Act (Move)
        log(f"✓ Selected: '{best_action}' (score={best_score:.3f})", self.id)
        self.current_task = best_action
