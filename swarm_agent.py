# swarm_agent.py
import numpy as np
from datetime import datetime
from swarm_physics import SwarmPhysics


def log(message, agent_id):
    """Timestamped logging for agents"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [Agent-{agent_id}] {message}", flush=True)


class VectorAgent:
    def __init__(
        self,
        agent_id,
        physics_engine: SwarmPhysics,
        llm_client,
        embed_func,
        starting_task="Idle",
        log_queue=None,
    ):
        self.id = agent_id
        self.physics = physics_engine
        self.llm = llm_client
        self.embed = embed_func
        self.current_task = starting_task
        self.log_queue = log_queue  # For detailed selection logging

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
        Examples: 1. [Analyze recent data logs]
                  2. [Communicate with nearby agents]

        IMPORTANT: 
        * Do not add newlines in an action
        * Separate each action with a newline
        """
        # Check if we're using LoggedLLMClient or raw OpenAI client
        if hasattr(self.llm, "chat_completion"):
            # Using LoggedLLMClient wrapper
            response = self.llm.chat_completion(
                model="Qwen/Qwen3-0.6B", messages=[{"role": "user", "content": prompt}]
            )
        else:
            # Using raw OpenAI client (fallback for compatibility)
            response = self.llm.chat.completions.create(
                model="Qwen/Qwen3-0.6B", messages=[{"role": "user", "content": prompt}]
            )
        full_response = response.choices[0].message.content

        # Extract thinking content (between <think> and </think>)
        thinking_content = None
        response_without_thinking = full_response
        if "<think>" in full_response and "</think>" in full_response:
            start = full_response.find("<think>")
            end = full_response.find("</think>") + 8  # Include </think> tag
            thinking_content = full_response[start+7:end-8].strip()  # Extract just the content
            # Remove the entire <think>...</think> block from response
            response_without_thinking = full_response[:start] + full_response[end:]

        options_text = response_without_thinking.split("\n")
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

        # 6. Log Selection Details (if queue available)
        if self.log_queue:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            selection_log = {
                "timestamp": timestamp,
                "agent_id": self.id,
                "selected_action": best_action,
                "alignment_score": float(best_score),
                "candidates": candidates[
                    :5
                ],  # Log first 5 candidates to avoid huge entries
                "thinking": thinking_content,  # Include LLM's reasoning process
                "type": "selection",  # Mark this as a selection log (vs. raw LLM log)
            }
            try:
                self.log_queue.put_nowait(selection_log)
            except Exception:
                pass  # Queue full, skip logging
