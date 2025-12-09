"""
Logged LLM Client Wrapper
Intercepts OpenAI client calls and logs them to a multiprocessing Queue.
"""
from datetime import datetime
from openai import OpenAI


class LoggedLLMClient:
    """Wrapper around OpenAI client that logs all interactions to a queue."""

    def __init__(self, base_client: OpenAI, agent_id: int, log_queue):
        """
        Args:
            base_client: The actual OpenAI client instance
            agent_id: ID of the agent using this client
            log_queue: multiprocessing.Queue to send log entries to
        """
        self.client = base_client
        self.agent_id = agent_id
        self.log_queue = log_queue

    def chat_completion(self, model: str, messages: list, **kwargs):
        """
        Wrapped chat completion that logs the interaction.

        Args:
            model: Model name (e.g., "qwen3:0.6b")
            messages: List of message dicts
            **kwargs: Additional arguments to pass to the client

        Returns:
            The response from the LLM
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Extract the prompt (assumes last message is user prompt)
        prompt = messages[-1]["content"] if messages else ""

        # Call the actual LLM
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

        # Extract response text
        response_text = response.choices[0].message.content

        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "prompt": prompt,
            "response": response_text,
            "model": model
        }

        # Send to queue (non-blocking, will be drained by server)
        try:
            self.log_queue.put_nowait(log_entry)
        except Exception:
            # If queue is full, just skip logging this entry
            pass

        return response
