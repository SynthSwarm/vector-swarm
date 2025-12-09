import time
from multiprocessing import Manager
from agent_memory import SharedMemoryStore  # Importing your previous class


class AgentContext:
    def __init__(
        self,
        memory_name="swarm_memory",
        max_items=10000,
        create=False,
        existing_metadata=None,
    ):
        """
        A unified context for agents to access shared vector memory and text metadata.

        Args:
            memory_name: ID for the shared memory block.
            create: True if this is the Main Orchestrator, False for Agents.
            existing_metadata: Must pass the Manager().dict() here if create=False.
        """
        # 1. Initialize the Fast Vector Store (The NumPy Slab)
        self.vector_store = SharedMemoryStore(
            name=memory_name, max_items=max_items, create=create
        )

        # 2. Initialize the Metadata Store (The Dictionary)
        # If we are the creator, we might create the Manager, but usually
        # the Manager must be created in the `if __name__ == "__main__":` block
        # to work correctly on all OSs. So we expect it to be passed in.
        if existing_metadata is None:
            raise ValueError(
                "Must pass a shared dictionary (from Manager) to AgentContext"
            )

        self.metadata = existing_metadata

    def save_memory(self, text, vector, source="agent"):
        """
        Saves a thought/log to the hive mind.
        """
        # 1. Write Vector to Shared Memory (Fast, returns integer index)
        # Note: vector_store.add is thread-safe via its internal lock
        idx = self.vector_store.add(vector)

        # 2. Write Text to Shared Dict (Slower, but only happens once)
        # We store a lightweight payload
        self.metadata[idx] = {"text": text, "source": source, "timestamp": time.time()}
        return idx

    def recall_memory(self, query_vector, k=3):
        """
        Semantic search over the shared hive mind.
        """
        # 1. Search Vector Space (Zero-Copy Math)
        indices, scores = self.vector_store.search(query_vector, k=k)

        results = []
        for i, idx in enumerate(indices):
            # 2. Retrieve Text from Shared Dict
            # Handle case where metadata might be lagging/missing (rare race condition)
            meta = self.metadata.get(
                idx, {"text": "[Memory Corruption]", "source": "system"}
            )

            results.append(
                {
                    "text": meta["text"],
                    "source": meta["source"],
                    "score": float(scores[i]),  # Convert numpy float to native float
                    "id": int(idx),
                }
            )

        return results

    def cleanup(self):
        """Close local handles"""
        self.vector_store.close()
