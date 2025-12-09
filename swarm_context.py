import time
from multiprocessing import Manager
from agent_memory import SharedMemoryStore
from protocol_constants import Space, SPACE_CONFIG


class SwarmContext:
    def __init__(self, create=False, existing_metadata=None):
        """
        Manages the 3 Stigmergic Vector Spaces of the Vector-Swarm Protocol.
        """
        self.stores = {}
        self.metadata = {}

        # If we are the Creator, we must have been passed a Manager().dict()
        # that already contains the sub-dicts for each space.
        if existing_metadata is None:
            raise ValueError("Must pass a shared dictionary structure to SwarmContext")

        # Initialize the 3 Spaces
        for space in Space:
            config = SPACE_CONFIG[space]

            # 1. Create/Attach to the Vector Slab (Fast Path)
            # Name becomes: swarm_problem, swarm_alignment, swarm_body
            slab_name = f"swarm_{space.value}"
            self.stores[space] = SharedMemoryStore(
                name=slab_name,
                max_items=config["max_items"],
                vector_dim=config["dim"],
                create=create,
            )

            # 2. Attach to the Metadata Dict (Slow Path)
            # existing_metadata must be a dict-of-dicts
            self.metadata[space] = existing_metadata[space.value]

    def stigmatize(self, space: Space, text: str, vector, source="agent"):
        """
        Writes a 'mark' (stigmergy) into a specific vector space.
        """
        store = self.stores[space]
        meta = self.metadata[space]

        # 1. Write Vector
        idx = store.add(vector)

        # 2. Write Metadata
        meta[idx] = {"text": text, "source": source, "timestamp": time.time()}
        return idx

    def perceive(self, space: Space, query_vector, k=3):
        """
        Queries a specific vector space to sense the environment/rules/self.
        """
        store = self.stores[space]
        meta = self.metadata[space]

        indices, scores = store.search(query_vector, k=k)

        results = []
        for i, idx in enumerate(indices):
            # Safe retrieval
            data = meta.get(idx, {"text": "VOID", "source": "VOID"})
            results.append(
                {
                    "space": space.value,
                    "text": data["text"],
                    "source": data["source"],
                    "score": float(scores[i]),
                    "id": int(idx),
                }
            )

        return results

    def cleanup(self):
        for store in self.stores.values():
            store.close()

    def destroy(self):
        """Total annihilation of the shared memory blocks"""
        for store in self.stores.values():
            store.destroy()
