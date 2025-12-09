import numpy as np
from multiprocessing import shared_memory, Lock, resource_tracker


class SharedMemoryStore:
    def __init__(
        self, name="agent_memory_slab", max_items=10000, vector_dim=768, create=False
    ):
        """
        A zero-copy shared memory vector store.

        Args:
            name: Unique name for the memory block.
            max_items: Maximum capacity of vectors (pre-allocated).
            vector_dim: Dimension of embeddings (768 for nomic-embed-text).
            create: Set True for the 'Server/Writer' process, False for 'Agent/Reader'.
        """
        self.name = name
        self.max_items = max_items
        self.vector_dim = vector_dim
        self.lock_name = f"{name}_lock"
        self.meta_name = f"{name}_meta"

        # Calculate bytes needed: Vectors (float32) + Count (int64)
        # We reserve the first 8 bytes (int64) to store the 'current_count' of items
        self.vector_size = max_items * vector_dim * 4  # 4 bytes per float32
        self.total_size = self.vector_size + 8

        if create:
            try:
                # Cleanup old memory if it exists (for dev cycles)
                try:
                    existing = shared_memory.SharedMemory(name=self.name)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass

                self.shm = shared_memory.SharedMemory(
                    create=True, size=self.total_size, name=self.name
                )
                # Initialize count to 0
                self.shm.buf[:8] = np.array([0], dtype=np.int64).tobytes()
                print(
                    f"[Memory] Allocated {self.total_size / 1024 / 1024:.2f} MB shared slab."
                )
            except FileExistsError:
                self.shm = shared_memory.SharedMemory(name=self.name)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name)

        # Create numpy view of the entire buffer
        # 1. Counter View (First 8 bytes)
        self.counter_view = np.ndarray(
            (1,), dtype=np.int64, buffer=self.shm.buf, offset=0
        )

        # 2. Vector View (Rest of the buffer)
        self.vector_view = np.ndarray(
            (max_items, vector_dim), dtype=np.float32, buffer=self.shm.buf, offset=8
        )

        # Simple file-based lock for cross-process write safety
        # (For production systems, use a semaphore, but file lock is robust for local)
        self.lock = Lock()

    def add(self, vector):
        """
        Writes a vector to the shared slab. Thread/Process safe.
        """
        # Normalize vector first (L2 norm) for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        with self.lock:
            idx = self.counter_view[0]
            if idx >= self.max_items:
                raise MemoryError("Shared memory slab is full. Increase max_items.")

            # Write direct to memory
            self.vector_view[idx] = vector.astype(np.float32)
            self.counter_view[0] += 1
            return idx

    def search(self, query_vector, k=5):
        """
        Performs a lock-free Matrix Multiplication (Dot Product) over the shared memory.
        Returns: indices, scores
        """
        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # Get current count (atomic read of int64 is safe enough for this)
        current_count = self.counter_view[0]

        if current_count == 0:
            return [], []

        # Slice only the active memory (Zero-Copy View)
        active_memory = self.vector_view[:current_count]

        # The Magic: Matrix Multiplication (Batch Dot Product)
        # This runs in C via Numpy, bypassing Python loop overhead
        scores = np.dot(active_memory, query_vector)

        # Get top K
        # argpartition is faster than sort for top-k
        k = min(k, current_count)
        top_k_indices = np.argpartition(scores, -k)[-k:]

        # Sort the top k by score descending
        top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])][::-1]
        top_k_scores = scores[top_k_indices]

        return top_k_indices.tolist(), top_k_scores.tolist()

    def close(self):
        self.shm.close()

    def destroy(self):
        """Call this only from the creator process on shutdown"""
        self.shm.close()
        self.shm.unlink()
