"""
Embedding Service - Abstraction layer for text embeddings

Provides a clean interface for embedding text that can work with different
backends (FastEmbed, OpenAI, etc.) and handles multiprocessing properly.

Task Type Prefixes (for nomic-embed models):
-------------------------------------------
The nomic-embed models require task-specific prefixes for optimal performance:

- "search_document": Use when STORING/INDEXING data (e.g., agent observations, documents)
                    This is the default and appropriate for most storage operations.

- "search_query": Use when SEARCHING for relevant documents (e.g., user queries, search terms)
                 Use this when creating query vectors for search_historical() or similar functions.

- "clustering": Use when GROUPING texts by semantic similarity
               Appropriate for finding related items or topic clustering.

- "classification": Use when extracting features for classification tasks
                   Useful when embeddings will be used as input to classifiers.

For more details, see: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#usage
"""

import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from queue import Empty
import time
from datetime import datetime


def log(message, level="EMBED"):
    """Timestamped logging helper"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}", flush=True)


# --- Abstract Base Class ---
class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends"""

    @abstractmethod
    def embed(self, text: str, task_type: str = "search_document") -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Text to embed
            task_type: Task instruction prefix for embedding models that support it.
                      Options: "search_document", "search_query", "clustering", "classification"
                      Default: "search_document" (for indexing/storing data)

        Returns:
            numpy array of embeddings
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension"""
        pass


# --- FastEmbed Backend ---
class FastEmbedBackend(EmbeddingBackend):
    """FastEmbed implementation (CPU-based, ONNX)"""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        log(f"Loading FastEmbed model: {model_name}")
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)
        self.dimension = 768  # nomic-embed-text-v1.5 dimension
        log("✓ FastEmbed model loaded")

    def embed(self, text: str, task_type: str = "search_document") -> np.ndarray:
        """
        Embed text using FastEmbed with task instruction prefix.

        For nomic-embed models, prepends task_type prefix (e.g., "search_document: <text>")
        as recommended in the model documentation.
        """
        # Prepend task prefix for nomic models
        prefixed_text = f"{task_type}: {text}"
        prefixed_text = prefixed_text.replace("\n", " ")

        try:
            embeddings = list(self.model.embed([prefixed_text]))
            return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            log(f"FastEmbed error: {e}", "ERROR")
            return np.zeros(self.dimension, dtype=np.float32)

    def get_dimension(self) -> int:
        return self.dimension


# --- OpenAI Backend (for comparison) ---
class OpenAIEmbedBackend(EmbeddingBackend):
    """OpenAI embeddings implementation"""

    def __init__(self, client, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model
        self.dimension = 1536  # text-embedding-3-small dimension
        log(f"Using OpenAI embeddings: {model}")

    def embed(self, text: str, task_type: str = "search_document") -> np.ndarray:
        """
        Embed text using OpenAI API.

        Note: task_type parameter is accepted for interface compatibility but not used,
        as OpenAI embeddings don't require task instruction prefixes.
        """
        text = text.replace("\n", " ")
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            log(f"OpenAI embedding error: {e}", "ERROR")
            return np.zeros(self.dimension, dtype=np.float32)

    def get_dimension(self) -> int:
        return self.dimension


# --- Multiprocessing Embedding Service ---
class EmbeddingService:
    """
    Multiprocessing-safe embedding service.

    Runs the embedding backend in a dedicated process and provides
    a queue-based request/response interface for other processes.
    """

    def __init__(self, backend: EmbeddingBackend):
        """
        Initialize the embedding service.

        Args:
            backend: EmbeddingBackend instance to use
        """
        self.backend = backend
        self.dimension = backend.get_dimension()

        # Multiprocessing queues for request/response
        self.request_queue = Queue()
        self.response_queue = Queue()  # Single response queue

        # Start the worker process
        self.worker_process = None
        self.start_worker()

    def start_worker(self):
        """Start the background worker process"""
        self.worker_process = Process(
            target=self._worker_loop,
            args=(self.backend, self.request_queue, self.response_queue),
            daemon=True
        )
        self.worker_process.start()
        log("Embedding service worker started")

    @staticmethod
    def _worker_loop(backend, request_queue, response_queue):
        """
        Worker process loop - processes embedding requests.

        This runs in a separate process and handles all embedding calls.
        """
        log("Embedding worker process started", "WORKER")

        while True:
            try:
                # Get next request (blocking)
                request = request_queue.get(timeout=1.0)

                request_id = request['id']
                text = request['text']
                task_type = request.get('task_type', 'search_document')

                # Perform embedding with task type
                embedding = backend.embed(text, task_type=task_type)

                # Send response back
                response = {
                    'id': request_id,
                    'embedding': embedding
                }

                response_queue.put(response)

            except Empty:
                continue
            except Exception as e:
                log(f"Worker error: {e}", "ERROR")
                continue

    def embed(self, text: str, task_type: str = "search_document", timeout: float = 30.0) -> np.ndarray:
        """
        Embed text (blocks until result is ready).

        Args:
            text: Text to embed
            task_type: Task instruction prefix ("search_document", "search_query", "clustering", "classification")
            timeout: Maximum time to wait for response

        Returns:
            numpy array of embeddings
        """
        # Generate unique request ID
        request_id = f"{time.time()}_{id(text)}"

        # Send request
        request = {
            'id': request_id,
            'text': text,
            'task_type': task_type
        }
        self.request_queue.put(request)

        # Wait for response with our request ID
        start_time = time.time()
        while True:
            try:
                # Try to get a response (non-blocking with short timeout)
                response = self.response_queue.get(timeout=0.1)

                # Check if this is our response
                if response['id'] == request_id:
                    return response['embedding']
                else:
                    # Not our response, put it back
                    self.response_queue.put(response)
                    time.sleep(0.01)  # Small delay to prevent busy-waiting

            except Empty:
                # Check timeout
                if time.time() - start_time > timeout:
                    log(f"Embedding timeout after {timeout}s", "ERROR")
                    return np.zeros(self.dimension, dtype=np.float32)
                continue

    def shutdown(self):
        """Shutdown the worker process"""
        if self.worker_process and self.worker_process.is_alive():
            self.worker_process.terminate()
            self.worker_process.join(timeout=5)
            log("Embedding service shutdown")


# --- Factory Function ---
def create_embedding_service(backend_type: str = "fastembed", **kwargs) -> EmbeddingService:
    """
    Factory function to create an embedding service with the specified backend.

    Args:
        backend_type: Type of backend ("fastembed", "openai")
        **kwargs: Backend-specific arguments

    Returns:
        EmbeddingService instance
    """
    if backend_type == "fastembed":
        model_name = kwargs.get("model_name", "nomic-ai/nomic-embed-text-v1.5")
        backend = FastEmbedBackend(model_name=model_name)

    elif backend_type == "openai":
        client = kwargs.get("client")
        if not client:
            raise ValueError("OpenAI backend requires 'client' argument")
        model = kwargs.get("model", "text-embedding-3-small")
        backend = OpenAIEmbedBackend(client=client, model=model)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    return EmbeddingService(backend)
