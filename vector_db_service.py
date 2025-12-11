"""
Vector Database Service
Provides a high-level interface for interacting with Qdrant vector database.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import asyncio
from queue import Queue
from threading import Thread

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, Range, MatchValue,
    SearchParams
)

from vector_db_config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_GRPC_PORT, QDRANT_TIMEOUT,
    QDRANT_BATCH_SIZE, QDRANT_ASYNC_QUEUE_SIZE, QDRANT_QUERY_EF,
    QDRANT_RUN_RETENTION_DAYS, QDRANT_DELETE_ON_COMPLETE,
    get_per_run_collection_configs, get_global_collection_configs,
    get_historical_collection_name, get_current_collection_name,
    get_mission_collection_name, MISSIONS_COLLECTION, SNAPSHOTS_COLLECTION
)


logger = logging.getLogger(__name__)


@dataclass
class VectorPoint:
    """Represents a vector with its metadata."""
    id: str
    vector: List[float]
    payload: Dict[str, Any]


class VectorDBService:
    """
    High-level service for vector database operations.
    Wraps Qdrant client and provides swarm-specific functionality.
    """

    def __init__(self, use_grpc: bool = True):
        """
        Initialize the Vector DB service.

        Args:
            use_grpc: If True, use gRPC for lower latency (recommended)
        """
        self.use_grpc = use_grpc

        # Initialize Qdrant client
        if use_grpc:
            self.client = QdrantClient(
                host=QDRANT_HOST,
                grpc_port=QDRANT_GRPC_PORT,
                timeout=QDRANT_TIMEOUT,
                prefer_grpc=True
            )
            logger.info(f"Connected to Qdrant via gRPC at {QDRANT_HOST}:{QDRANT_GRPC_PORT}")
        else:
            self.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                timeout=QDRANT_TIMEOUT
            )
            logger.info(f"Connected to Qdrant via HTTP at {QDRANT_HOST}:{QDRANT_PORT}")

        # Async write queue (for non-blocking writes)
        self.write_queue: Queue = Queue(maxsize=QDRANT_ASYNC_QUEUE_SIZE)
        self.write_thread: Optional[Thread] = None
        self.stop_writer = False

    def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    # Collection Management

    def create_collection(self, collection_name: str, vector_params, hnsw_config) -> bool:
        """
        Create a collection if it doesn't exist.

        Args:
            collection_name: Name of the collection
            vector_params: Vector parameters (dimension, distance)
            hnsw_config: HNSW index configuration

        Returns:
            True if created or already exists, False on error
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                logger.debug(f"Collection {collection_name} already exists")
                return True

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_params,
                hnsw_config=hnsw_config,
                on_disk_payload=False  # Keep in memory for fast access
            )
            logger.info(f"Created collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False

    def initialize_global_collections(self) -> bool:
        """Initialize global collections (missions, snapshots)."""
        global_configs = get_global_collection_configs()

        for key, config in global_configs.items():
            success = self.create_collection(
                config.name,
                config.vector_params,
                config.hnsw_config
            )
            if not success:
                logger.error(f"Failed to initialize global collection: {config.name}")
                return False

        logger.info("Global collections initialized successfully")
        return True

    def initialize_run_collections(self, run_id: str) -> bool:
        """
        Initialize per-run collections for a swarm run.

        Args:
            run_id: Unique identifier for the swarm run

        Returns:
            True if all collections created successfully
        """
        run_configs = get_per_run_collection_configs(run_id)

        for key, config in run_configs.items():
            success = self.create_collection(
                config.name,
                config.vector_params,
                config.hnsw_config
            )
            if not success:
                logger.error(f"Failed to initialize run collection: {config.name}")
                return False

        logger.info(f"Run collections initialized for run_id: {run_id}")
        return True

    def cleanup_run_collections(self, run_id: str) -> bool:
        """
        Delete per-run collections after run completion.

        Args:
            run_id: Unique identifier for the swarm run

        Returns:
            True if all collections deleted successfully
        """
        collections_to_delete = [
            get_historical_collection_name(run_id),
            get_current_collection_name(run_id),
            get_mission_collection_name(run_id)
        ]

        all_success = True
        for collection in collections_to_delete:
            if not self.delete_collection(collection):
                all_success = False

        return all_success

    # Write Operations

    def upsert_vector(
        self,
        collection_name: str,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any]
    ) -> bool:
        """
        Upsert a vector (insert or update if exists).
        Used for current actions where each agent has one point.

        Args:
            collection_name: Target collection
            point_id: Unique ID (typically agent_id for current actions)
            vector: Embedding vector
            payload: Metadata

        Returns:
            True if successful
        """
        try:
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert vector to {collection_name}: {e}")
            return False

    def append_vector(
        self,
        collection_name: str,
        vector: List[float],
        payload: Dict[str, Any],
        point_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Append a new vector (always creates new point).
        Used for historical actions.

        Args:
            collection_name: Target collection
            vector: Embedding vector
            payload: Metadata
            point_id: Optional custom ID (generates UUID if not provided)

        Returns:
            Point ID if successful, None on error
        """
        try:
            if point_id is None:
                point_id = str(uuid.uuid4())

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            return point_id
        except Exception as e:
            logger.error(f"Failed to append vector to {collection_name}: {e}")
            return None

    def batch_append_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Batch append multiple vectors for efficiency.

        Args:
            collection_name: Target collection
            vectors: List of embedding vectors
            payloads: List of metadata dicts

        Returns:
            List of point IDs
        """
        try:
            points = []
            point_ids = []

            for vector, payload in zip(vectors, payloads):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))

            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return point_ids
        except Exception as e:
            logger.error(f"Failed to batch append to {collection_name}: {e}")
            return []

    # Query Operations

    def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Filter] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.

        Args:
            collection_name: Collection to search
            query_vector: Query embedding
            limit: Max results to return
            score_threshold: Minimum similarity score
            filter_conditions: Optional filters on payload

        Returns:
            List of (point_id, score, payload) tuples
        """
        try:
            search_params = SearchParams(
                hnsw_ef=QDRANT_QUERY_EF,
                exact=False  # Use HNSW approximation for speed
            )

            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
                search_params=search_params
            )

            return [
                (str(hit.id), hit.score, hit.payload)
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Failed to search {collection_name}: {e}")
            return []

    def get_by_id(
        self,
        collection_name: str,
        point_id: str
    ) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Retrieve a specific vector by ID.

        Args:
            collection_name: Collection to query
            point_id: Point ID to retrieve

        Returns:
            (vector, payload) tuple if found, None otherwise
        """
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_vectors=True,
                with_payload=True
            )

            if not points:
                return None

            point = points[0]
            return (point.vector, point.payload)
        except Exception as e:
            logger.error(f"Failed to get point {point_id} from {collection_name}: {e}")
            return None

    def get_all_from_collection(
        self,
        collection_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """
        Retrieve all vectors from a collection (paginated).

        Args:
            collection_name: Collection to query
            limit: Max results per page
            offset: Number of results to skip

        Returns:
            List of (point_id, vector, payload) tuples
        """
        try:
            # Scroll through collection
            records, _ = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_vectors=True,
                with_payload=True
            )

            return [
                (str(record.id), record.vector, record.payload)
                for record in records
            ]
        except Exception as e:
            logger.error(f"Failed to retrieve from {collection_name}: {e}")
            return []

    # Cleanup Operations

    def cleanup_old_runs(self, retention_days: int = QDRANT_RUN_RETENTION_DAYS) -> int:
        """
        Delete per-run collections older than retention period.

        Args:
            retention_days: Number of days to keep per-run collections

        Returns:
            Number of collections deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            collections = self.client.get_collections().collections
            deleted_count = 0

            for collection in collections:
                # Check if it's a per-run collection
                if collection.name.startswith("run_"):
                    # Get a sample point to check timestamp
                    try:
                        records, _ = self.client.scroll(
                            collection_name=collection.name,
                            limit=1,
                            with_payload=True
                        )

                        if records:
                            timestamp_str = records[0].payload.get("timestamp")
                            if timestamp_str:
                                timestamp = datetime.fromisoformat(timestamp_str)
                                if timestamp < cutoff_date:
                                    self.delete_collection(collection.name)
                                    deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Could not check timestamp for {collection.name}: {e}")

            logger.info(f"Cleaned up {deleted_count} old run collections")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {e}")
            return 0

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections."""
        try:
            collections = self.client.get_collections().collections
            stats = {}

            for collection in collections:
                info = self.client.get_collection(collection.name)
                stats[collection.name] = {
                    "points_count": info.points_count,
                    "segments_count": info.segments_count,
                    "status": info.status
                }

            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    def close(self):
        """Close the Qdrant client connection."""
        # Stop async writer if running
        if self.write_thread and self.write_thread.is_alive():
            self.stop_writer = True
            self.write_thread.join(timeout=5)

        logger.info("Vector DB service closed")
