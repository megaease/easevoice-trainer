import os
import uuid
import json
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
from src.api.api import Namespace, Progress


class NamespaceService:
    """Service class for managing namespaces using the filesystem."""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the NamespaceService.

        Args:
            base_dir (str): Base directory for namespace storage. Defaults to `$HOME/easevoice_trainer_namespaces`.
        """
        if base_dir is None:
            home_dir = os.path.expanduser("~")
            base_dir = os.path.join(home_dir, "easevoice_trainer_namespaces")
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        namespaces = self.get_namespaces()
        self._namespaces: Dict[str, Namespace] = {}
        for namespace in namespaces:
            self._namespaces[namespace.namespaceID] = namespace

    def _namespace_metadata_path(self, namespace_id: str) -> str:
        """Get the path to the namespace metadata file."""
        return os.path.join(self.base_dir, namespace_id, ".metadata.json")

    def filter_namespaces(self, fn: Callable[[Namespace], bool]) -> List[Namespace]:
        """
        Filter namespace using a function.
        For example, to get all pending namespaces for a service.
        """
        return sorted(list(filter(fn, self._namespaces.values())), key=lambda t: t.createdAt)

    def create_namespace(self) -> Namespace:
        """Create a new namespace."""
        namespace_id = str(uuid.uuid4())
        namespace_name = f"Namespace-{namespace_id[:8]}"
        created_at = datetime.now(timezone.utc)
        home_path = os.path.join(self.base_dir, namespace_id)
        os.makedirs(home_path, exist_ok=True)
        os.makedirs(os.path.join(home_path, "voices"), exist_ok=True)

        namespace = Namespace(
            namespaceID=namespace_id,
            name=namespace_name,
            createdAt=created_at,
            homePath=home_path,
        )
        self._namespaces[namespace_id] = namespace

        self._save_namespace_metadata(namespace)

        return namespace

    def get_namespaces(self) -> List[Namespace]:
        """Get all namespaces."""
        namespaces = []
        for namespace_id in os.listdir(self.base_dir):
            namespace_path = os.path.join(self.base_dir, namespace_id)
            if os.path.isdir(namespace_path):
                try:
                    namespace = self._load_namespace_metadata(namespace_id)
                    namespaces.append(namespace)
                    self._namespaces[namespace_id] = namespace
                except FileNotFoundError:
                    pass  # Skip invalid namespaces
        return namespaces

    def update_namespace(self, namespace_id: str, name: str) -> Namespace:
        """Update an existing namespace."""
        namespace = self._load_namespace_metadata(namespace_id)
        namespace.name = name
        self._save_namespace_metadata(namespace)
        self._namespaces[namespace.namespaceID] = namespace
        return namespace

    def delete_namespace(self, namespace_id: str):
        """Delete a namespace."""
        namespace = self._load_namespace_metadata(namespace_id)
        self._namespaces.pop(namespace_id)
        self._delete_directory(namespace.homePath)

    def _save_namespace_metadata(self, namespace: Namespace):
        """Save namespace metadata to a file."""
        metadata_path = self._namespace_metadata_path(namespace.namespaceID)
        with open(metadata_path, "w") as f:
            json.dump(namespace.model_dump(), f, default=str)

    def _load_namespace_metadata(self, namespace_id: str) -> Namespace:
        """Load namespace metadata from a file."""
        metadata_path = self._namespace_metadata_path(namespace_id)
        if not os.path.exists(metadata_path):
            raise ValueError(f"Namespace with ID {namespace_id} not found.")
        with open(metadata_path, "r") as f:
            data = json.load(f)
        data["createdAt"] = datetime.fromisoformat(data["createdAt"])
        return Namespace(**data)

    def _delete_directory(self, directory_path: str):
        """Delete a directory and its contents."""
        if os.path.exists(directory_path):
            for root, dirs, files in os.walk(directory_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(directory_path)
