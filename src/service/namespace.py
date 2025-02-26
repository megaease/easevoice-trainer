import os
import json
import shutil
from datetime import datetime, timezone
from typing import List, Optional

class NamespaceService:
    """Manages namespaces on the filesystem."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(os.path.expanduser("~"), "easevoice_trainer_namespaces")
        os.makedirs(self.base_dir, exist_ok=True)

    def _namespace_metadata_path(self, name: str) -> str:
        return os.path.join(self.base_dir, name, ".metadata.json")

    def create_namespace(self, name: str) -> dict:
        """Create a new namespace, raising FileExistsError if it already exists."""
        home_path = os.path.join(self.base_dir, name)
        if os.path.exists(home_path):
            raise FileExistsError("Namespace already exists")

        os.makedirs(os.path.join(home_path, "voices"), exist_ok=True)
        os.makedirs(os.path.join(home_path, "outputs"), exist_ok=True)
        namespace = {"name": name, "createdAt": int(datetime.now(tz=timezone.utc).timestamp() * 1000), "homePath": home_path}
        self._save_namespace_metadata(namespace)
        return namespace

    def get_namespaces(self) -> List[dict]:
        """List all namespaces."""
        namespaces = []
        for name in os.listdir(self.base_dir):
            namespace_path = os.path.join(self.base_dir, name)
            if os.path.isdir(namespace_path):
                try:
                    namespaces.append(self._load_namespace_metadata(name))
                except FileNotFoundError:
                    pass
        return namespaces

    def update_namespace(self, old_name: str, new_name: str) -> dict:
        """Rename a namespace, raising FileExistsError if the new name is taken."""
        old_home_path = os.path.join(self.base_dir, old_name)
        new_home_path = os.path.join(self.base_dir, new_name)

        if not os.path.exists(old_home_path):
            raise ValueError("Namespace not found")

        if os.path.exists(new_home_path):
            raise FileExistsError("Target namespace already exists")

        namespace = self._load_namespace_metadata(old_name)

        os.rename(old_home_path, new_home_path)

        namespace["name"] = new_name
        namespace["homePath"] = new_home_path
        self._save_namespace_metadata(namespace)
        return namespace

    def delete_namespace(self, name: str):
        """Delete a namespace, raising ValueError if it does not exist."""
        home_path = os.path.join(self.base_dir, name)
        if not os.path.exists(home_path):
            raise ValueError("Namespace not found")

        shutil.rmtree(home_path)

    def _save_namespace_metadata(self, namespace: dict):
        with open(self._namespace_metadata_path(namespace["name"]), "w") as f:
            json.dump(namespace, f)

    def _load_namespace_metadata(self, name: str) -> dict:
        metadata_path = self._namespace_metadata_path(name)
        if not os.path.exists(metadata_path):
            raise ValueError("Namespace not found")
        with open(metadata_path, "r") as f:
            return json.load(f)