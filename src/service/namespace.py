import os
import logging
import json
import shutil
from datetime import datetime, timezone
from typing import List, Optional
from src.logger import logger

class NamespaceService:
    """Manages namespaces on the filesystem."""

    def __init__(self):
        ns_root = os.getenv("EASEVOICE_TRAINER_NAMESPACES_ROOT", os.path.join(os.getcwd(), "easevoice_trainer_namespaces"))

        # Fixed path to the metadata file for the namespaces root.
        self.ns_root_metadata_path = os.path.join(os.getcwd(), ".namespaces_root.metadata.json")

        # Create the default namespace root anyway in the beginning.
        self.ns_root = ns_root
        os.makedirs(self.ns_root, exist_ok=True)

        # The first time saving does not count as a set-once operation.
        self._save_ns_root_metadata(setOnce=False)

    def _save_ns_root_metadata(self, setOnce: bool):
        with open(self.ns_root_metadata_path, "w") as f:
            json.dump({
                "namespaces-root": self.ns_root,
                "setOnce": setOnce,
            }, f)

    def get_namespaces_root_metadata(self) -> str:
        with open(self.ns_root_metadata_path, "r") as f:
            return json.load(f)

    def set_namespaces_root(self, ns_root: str):
        if os.path.exists(self.ns_root_metadata_path):
            with open(self.ns_root_metadata_path, "r") as f:
                metadata = json.load(f)
                if metadata["setOnce"]:
                    logging.warning("change namespaces root %s to %s while setOnce is true", self.ns_root, ns_root)

        self.ns_root = ns_root
        os.makedirs(self.ns_root, exist_ok=True)

        self._save_ns_root_metadata(setOnce=True)

    def _namespace_metadata_path(self, name: str) -> str:
        return os.path.join(self.ns_root, name, ".metadata.json")

    def create_namespace(self, name: str) -> dict:
        """Create a new namespace, raising FileExistsError if it already exists."""
        home_path = os.path.join(self.ns_root, name)
        if os.path.exists(home_path):
            raise FileExistsError("Namespace already exists")

        # Prepare the namespace directory structure.
        os.makedirs(os.path.join(home_path, "voices"), exist_ok=True)
        os.makedirs(os.path.join(home_path, "outputs"), exist_ok=True)
        os.makedirs(os.path.join(home_path, "training-audios"), exist_ok=True)
        os.makedirs(os.path.join(home_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(home_path, "models", "sovits_train"), exist_ok=True)
        os.makedirs(os.path.join(home_path, "models", "gpt_train"), exist_ok=True)

        namespace = {"name": name, "createdAt": int(datetime.now(tz=timezone.utc).timestamp() * 1000), "homePath": home_path}
        self._save_namespace_metadata(namespace)
        return namespace

    def get_namespaces(self) -> List[dict]:
        """List all namespaces."""
        namespaces = []
        for name in os.listdir(self.ns_root):
            namespace_path = os.path.join(self.ns_root, name)
            if os.path.isdir(namespace_path):
                try:
                    namespaces.append(self._load_namespace_metadata(name))
                except FileNotFoundError as e:
                    logger.warning(f"Namespace {name} metadata not found: {e}", exc_info=True)
        return namespaces

    def update_namespace(self, old_name: str, new_name: str) -> dict:
        """Rename a namespace, raising FileExistsError if the new name is taken."""
        old_home_path = os.path.join(self.ns_root, old_name)
        new_home_path = os.path.join(self.ns_root, new_name)

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
        home_path = os.path.join(self.ns_root, name)
        if not os.path.exists(home_path):
            raise ValueError("Namespace not found")

        shutil.rmtree(home_path)

    def _save_namespace_metadata(self, namespace: dict):
        with open(self._namespace_metadata_path(namespace["name"]), "w") as f:
            json.dump(namespace, f)

    def _load_namespace_metadata(self, name: str) -> dict:
        metadata_path = self._namespace_metadata_path(name)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Namespace metadata in {metadata_path} not found")
        with open(metadata_path, "r") as f:
            return json.load(f)