import os
import uuid
import json
from datetime import datetime, timezone
from typing import List, Optional
from api.api import Task


class TaskService:
    """Service class for managing tasks using the filesystem."""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the TaskService.

        Args:
            base_dir (str): Base directory for task storage. Defaults to `$HOME/easevoice_trainer_tasks`.
        """
        if base_dir is None:
            home_dir = os.path.expanduser("~")
            base_dir = os.path.join(home_dir, "easevoice_trainer_tasks")
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _task_metadata_path(self, task_id: str) -> str:
        """Get the path to the task metadata file."""
        return os.path.join(self.base_dir, task_id, ".metadata.json")

    def create_task(self) -> Task:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        task_name = f"Task-{task_id[:8]}"
        created_at = datetime.now(timezone.utc)
        home_path = os.path.join(self.base_dir, task_id)
        os.makedirs(home_path, exist_ok=True)

        task = Task(
            taskID=task_id,
            name=task_name,
            createdAt=created_at,
            homePath=home_path,
        )
        self._save_task_metadata(task)
        return task

    def get_tasks(self) -> List[Task]:
        """Get all tasks."""
        tasks = []
        for task_id in os.listdir(self.base_dir):
            task_path = os.path.join(self.base_dir, task_id)
            if os.path.isdir(task_path):
                try:
                    task = self._load_task_metadata(task_id)
                    tasks.append(task)
                except FileNotFoundError:
                    pass  # Skip invalid tasks
        return tasks

    def update_task(self, task_id: str, name: str) -> Task:
        """Update an existing task."""
        task = self._load_task_metadata(task_id)
        task.name = name
        self._save_task_metadata(task)
        return task

    def delete_task(self, task_id: str):
        """Delete a task."""
        task = self._load_task_metadata(task_id)
        self._delete_directory(task.homePath)

    def _save_task_metadata(self, task: Task):
        """Save task metadata to a file."""
        metadata_path = self._task_metadata_path(task.taskID)
        with open(metadata_path, "w") as f:
            json.dump(task.dict(), f, default=str)

    def _load_task_metadata(self, task_id: str) -> Task:
        """Load task metadata from a file."""
        metadata_path = self._task_metadata_path(task_id)
        if not os.path.exists(metadata_path):
            raise ValueError(f"Task with ID {task_id} not found.")
        with open(metadata_path, "r") as f:
            data = json.load(f)
        data["createdAt"] = datetime.fromisoformat(data["createdAt"])
        return Task(**data)

    def _delete_directory(self, directory_path: str):
        """Delete a directory and its contents."""
        if os.path.exists(directory_path):
            for root, dirs, files in os.walk(directory_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(directory_path)
