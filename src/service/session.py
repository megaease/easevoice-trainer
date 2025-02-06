import threading
from functools import wraps
from typing import Optional, Dict, Any
from enum import Enum

class Status(Enum):
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"

class SessionManager:
    """Manages training session, ensuring single GPU task execution and tracking task state."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one instance of SessionManager exists."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(SessionManager, cls).__new__(cls)
                    cls._instance.current_session = None
        return cls._instance

    def start_session(self, task_name: str):
        """Attempts to start a new session; rejects if another task is already running."""
        if self.current_session and self.current_session["status"] == Status.RUNNING:
            raise RuntimeError(f"A task '{self.current_session['task_name']}' is already running. Cannot submit another task!")

        self.current_session = {
            "task_name": task_name,
            "status": Status.RUNNING,
            "error": None,  # Stores error details if task fails
        }

    def end_session(self, result: Any):
        """Marks task as completed successfully."""
        if self.current_session:
            self.current_session["status"] = Status.COMPLETED
            self.current_session["result"] = result

    def fail_session(self, error: str):
        """Marks task as failed and stores error information."""
        if self.current_session:
            self.current_session["status"] = Status.FAILED
            self.current_session["error"] = error

    def update_session_info(self, info: Dict[str, Any]):
        """Updates task session with arbitrary info."""
        if not self.current_session or self.current_session["status"] != Status.RUNNING:
            raise RuntimeError("No active task to update session info!")

        self.current_session.update(info)
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Returns current task state information."""
        return self.current_session

# Decorator to wrap task execution logic
def session_guard(task_name: str):
    """Ensures tasks are managed within SessionManager and handles failure states."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            session_manager = SessionManager()

            try:
                session_manager.start_session(task_name)
                result = func(*args, **kwargs)  # Execute the training task
                session_manager.end_session(result)
                return result
            except Exception as e:
                session_manager.fail_session(str(e))  # Record failure details
                # NOTICE: No Re-raise exception here,
                # as we capture the error and record it in session info.
                # raise e
        return wrapper
    return decorator

# Example for using SessionManager and session_guard decorator.
@session_guard("TrainingModel")
def train_model():
    session_manager = SessionManager()

    for epoch in range(1, 6):
        if epoch == 3:  # Simulate task failure
            raise RuntimeError("Error occurred at epoch 3!")

        session_manager.update_session_info({
            "progress": epoch / 5,
            "loss": 0.05 * (6 - epoch),
            "epoch": epoch,
        })

    return "Training Completed"
