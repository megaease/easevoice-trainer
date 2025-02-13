from multiprocess import Process, Queue
import os
import signal
import threading
import traceback
from enum import Enum
from functools import wraps
from typing import Optional, Dict, Any

import psutil

from src.utils.response import EaseVoiceResponse, ResponseStatus


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
                    cls._instance._session_lock = threading.Lock()  # Protects session state
                    cls._instance.current_session = None
                    cls._instance.last_session = None
        return cls._instance

    def start_session(self, task_name: str):
        """Attempts to start a new session; rejects if another task is already running."""
        with self._session_lock:
            if self.current_session is not None:
                raise RuntimeError(
                    f"A task '{self.current_session['task_name']}' is already running. Cannot submit another task!"
                )

            self.current_session = {
                "task_name": task_name,
                "status": Status.RUNNING,
                "error": None,  # Stores error details if task fails
            }

    def end_session(self, result: Any):
        """Marks task as completed successfully."""
        with self._session_lock:
            if self.current_session:
                self.current_session["status"] = Status.COMPLETED
                self.current_session["result"] = result
                self.last_session = self.current_session.copy()
                self.current_session = None  # Clear session after completion

    def fail_session(self, error: str):
        """Marks task as failed and stores error information."""
        with self._session_lock:
            if self.current_session:
                self.current_session["status"] = Status.FAILED
                self.current_session["error"] = error
                self.last_session = self.current_session.copy()
                self.current_session = None  # Clear session after failure

    def update_session_info(self, info: Dict[str, Any]):
        """Updates task session with arbitrary info."""
        with self._session_lock:
            if not self.current_session or self.current_session["status"] != Status.RUNNING:
                raise RuntimeError("No active task to update session info!")
            self.current_session.update(info)

    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Returns current task state information."""
        with self._session_lock:
            return {
                "current_session": self.current_session.copy() if self.current_session else None,
                "last_session": self.last_session.copy() if self.last_session else None,
            }

session_manager = SessionManager()

# Decorator to wrap task execution logic
def session_guard(task_name: str):
    """Ensures tasks are managed within SessionManager and handles failure states."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            session_manager = SessionManager()

            try:
                session_manager.start_session(task_name)
            except Exception as e:
                return {"error": f"failed to start session: {e}"}

            try:
                result = func(*args, **kwargs)  # Execute the training task
                session_manager.end_session(result)
                return result
            except Exception as e:
                print(traceback.format_exc())
                session_manager.fail_session(str(e))  # Record failure details
                # NOTICE: No Re-raise exception here,
                # as we capture the error and record it in session info.
                # raise e
                return {"error": f"failed to run {task_name}: {e}"}

        return wrapper

    return decorator


def start_session_with_subprocess(func, target_name: str, **kwargs):
    """Starts a new session in a separate process."""
    session_manager.start_session(target_name)
    return_queue = Queue()

    def wrapper(queue, **kwargs):
        result = func(**kwargs)
        queue.put(result)

    process = Process(target=wrapper, args=(return_queue,), kwargs=kwargs)
    process.start()
    session_manager.update_session_info({
        "status": Status.RUNNING,
        "pid": process.pid,
    })
    process.join()

    if not return_queue.empty():
        session_manager.end_session(return_queue.get())
    else:
        session_manager.fail_session("No result returned from subprocess.")


def stop_session_with_subprocess(task_name: str):
    """Stops a session started in a separate process."""
    session_info = session_manager.get_session_info()
    if not session_info:
        response = EaseVoiceResponse(ResponseStatus.SUCCESS, "No active task to stop.")
        session_manager.end_session(response)
        return response
    current_session = session_info.get("current_session", {})
    if current_session.get("task_name") != task_name or current_session.get("status") != Status.RUNNING:
        response = EaseVoiceResponse(ResponseStatus.FAILED, "Task name does not match.")
        session_manager.end_session(response)
        return response
    if current_session.get("pid"):
        _kill_proc_tree(current_session.get("pid"))
    response = EaseVoiceResponse(ResponseStatus.SUCCESS, "Task stopped by user.")
    session_manager.end_session(response)
    return response


def _kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)
        except OSError:
            pass


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

