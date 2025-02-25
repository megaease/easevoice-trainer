import subprocess
import threading
import tensorflow as tf
import numpy as np

tb_log_dir = "tb_logs"

def get_tensorboard_log_dir(name: str) -> str:
    """Get the TensorBoard log directory for a specific run.

    Args:
        name (str): Name of the run.

    Returns:
        str: Path to the TensorBoard log directory.
    """
    return f"{tb_log_dir}/{name}"

class TensorBoardService:
    """Service to run TensorBoard as a background process."""

    def __init__(self):
        self.log_dir = tb_log_dir
        self.tensorboard_process = None

    def run_tensorboard(self):
        """Run TensorBoard in a subprocess."""
        self.tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", self.log_dir, "--port", "6006"]
        )

    def start(self):
        """Start TensorBoard in a separate background thread."""
        thread = threading.Thread(target=self.run_tensorboard, daemon=True)
        thread.start()

    def stop(self):
        """Stop the TensorBoard process."""
        if self.tensorboard_process:
            self.tensorboard_process.terminate()  # Terminate the process
            self.tensorboard_process.wait()  # Wait for it to exit gracefully