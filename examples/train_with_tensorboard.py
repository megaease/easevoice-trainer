import tensorflow as tf
import time
import random
import numpy as np

# Run this script to simulate training a model and log various types of data to TensorBoard
# uv run ./examples/train_with_tensorboard.py

# Path to the directory where TensorBoard logs will be stored
log_dir = "tb_logs/train"

# Create a file writer for TensorBoard
writer = tf.summary.create_file_writer(log_dir)

# Function to simulate training and log different types of data to TensorBoard
def log_training_data():
    """Simulate model training and log various types of data to TensorBoard periodically."""
    for epoch in range(1, 101):  # Simulate 100 epochs
        # Simulate some training data
        loss = random.uniform(0.1, 1.0)  # Random loss between 0.1 and 1.0
        accuracy = random.uniform(0.5, 1.0)  # Random accuracy between 0.5 and 1.0

        # Log scalar values (e.g., loss, accuracy)
        with writer.as_default():
            tf.summary.scalar("loss", loss, step=epoch)
            tf.summary.scalar("accuracy", accuracy, step=epoch)

        # Log histogram values (e.g., weights, activations)
        random_weights = np.random.rand(10)  # Simulated weights (array of random values)
        tf.summary.histogram("weights", random_weights, step=epoch)

        # Log image data (e.g., visualizing model filters, feature maps)
        random_image = np.random.rand(64, 64, 3)  # Simulated random image (64x64 RGB)
        tf.summary.image("random_image", np.expand_dims(random_image, axis=0), step=epoch)  # Shape: [1, 64, 64, 3]

        # Log text data (e.g., logging custom training messages)
        text = f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {accuracy:.4f}"
        tf.summary.text("training_message", text, step=epoch)

        # Log histogram for random data distribution
        random_data = np.random.randn(1000)  # 1000 points from a normal distribution
        tf.summary.histogram("random_data_distribution", random_data, step=epoch)

        # Log audio data (e.g., model-generated audio)
        sample_rate = 16000  # Simulated audio sample rate
        audio = np.random.randn(16000)  # 1-second random audio signal
        tf.summary.audio("random_audio", np.expand_dims(audio, axis=-1), sample_rate=sample_rate, step=epoch)

        # Flush the writer to ensure data is written to disk
        writer.flush()

        print(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")

        # Simulate some training delay (e.g., processing time)
        time.sleep(1)  # Sleep for 1 second to simulate training time

    print("Training complete!")

if __name__ == "__main__":
    log_training_data()