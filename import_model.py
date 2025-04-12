import torch
import numpy as np

# Load the scripted model
model_path = "runs/topk_experiment/best_model.pt.jit"
model = torch.jit.load(model_path)
model.eval()

# Example input: batch of 2 samples
# Each input has shape (4, 3): [mu, sigma, n] for 4 groups
x = torch.tensor([
    [[2.0, 1.0, 10], [1.5, 0.5, 8], [0.0, 2.0, 12], [3.5, 1.2, 10]],
    [[1.0, 1.0, 20], [1.0, 1.0, 20], [1.0, 1.0, 20], [1.0, 1.0, 0]]
], dtype=torch.float32)  # shape (2, 4, 3)

# Example normalized k values (e.g., k = 5 out of max 60)
k = torch.tensor([[5 / 60], [3 / 60]], dtype=torch.float32)  # shape (2, 1)

# Run prediction
with torch.no_grad():
    predictions = model(x, k)

print("Predicted top-k sums:", predictions.numpy())
