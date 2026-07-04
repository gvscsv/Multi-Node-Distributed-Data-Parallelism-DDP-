import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from torch.utils.data import DataLoader, TensorDataset
import argparse

def main():
    # --- 1. Argument parser for DeepSpeed ---
    parser = argparse.ArgumentParser(description="DeepSpeed CPU Distributed Training Script")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # --- 2. Environment Setup ---
    # Explicitly configure PyTorch for CPU-only execution paths
    device = torch.device('cpu')

    # --- 3. Synthetic Dataset Generation ---
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- 4. Model Architecture ---
    class SimpleNet(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
            super(SimpleNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleNet().to(device)

    # --- 5. Optimization Strategy ---
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- 6. DeepSpeed Initialization ---
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer
    )

    # --- 7. Distributed Training Loop ---
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Bind data stream directly to CPU execution runtime
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model_engine(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            model_engine.backward(loss)
            model_engine.step()

            # Prevent terminal clutter: aggregate print logging exclusively on Rank 0
            if batch_idx % 10 == 0:
                if model_engine.local_rank == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] | "
                          f"Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")

    print(f"Process Node [Rank {model_engine.local_rank}] completed training run successfully.")

if __name__ == "__main__":
    main()
