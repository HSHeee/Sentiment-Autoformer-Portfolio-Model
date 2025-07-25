# modeling/train_loop.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from modeling.nsautoformer import NSAutoformer

def train_model(X, y, input_dim, epochs=10, lr=1e-3):
    model = NSAutoformer(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
