import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.out(hn[-1])


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hn = self.gru(x)
        return self.out(hn[-1])


class LinearModel(nn.Module):
    def __init__(self, input_dim, window_size=30, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(input_dim * window_size, output_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # flatten
        return self.fc(x)
