import torch
import torch.nn as nn
import numpy as np



class SimpleDQN(nn.Module):
    """A simple Feed-Forward Neural Network (MLP) for Q-value approximation."""

    def __init__(self, n_states: int, n_hidden: int, n_actions: int):
        super(SimpleDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DuelingDQN(nn.Module):
    """Dueling Network Architecture for MLPs."""

    def __init__(self, n_states: int, n_hidden: int, n_actions: int):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


class CNN1D_DQN(nn.Module):
    """1D Convolutional Neural Network for Q-value approximation."""

    def __init__(self, n_states: int, n_hidden: int, n_actions: int):
        super(CNN1D_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Calculate flattened size after conv layers
        dummy_input = torch.zeros(1, 1, n_states)
        conv_out_size = self._get_conv_out(dummy_input)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )

    def _get_conv_out(self, x: torch.Tensor) -> int:
        return self.conv(x).view(x.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is (batch_size, n_states), reshape to (batch_size, 1, n_states) for Conv1d
        if x.ndim == 2:
            x = x.unsqueeze(1)
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)


class Dueling_CNN1D_DQN(nn.Module):
    """Dueling variant of the 1D CNN."""

    def __init__(self, n_states: int, n_hidden: int, n_actions: int):
        super(Dueling_CNN1D_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        dummy_input = torch.zeros(1, 1, n_states)
        conv_out_size = self._get_conv_out(dummy_input)

        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_actions)
        )

    def _get_conv_out(self, x: torch.Tensor) -> int:
        return self.conv(x).view(x.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        features = self.conv(x).view(x.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


# --- For Environments Requiring Memory (LSTM) ---
# NOTE: Training recurrent networks with a standard replay buffer has caveats.
# The hidden state is not stored in the buffer, so the LSTM's memory is reset
# for each batch, limiting its ability to learn long-term dependencies across batches.
# However, it can still learn short-term patterns within the data.

class LSTM_DQN(nn.Module):
    """LSTM-based network for Q-value approximation."""

    def __init__(self, n_states: int, n_hidden: int, n_actions: int):
        super(LSTM_DQN, self).__init__()
        self.lstm = nn.LSTM(input_size=n_states, hidden_size=n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is (batch_size, n_states), reshape to (batch_size, 1, n_states) for LSTM
        if x.ndim == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        # We only care about the output of the last time step
        return self.fc(lstm_out[:, -1, :])


class Dueling_LSTM_DQN(nn.Module):
    """Dueling variant of the LSTM network."""

    def __init__(self, n_states: int, n_hidden: int, n_actions: int):
        super(Dueling_LSTM_DQN, self).__init__()
        self.lstm = nn.LSTM(input_size=n_states, hidden_size=n_hidden, batch_first=True)
        self.value_stream = nn.Linear(n_hidden, 1)
        self.advantage_stream = nn.Linear(n_hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

