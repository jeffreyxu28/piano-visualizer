"""CNN + BiLSTM piano transcription model.

This architecture is intentionally identical to the one used to produce
``best_piano_model.pt`` during training. Do not change layer shapes here
without retraining, or ``load_state_dict`` will fail to match.
"""

from __future__ import annotations

import torch
from torch import nn

from backend.config import N_KEYS, N_MELS


class PianoTranscriptionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        frequency_after_pooling = N_MELS // 4
        cnn_features = 64 * frequency_after_pooling

        self.projection = nn.Sequential(
            nn.Linear(cnn_features, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
        )

        self.onset_lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.onset_head = nn.Linear(256, N_KEYS)

        self.frame_lstm = nn.LSTM(
            input_size=256 + N_KEYS,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.frame_head = nn.Linear(256, N_KEYS)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [batch, time, mel] -> [batch, 1, mel, time]
        x = mel.transpose(1, 2).unsqueeze(1)
        x = self.cnn(x)

        # [batch, channels, frequency, time] -> [batch, time, features]
        x = x.permute(0, 3, 1, 2).contiguous().flatten(start_dim=2)
        features = self.projection(x)

        onset_hidden, _ = self.onset_lstm(features)
        onset_logits = self.onset_head(onset_hidden)
        onset_probabilities = torch.sigmoid(onset_logits)

        frame_input = torch.cat([features, onset_probabilities], dim=-1)
        frame_hidden, _ = self.frame_lstm(frame_input)
        frame_logits = self.frame_head(frame_hidden)

        return onset_logits, frame_logits
