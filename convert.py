import torch

import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM, self).__init__()

        # CNN Feature extractor
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),  # (batch, 16, seq_len)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (batch, 16, seq_len/2)

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),  # (batch, 32, seq_len/2)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # (batch, 32, seq_len/4)
        )

        # LSTM Sequence Modeler
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        # Final classification layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        x = self.cnn_layers(x)

        # Prepare for LSTM: (batch, sequence_length, features)
        x = x.permute(0, 2, 1)

        # Pass through LSTM
        out, (hn, cn) = self.lstm(x)

        # Take only last LSTM output for classification
        out = out[:, -1, :]  # (batch, hidden_size)

        out = self.fc(out)
        return out

# Step 1: Rebuild your model architecture
model = CNN_LSTM(num_classes=4)  

# Step 2: Load the weights
model.load_state_dict(torch.load("best_cnn_lstm_model.pth", map_location=torch.device('cpu')))
model.eval()

# Step 3: Prepare dummy input (shape must match training input)
dummy_input = torch.randn(1, 3, 200)  # (batch, channels, seq_len)

# Step 4: Trace the model
traced_model = torch.jit.trace(model, dummy_input)

# Step 5: Save the traced model
traced_model.save("cnn_lstm.pt")
