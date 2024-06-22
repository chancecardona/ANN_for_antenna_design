import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        hidden_size = (2*input_size + 1)
        self.layers = nn.Sequential(
            # Input layer
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # First hidden layer
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            # Output layer
            nn.Linear(output_size, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def loss_fn(self, predicted_coeffs, actual):
        # Predicted is currently just predicted pole coefficients. Need to plug in.
        # Actual is S-params.
        freqs = actual[0][:, 0]
        predicted = PoleResidueTF(predicted_coeffs, freqs)
        return mean_absolute_percentage_error(actual, predicted)

    def PoleResidueTF(self, coefficients, freqs):
        # s is the frequency
        H = np.zeros(len(freqs))
        for s in range(len(freqs)):
            for i in range(len(coefficents)):
                H[s] += (coefficients[i, 0]) / (freqs[s] - coefficients[i, 1])
        return H
