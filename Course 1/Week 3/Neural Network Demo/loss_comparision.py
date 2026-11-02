
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load real data
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Model A: predicts a mean only
mse_model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Model B: predicts a mean and log variance
gaussian_model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

mse_optimizer = torch.optim.Adam(mse_model.parameters(), lr=0.001)
nll_optimizer = torch.optim.Adam(gaussian_model.parameters(), lr=0.001)

for epoch in range(100):
    # Training Model A
    mse_optimizer.zero_grad()

    predicted_mean = mse_model(X_train)
    mse_loss = nn.functional.mse_loss(predicted_mean, y_train)

    mse_loss.backward()
    mse_optimizer.step()

    # Training Model B
    nll_optimizer.zero_grad()

    output = gaussian_model(X_train)
    mean = output[:, :1]
    log_variance = output[:, 1:]

    # Gaussian NLL
    variance = torch.exp(log_variance)
    nll_loss = nn.functional.gaussian_nll_loss(
        mean, y_train, variance,
        full=True
    )

    nll_loss.backward()
    nll_optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}")
        print(f"  MSE Loss: {mse_loss.item():.4f}")
        print(f"  Gaussian NLL: {nll_loss.item():.4f}")
