
import numpy as np

# Observed and predicted temperature
y_true = np.array([23.0])
y_pred = np.array([20.0])

# 1. Ordinary MSE
mse = np.mean((y_true - y_pred) ** 2)

# 2. Gaussian negative log-likelihood
sigma = 1.0

gaussian_nll = np.mean(
    (y_true - y_pred)**2 / (2 * sigma**2)
    + 0.5 * np.log(2 * np.pi * sigma**2)
)

print("MSE:", mse)
print("Gaussian NLL:", gaussian_nll)

# Change sigma without changing the mean
for sigma in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    nll = np.mean(
        (y_true - y_pred)**2 / (2 * sigma**2)
        + 0.5 * np.log(2 * np.pi * sigma**2)
    )
    print(f"sigma={sigma:4.1f}, NLL={nll:.4f}")
