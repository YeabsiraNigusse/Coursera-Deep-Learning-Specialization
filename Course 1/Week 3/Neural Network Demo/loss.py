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


# 3 observed temperatures
y_true = np.array([23.0, 18.0, 30.0])

# Mean predictions from the neural network
mu = np.array([20.0, 17.0, 29.0])

# Standard deviations predicted by the network
sigma = np.array([3.0, 0.5, 2.0])

# MSE
mse = np.mean((y_true - mu)**2)

# Gaussian NLL
nll_per_sample = (
    (y_true - mu)**2 / (2 * sigma**2)
    + np.log(sigma)
    + 0.5 * np.log(2 * np.pi)
)

gaussian_nll = np.mean(nll_per_sample)

print("MSE:", mse)
print("Gaussian NLL:", gaussian_nll)
print("Individual NLL:", nll_per_sample)


# The crucial insight: With a fixed standard deviation, Gaussian NLL and MSE are equivalent for optimizing the mean. 
# With a learned standard deviation, Gaussian NLL trains the network to model both the predicted value and the uncertainty around it.