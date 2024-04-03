import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt


# Function to compute the quantile from uniform random numbers using the Lambert W function
def quantile(u):
    # Use the principal branch of the Lambert W function
    return -float(lambertw((u - 1) * np.exp(-1), k=0).real) - 1


# Generate uniform random numbers
np.random.seed(42)  # For reproducibility
num_samples = 10000
uniform_samples = np.random.uniform(low=0, high=1, size=num_samples)

# Generate samples from the desired distribution
generated_samples = np.array([quantile(u) for u in uniform_samples])

# Plot the empirical distribution of the generated samples
plt.hist(generated_samples, bins=50, density=True, alpha=0.6, color='g', label='Empirical PDF')

# Plot the theoretical PDF
x_vals = np.linspace(0, 7, 500)
y_vals = x_vals * np.exp(-x_vals)
plt.plot(x_vals, y_vals, 'r', label='Theoretical PDF')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Comparison of Empirical and Theoretical PDF')
plt.legend()

# Show the plot
plt.show()
