import numpy as np


def estimate_probability(mean, cov, desired_accuracy, initial_sample_size=100000):
    # Function to generate samples and estimate the probability
    num_samples = initial_sample_size
    while True:
        # Generate samples from Gaussian distributions
        x_samples = np.random.normal(mean[0], cov[0][0], num_samples)
        y_samples = np.random.normal(mean[1], cov[1][1], num_samples)

        # Check if the samples are within the unit circle
        inside_circle = (x_samples ** 2 + y_samples ** 2) < 1

        # Estimate the probability
        probability_estimate = inside_circle.mean()
        # Calculate the standard error
        standard_error = inside_circle.std() / np.sqrt(num_samples)

        # Check if the standard error is within the desired accuracy
        if standard_error <= desired_accuracy * probability_estimate:
            break

        # If not, increase the sample size
        num_samples *= 2

    return probability_estimate, standard_error, num_samples


# Parameters for the Gaussian distributions
mean = [2, 3]
cov = [[1, 0], [0, 1]]  # Independent variables, diagonal covariance
desired_accuracy = 0.01  # 1% of the estimated probability

# Calculate the estimate and the required number of samples
probability_estimate, standard_error, sample_size = estimate_probability(mean, cov, desired_accuracy)
print(f'Estimated probability: {probability_estimate}')
print(f'Standard error: {standard_error}')
print(f'Number of samples used: {sample_size}')
