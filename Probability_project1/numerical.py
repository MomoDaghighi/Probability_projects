import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad

# Define the mean and covariance for the Gaussian distribution
mean = [2, 3]
cov = [[1, 0], [0, 1]]  # Independent variables, diagonal covariance


# Define the density function (pdf) for X and Y
def joint_density(x, y):
    return multivariate_normal.pdf([x, y], mean=mean, cov=cov)


# Integral limits for the circle with radius 1
def circle_upper(x):
    return np.sqrt(1 - x ** 2)


def circle_lower(x):
    return -np.sqrt(1 - x ** 2)


# Numerical calculation of the double integral over the circle with radius 1
numerical_integral, error = dblquad(joint_density, -1, 1, circle_lower, circle_upper)
print(numerical_integral, error)


