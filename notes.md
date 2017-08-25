# Notes on Intro do Data Science Course.

# Lesson 5

* Statistical Rigor.
* Data observational or experimental.
* Check Statistical Significance.
* Understand Confidance Intervals.

    * Tests changes based on data distribution

## T-Tests

Accept or reject a *null hypothesis*.

Specified in terms of a *test statistic*.

TEST STATISTIC: One number that helps accept or reject the null hypothesis.

There are a number of different T-tests depending on assumptions like: "Equal sample size?" and "Same variance?"

The **Welch's Two-Sample t-Test:** A general test for a gaus distributions that doesn't assume equal sample size or variance.

* Calculate T (T-statistic) and V(Aproximated degrees of freedom) to Calculate P-Value and compate it with P-Critical.

The **Shapiro-Wilk Test:** A test of normality in frequentist statistics

**Non-parametric tests:** A statistical test that does not assume our data is drawn from any particular underlying probability distribution.

**Mann-Whitney U test:** Tests null hypothesis that two populations are the same.

 ## Linear regression with gradient descent.

Gradient descent is an algorithm that minimizes functions. Given a function defined by a set of parameters, gradient descent starts with an initial set of parameter values and iteratively moves toward a set of parameter values that minimize the function. This iterative minimization is achieved using calculus, taking steps in the negative direction of the function gradient.

`J(Theta)` is the cost function and `h(X^i)` is the predicted value of `Y^i`:

 ![alt text](https://lh3.ggpht.com/ybMIr_Y3W7q8aDRqBVkXVt0UxMhX5Q6O7XOzLzUgmnlpFVLDmj9yLxa0bEB1qPlbET4ItObiUhBMVJ-Xlj-R=s0#w=457&h=112)

 ```python
    m = len(values)
    sum_of_square_errors = numpy.square(numpy.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)
```

 Assigment statement used to update `theta_j`:

 ![update theta j](https://lh4.ggpht.com/GZTmhSbMXpeARDkr30QB0WPiMNiqJmQ1cUA3lidq5ybwPUzQmhY7d-33Izvfk4MDLOJDM_L0wGWKXlRVIRbQ=s0#w=256&h=51)

 ```python
    predicted_values = numpy.dot(features, theta)
    theta -= alpha / m * numpy.dot((predicted_values - values), features)
 ```

 **Additional considerations on linear regration:**
  - Other types of linear regression.
    - ordinary least squares regression.
  - Parameter estimation.
  - Under / Overfitting.
  - Multiple local minima.

 ## Coefficient of Determination (R²)

The coefficient of determination `R²`, is the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

It can be used to determine the effectiveness of a model.

```python
import numpy as np

def compute_r_squared(data, predictions):
    # The closer to 1 better is the model. Closer to 0 poorer is the model.
    avarageData = np.mean(data)
    r_squared = 1 - np.sum(np.square(data - predictions)) / np.sum(np.square(data - avarageData))

    return r_squared
```

## Aditional considerations

- Clustering : 'k-means clustering', 'hierarchical clustering'.
- PCA: principle component analysis.
- Understand casual connections.
