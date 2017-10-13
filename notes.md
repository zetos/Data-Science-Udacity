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

**Mann-Whitney U test:** Tests null hypothesis that two populations are the same. It is a version of the *independent samples t-Test* that can be performed on *ordinal(ranked) data*.

 ## Linear regression with gradient descent.

Gradient descent is an algorithm that minimizes functions. Given a function defined by a set of parameters, gradient descent starts with an initial set of parameter values and iteratively moves toward a set of parameter values that minimize the function. This iterative minimization is achieved using calculus, taking steps in the negative direction of the function gradient.

`J(Theta)` is the cost function and `h(X^i)` is the predicted value of `Y^i`:

 ![alt text](https://lh3.ggpht.com/ybMIr_Y3W7q8aDRqBVkXVt0UxMhX5Q6O7XOzLzUgmnlpFVLDmj9yLxa0bEB1qPlbET4ItObiUhBMVJ-Xlj-R=s0#w=457&h=112)

 ```python
    import numpy

    def compute_cost(features, values, theta):
        m = len(values)
        predicted_values = numpy.dot(features, theta)
        sum_of_square_errors = numpy.square(predicted_values - values).sum()
        cost = sum_of_square_errors / (2*m)

        return cost
```

 Assigment statement used to update `theta_j`:

 ![update theta j](https://lh4.ggpht.com/GZTmhSbMXpeARDkr30QB0WPiMNiqJmQ1cUA3lidq5ybwPUzQmhY7d-33Izvfk4MDLOJDM_L0wGWKXlRVIRbQ=s0#w=256&h=51)

 Update theta using Gradient descent:

 ```python
    import numpy
    import pandas

    def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features. Performs num_iterations updates to the elements of theta.
    """
    cost_history = []
    m = len(values)
    
    for x in range(num_iterations):
        predicted_values = numpy.dot(features, theta)
        theta -= alpha / m * numpy.dot((predicted_values - values), features)
        
        cost = compute_cost(features, values, theta)
        cost_history.append(cost)
    

    return theta, pandas.Series(cost_history)
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

# Lesson 7 - Data Visualization

## Components of effetive visualization:

- Visual Cues / Visual encoding (Shapes, colors and sizes)
- Coordinating Systems. (What visual cues represent, dimensions X/Y and meaning)
- Scale / Data types. (Numerical, Dates, Temperatures, time, etc)
- Context.
    - Clarify what values represent.
    - Explain how to interpret data.
    - Titles, axis labels, annotations, etc.

Ps. Loess Curves are awesome

## Visual Encodings list by the most accurate:

1. Positon.
2. Lenght.
3. Angle.
4. Direction.
5. Area.
6. Volume.
7. Saturation.
8. Hue.

## Ploting in Python

packages: `Matplotlib`, `ggplot`, etc.

Using `ggplot`:

```python
print ggplot(data, aes(xvar, yvar)) + geom_point(color = 'coral') + geom_line(color='coral') + \
      ggtitle('title') + xlab('x-label') + ylab('y-label')
```

# Lesson 9 - Big Data and MapReducer

Split the data and compute in parallel.

## Mapper

Performs filtering and sorting.

## Reducer

Performs a summary operation.