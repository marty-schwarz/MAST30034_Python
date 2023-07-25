# Lecture 1

***All visualisations are in bold***

## Descriptive Statistics
- Continuous, e.g. trip distance, trip amount
- Discrete, e.g., passenger count
*both continuous and discrete are numerical*
- Categorical, e.g., payment type

### Types of Descriptive Statistics
Measures of
- frequency --> how often something occurs
- central tendency (mean / median / mode) --> use when want to show average or most commonly indicated response
- dispersion or variation (range (=max-min) variance / standard deviation / skew) --> use to show how spread the data is
- association (covariance, (Pearson) correlation, MI, **scatterplots**)
e.g., for MVN need vector mean and covariance matrix
PCA is a method to represent the original dependent variances using some new, transformed independent variables

Model: input --> output; want to know which features are most likely to affect 

Key characteristics
- Do the values tend to cluster around a particular point
- Is there more than one cluster? Multimodel?
- Variability in the values / how quickly do the probabilities drop off as we move away from the modes?
- Outliers: are there extreme values in the data?

## Machine Learning
*Want to make predictions with unseen inputs*
Most relationships are stochastic; this state of uncertainty is due mainly to the present of
- latent factors on which y depends but that are not observed or measured (hidden variables)
- measurement noise