# A Comparative Study in forecasting solar irradiation level using Non-linear Autoregressive Exogenous Models and Multilayer Perceptron

This is a comparitive study, presenting an evaluation of two algorithm models, the Nonlinear autoregressive exogenous model (NARX) and multilayer perceptron (MLP) in a supervised time series regression task to forecast solar irradiation levels. Variations of the two algorithms are evaluated, varying their hyperparameters in a grid search manner, and validated through cross validation of time series training and test sequences. The tested results from the best evaluated models are compared by looking at the one step ahead forecast mean squared error (MSE) performance. NARX is found to have a considerable performance advantage over MLP in such a task.

evaluates and compares the performance of NARX and MLP in forecasting the hourly irradiance levels of the Vichy Rolla National Airport’s photovoltaic station in 2009. This is the same data used in a previous NARX study to predict solar irradiation [1].

its greater ability to model non-linear time dependencies in time series data

## Hypothesis
The initial hypothesis is that NARX is expected to perform better than MLP in forecasting solar radiation because of its greater ability to model non-linear time dependencies in time series data

## Initial analysis and data description
In the Solar_viz.m file, the trend of solar irradiance is plotted over time to identify any seasonality or stationarity related issues, which may hinder the performance of neural networks (as outlined in [2]).

The trend in solar irradiance is stationary as overall the moving average of solar irradiance is not increasing or decreasing over time. However, there are seasonal trends as solar irradiance drops at night and is greater over the summer months.

To forecast solar irradiance, lagged predictors from five variables were constructed, including solar irradiance itself. SumStats.mat shows summary statistics (minimum, maximum, mean, variation, kurtosis) for each of these variables. Three of are continuous (Solar irradiance, Azimuth angle and Zenith angle) and two are discrete time variables (Day & Hour).

To identify potential lagged predictors to derive from these variables, their correlation with solar irradiance was considered. It was found that, apart from the hour variable, which has a consistently low correlation, for at least two variables, lags 1-4 had a correlation greater than 0.5. Although any lagged predictors derived from hours could be omitted, it was decided they should remain, since they may help detrend seasonality issues from the time series forecasts.

## Methodology
80% of the data os used for training (and cross-validation) and 20% for testing. However, in order to keep time series dependencies, rather than randomly sampling the test set, the latest 20% of the time sequence is chosen as the test data sequence.

For cross-validation: The training time series data is partitioned into 10 folds, each with a training set and a randomly sampled test sequence (not a test set). This differs from conventional k-fold cross validation in that test sequences are randomly sampled in each fold; rather than individual training data points. This method is preferred since it better preserves the time dependencies in the test and training data within each fold. The comparative performance of different hyperparameter value combinations is evaluated on each model using one step ahead MSE performance.

For final training/testing: The variants of the two models (MLP and NARX) are trained with optimal hyperparameter combinations. The variants are then evaluated on the test data, assessing their performance in making one step ahead predictions using MSE.

## NARX and BACKPROPOGATION hyperparameters
The “Levenberg-Marquardt” (LM) function is selected to update the weights for each neuron in both models. This algorithm has been found to perform better than Bayesian Regularization (BR) and Scaled Conjugate algorithm (SCG) when training similar NARX models to forecast solar irradiance in [3] and [4]. The function allows the NARX model to converge to local minima much faster than the standard backpropagation algorithm.

Training NARX and MLP using LM usually begins with setting random weight values. This avoids networks being stuck in the same local minimum each time they are trained. Each model then follows their respective backpropagation algorithms to train. When informally training both models, it was noticed that the difference between the training error and the test error monotonically diverges over time. This means that after many training epochs the model no longer improves in test accuracy and overfits the data. This may be prevented with early stopping based on certain criteria.

The elected stopping criterion is that based on the validation error increasing over successive steps, as mentioned in [5]. More specifically, a maximum of 10 validation error increases are selected (alongside a maximum of 30 epochs and a minimum gradient of 1e-5) as stopping criterion in cross validation and final training/testing. A hyperparameter grid search is also undertaken to find the optimal learning rate, momentum and size of the hidden layers. This can increase the learning speed, help avoid local minima and improve generalization and performance of our model respectively.

## Sliding window
Choosing the optimal sliding window size for each model is important as the cyclical nature of the solar irradiance time series trend may mean certain lagged predictors of a given variable are more useful for predicting solar irradiance than others. More specifically, to ensure the best performance of each model, cross-validation is required to select the optimal sliding window of n lagged exogenous predictors (y(t-1),...y(t-n), u(t-1),... u(t-n) for an equation of the form:

y(t) = f(y(t-1) +y(t-2)+... y(t-n) + u(t-1) + u(t-2)...u(t-n))


## Results, Findings & Evaluation
Cross-validation results

As per the FINALCROSSVALRESULTS.mat file, the lowest MSEs, for the hyperparameter combination where the two models perform best, are when the window is restricted to a size of two lags and the hidden layer size is limited to twenty neurons. This is consistent with [1].

NARX tends to favour a lower learning rate and momentum relative to MLP. That being said, across all hyperparameter combinations, there was not a large performance difference. For NARX and MLP there was only a 9.7% and 5.5% MSE test difference between the most and least optimal hyperparameter combination.


Comparison of final training and test results

As expected, NARX significantly outperforms MLP in forecasting future levels of solar irradiation one step ahead. The two models have a test MSE performance of 0.0116 and 0.0226 respectively, as per the FINALTEST....mat files.

Yet the absolute values for MSE performance are low for both algorithms, and uncontrolled factors may contribute to performance differences between them. For example, the random setting of initial weight values in each model may lead to significant performance differences between different training runs.

Further, looking at the respective train/test performance graphs for each algorithm, contained within the .fig files, it was noticed that the difference in training and test performance (the red and blue lines) for MLP is much greater than it is for NARX, suggesting more overfitting under MLP. And that the training and test performance diverges in a much earlier epoch for MLP than it does for NARX. This may mean that differences in performance between the two algorithms are down to suboptimal stopping criterion when training MLP – and not superior test performance under NARX.

Finally, it should be noted that some of the performance differences between the two algorithms may be attributed to suboptimal hyperparameter value choices – as the full hyperparameter space has not been fully searched through the cross-validation grid search.

## Conclusion, Lessons learned and Further work
NARX outperforms MLP in forecasting solar irradiation time series. This is likely down to its greater ability to model time series dependencies in data.

Two major lessons are also identified in cross validating Neural Networks that model time series data. The first is that standard cross-validation techniques, such as K-folds, are not appropriate for training/cross validating time series models since these techniques remove the time dependent nature from the data, so time series alternatives that preserve time dependency need to be used. The second is that, in addition to finding the optimal hyperparameter values for each model through cross-validation, it is also important to identify the optimal training window size to train the model with.

In the evaluation section, confounding factors were identified (stopping criterion, initial weight values and hyperparameter values) that may have contributed to performance differences between NARX and MLP. Future work could focus on further optimizing/controlling for these values through cross-validation.


## References
[1] A. Alzahrani, J. W. Kimball, and C. Dagli, "Predicting Solar Irradiance Using Time Series Neural Networks," in Complex Adaptive Systems, Philadelphia, PA, 2014, pp. 623 – 628.

[2] Diaconescu, E., 2008. The use of NARX neural networks to predict chaotic time series. WSEAS Trans. Comput. Res. 3, 182–191

[3] Sfetsos and A. H. Coonick, "Univariate and multivariate forecasting of hourly solar radiation with artificial intelligence techniques," Solar Energy, pp. 169-178, 2000.

[4] S. Islam, M. Kabir and N. Kabir, "Artificial neural networks based prediction of insolation on horizontal surfaces for Bangladesh," in International Conference on Computational Intelligence: Modeling Techniques and Applications (CIMTA) 2013, 2013.

[5] L. Prechelt, Early stopping - but when?, in: Neural Networks: Tricks of the Trade - Second Edition, 2012, pp. 53–67.
