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

The elected stopping criterion is that based on the validation error increasing over successive steps, as mentioned in [5]. More specifically, we select a maximum of 10 validation error increases (alongside a maximum of 30 epochs and a minimum gradient of 1e-5) as stopping criterion in cross validation and final training/testing.
We also undertook a hyperparameter grid search to find the optimal learning rate, momentum and size of the hidden layers. This can increase the learning speed, help avoid local minima and improve generalization and performance of our model respectively.
Sliding window
Choosing the optimal sliding window size for each model is important as the cyclical nature of the solar irradiance time series trend may mean certain lagged predictors of a given variable are more useful for predicting solar irradiance than others. More specifically, to ensure the best performance of our models, we need to use cross-validation to select the optimal sliding window of n lagged exogenous predictors (y(t-1),...y(t-n), u(t-1),... u(t-n) for an equation of the form:
y(t) = f(y(t-1) +y(t-2)+... y(t-n) + u(t-1) + u(t-2)...u(t-n))


## References
[1] A. Alzahrani, J. W. Kimball, and C. Dagli, "Predicting Solar Irradiance Using Time Series Neural Networks," in Complex Adaptive Systems, Philadelphia, PA, 2014, pp. 623 – 628.

[2] Diaconescu, E., 2008. The use of NARX neural networks to predict chaotic time series. WSEAS Trans. Comput. Res. 3, 182–191

[3] Sfetsos and A. H. Coonick, "Univariate and multivariate forecasting of hourly solar radiation with artificial intelligence techniques," Solar Energy, pp. 169-178, 2000.

[4] S. Islam, M. Kabir and N. Kabir, "Artificial neural networks based prediction of insolation on horizontal surfaces for Bangladesh," in International Conference on Computational Intelligence: Modeling Techniques and Applications (CIMTA) 2013, 2013.

[5] L. Prechelt, Early stopping - but when?, in: Neural Networks: Tricks of the Trade - Second Edition, 2012, pp. 53–67.
