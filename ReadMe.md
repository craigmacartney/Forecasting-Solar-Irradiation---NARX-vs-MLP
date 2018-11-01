# A Comparative Study in forecasting solar irradiation level using Non-linear Autoregressive Exogeneous Models and Multilayer Perceptron

This is a comparitive study, presenting an evaluation of two algorithm models, the Nonlinear autoregressive exogenous model (NARX) and multilayer perceptron (MLP) in a supervised time series regression task to forecast solar irradiation levels. Variations of the two algorithms are evaluated, varying their hyperparameters in a grid search manner, and validated through cross validation of time series training and test sequences. The tested results from the best evaluated models are compared by looking at the one step ahead forecast mean squared error (MSE) performance. NARX is found to have a considerable performance advantage over MLP in such a task.

This paper critically evaluates and compares the performance of NARX and MLP in forecasting the hourly irradiance levels of the Vichy Rolla National Airport’s photovoltaic station in 2009. This is the same data used in a previous NARX study to predict solar irradiation [1].

its greater ability to model non-linear time dependencies in time series data



## References
[1] A. Alzahrani, J. W. Kimball, and C. Dagli, "Predicting Solar Irradiance Using Time Series Neural Networks," in Complex Adaptive Systems, Philadelphia, PA, 2014, pp. 623 – 628.
