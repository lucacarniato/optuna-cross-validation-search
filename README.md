# Optuna cross validation search

Performing hyper-parameters search for models implementing the scikit-learn interface, by using cross-validation and the Bayesian framework [Optuna](https://github.com/optuna/optuna).

# Usage examples

In the following example, the hyperparameters of a [lightgbm](https://lightgbm.readthedocs.io/en/latest/) classifier are estimated:

        from OptunaCrossValidationSearch import OptunaCrossValidationSearch
        import lightgbm as lgb
        
        classifier = lgb.LGBMClassifier(n_jobs=4)
        # The parameters prior distributions
        parameter_distributions = {'min_data_in_leaf': optuna.distributions.IntUniformDistribution(10, 100),
                                   'bagging_fraction': optuna.distributions.LogUniformDistribution(0.01, 1.0)}
        optuna_cross_validation = OptunaCrossValidationSearch(classifier = classifier,
                                                              parameter_distributions = parameter_distributions,
                                                              cv_folds = 5,
                                                              n_trials = 10,
                                                              sample_weight_balance = 'balanced')
        optuna_cross_validation.fit(X_train, y_train)
        y_test_pred = optuna_cross_validation.predict(X_test)
        score = sklearn.metrics.accuracy_score(y_test, y_test_pred)

In the following example, the hyperparameters of a Keras deep learning model are estimated. 
In this case the number of tunable parameters is not fixed, and the Keras model is wrapped in the ModelKerasFullyConnected class.
ModelKerasFullyConnected derives from ModelKeras base class, where the methods are defined. 
During cross-validation of a keras model, a callback function is used to stop fitting the model when the validation accuracy does not improve after 50 epochs.

        from OptunaCrossValidationSearch import OptunaCrossValidationSearch
        from ModelKerasFullyConnected import ModelKerasFullyConnected

        classifier = ModelKerasFullyConnected.ModelKerasFullyConnected(X_train.shape, len(np.unique(y_train)))
        # The parameters prior distributions
        parameter_distributions = {'num_units': optuna.distributions.IntUniformDistribution(32, 128),
                                   'num_hidden': optuna.distributions.IntUniformDistribution(0, 3),
                                   'dropout': optuna.distributions.LogUniformDistribution(0.05, 0.4),
                                   'learning_rate': optuna.distributions.LogUniformDistribution(1e-8, 1e-4)}

        optuna_cross_validation = OptunaCrossValidationSearch(classifier = classifier,
                                                              parameter_distributions = parameter_distributions,
                                                              cv_folds = 5,
                                                              n_trials = 10,
                                                              sample_weight_balance = 'balanced')
        optuna_cross_validation.fit(X_train, y_train)

        y_test_pred = optuna_cross_validation.predict(X_test)
        score = sklearn.metrics.accuracy_score(y_test, y_test_pred)


