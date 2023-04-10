import pandas as pd
import numpy as np

# modeling methods
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE

import matplotlib.pyplot as plt

def get_baseline_model(y_train):
    '''
    generate a baseline model for comparison
    '''
    # get the mean tax value
    y_train['tax_value_pred_mean'] = y_train.tax_value.mean()
    # calculate the rmse of the mean tax value
    rmse_train_mu = mean_squared_error(y_train.tax_value,
                                   y_train.tax_value_pred_mean, squared=False)
    # print baseline metrics
    print('Baseline Model (mean)')
    print(f'RMSE for baseline model: {rmse_train_mu:.08}')
    print('R^2 for baseline model: 0.0')


def get_lars_model(X_train_scaled, y_train, 
                   X_validate_scaled, y_validate, f_features):    
    '''
    create a lasso + lars model
    '''
    # make the lars model
    lars = LassoLars(alpha=0.1)
    # fit the model to the training data
    lars.fit(X_train_scaled[f_features], y_train.tax_value)
    # use the model to make predictions
    y_train['tax_value_pred_lars'] = lars.predict(X_train_scaled[f_features])
    # Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars) ** .5
    
    # repeat usage on validate
    y_validate['tax_value_pred_lars'] = lars.predict(X_validate_scaled[f_features])
    # evaluate: RMSE on validate
    rmse_validate = mean_squared_error(y_validate.tax_value, 
                                       y_validate.tax_value_pred_lars) ** .5
    # calculate the r^2
    r_2 = explained_variance_score(y_validate.tax_value,
                                       y_validate.tax_value_pred_lars)
    # print the lasso lars metrics
    print('Lasso + Lars Model')
    print(f'RMSE on training data: {rmse_train:.08}')
    print(f'RMSE on validation data: {rmse_validate:.08}')
    print(f'Difference in RMSE: {rmse_validate - rmse_train:.08}')
    print(f'R^2 value: {r_2:0.4}')
    # return the model for potential use on test data
    return lars


def get_tweedie_model(X_train_scaled, y_train, 
                      X_validate_scaled, y_validate, f_features):
    '''
    create a general linear model (tweedie)
    '''
    # make a tweedie regressor model
    glm = TweedieRegressor(power=1, alpha=0)
    # fit the model to the training data
    glm.fit(X_train_scaled[f_features], y_train.tax_value)
    # use the model for predictions
    y_train['tax_value_pred_glm'] = glm.predict(X_train_scaled[f_features])
    # Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_glm) ** .5

    # repeat usage on validate
    y_validate['tax_value_pred_glm'] = glm.predict(X_validate_scaled[f_features])
    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_validate.tax_value, 
                                       y_validate.tax_value_pred_glm) ** .5
    # calculate the r^2
    r_2 = explained_variance_score(y_validate.tax_value,
                                       y_validate.tax_value_pred_glm)
    # print the metrics
    print('Tweedie Model')
    print(f'RMSE on training data: {rmse_train:.08}')
    print(f'RMSE on validation data: {rmse_validate:.08}')
    print(f'Difference in RMSE: {rmse_validate - rmse_train:.08}')
    print(f'R^2 value: {r_2:0.4}')
    # return the model for potential use on test data
    return glm


def get_polynomial_model(X_train_scaled, y_train, 
                         X_validate_scaled, y_validate,
                         X_test_scaled,
                         f_features):
    '''
    create a polynomial model
    '''
    # Create the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2) #quadratic function
    #1 Fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled[f_features])
    # Transform X_validate_scaled & X_test_scaled 
    X_validate_degree2 = pf.fit_transform(X_validate_scaled[f_features])
    X_test_degree2 = pf.fit_transform(X_test_scaled[f_features])

    # make model
    lm2 = LinearRegression()
    # fit the model to the training data
    lm2.fit(X_train_degree2, y_train.tax_value)
    # using the model to make predictions
    y_train['tax_value_pred_lm2'] = lm2.predict(X_train_degree2)
    # Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.tax_value, 
                                    y_train.tax_value_pred_lm2) ** .5

    # repeat usage on validate
    y_validate['tax_value_pred_lm2'] = lm2.predict(X_validate_degree2)
    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_validate.tax_value, 
                                       y_validate.tax_value_pred_lm2) ** .5
    # calculate the r^2
    r_2 = explained_variance_score(y_validate.tax_value,
                                       y_validate.tax_value_pred_lm2)
    # print the metrics
    print('Polynomial Model')
    print(f'RMSE on training data: {rmse_train:.08}')
    print(f'RMSE on validation data: {rmse_validate:.08}')
    print(f'Difference in RMSE: {rmse_validate - rmse_train:.08}')
    print(f'R^2 value: {r_2:0.4}')
    # return the model and 2nd degree test dataset for potential use on test data
    return lm2, X_test_degree2

def get_polynomial_test(lm2, X_test_degree2, y_test):
    '''
    create predictions on the test dataset
    '''
    # create predictions on test data
    y_test['tax_value_pred_lm2'] = lm2.predict(X_test_degree2)
    # Evaluate: RMSE for test data
    rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm2) ** .5
    # calculate r^2 value of test data
    r_2 = explained_variance_score(y_test.tax_value,
                                       y_test.tax_value_pred_lm2)
    # print metrics
    print('Polynomial Model on Test Data')
    print(f'RMSE on test data: {rmse_test:.08}')
    print(f'R^2 value: {r_2:0.4}')
    # return test modified test data for plotting
    return y_test

def get_pred_error_plot(y_test):
    '''
    create plot of prediction residuals
    '''
    # create the figure
    plt.figure(figsize=(16,8))
    plt.axhline(label="No Error")
    plt.scatter(y_test.tax_value, (y_test.tax_value_pred_lm2 - y_test.tax_value), 
                alpha=.5, color="grey", s=100, label="Model 2nd degree Polynomial")
    # change the ticks to something readable
    plt.xticks(ticks=[0,200_000,400_000,600_000,800_000,1_000_000], 
               labels=['0', '200,000', '400,000', '600,000', '800,000', '1,000,000'],
               size = 12)
    plt.yticks(size=12,
               ticks=[600_000, 400_000, 200_000, 0, -200_000, -400_000, 
                      -600_000, -800_000, -1_000_000],
               labels=['600,000', '400,000', '200,000', '0', '-200,000', '-400,000', 
                      '-600,000', '-800,000', '-1,000,000'])
    # add axis labels
    plt.xlabel('Actual Home Value (Dollars)', size=14)
    plt.ylabel('Prediction Error (Dollars)', size=14)
    # add a title
    plt.title('Prediction Error of Polynomial Regression Model', size=16)
    # add a legend
    plt.legend(loc=1)
    # show the plot
    plt.show()