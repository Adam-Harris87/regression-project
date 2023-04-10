import pandas as pd
import numpy as np

# modeling methods
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE

def get_baseline_model(y_train):
    y_train['tax_value_pred_mean'] = y_train.tax_value.mean()
    rmse_train_mu = mean_squared_error(y_train.tax_value,
                                   y_train.tax_value_pred_mean, squared=False)
    print('Baseline Model (mean)')
    print(f'RMSE for baseline model: {rmse_train_mu:.08}')
    print('R^2 for baseline model: 0.0')


def get_lars_model(X_train_scaled, y_train, 
                   X_validate_scaled, y_validate, f_features):    
    # make la thing
    lars = LassoLars(alpha=0.1)
    # fit za thing
    lars.fit(X_train_scaled[f_features], y_train.tax_value)
    # usage of a thing
    y_train['tax_value_pred_lars'] = lars.predict(X_train_scaled[f_features])
    # Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars) ** .5
    
    # repeat usage on validate
    y_validate['tax_value_pred_lars'] = lars.predict(X_validate_scaled[f_features])
    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_validate.tax_value, 
                                       y_validate.tax_value_pred_lars) ** .5
    r_2 = explained_variance_score(y_validate.tax_value,
                                       y_validate.tax_value_pred_lars)
    
    print('Lasso + Lars Model')
    print(f'RMSE on training data: {rmse_train:.08}')
    print(f'RMSE on validation data: {rmse_validate:.08}')
    print(f'Difference in RMSE: {rmse_validate - rmse_train:.08}')
    print(f'R^2 value: {r_2:0.4}')

    return lars


def get_tweedie_model(X_train_scaled, y_train, 
                      X_validate_scaled, y_validate, f_features):
    # make la thing
    glm = TweedieRegressor(power=1, alpha=0)
    # fit za thing
    glm.fit(X_train_scaled[f_features], y_train.tax_value)
    # usage of a thing
    y_train['tax_value_pred_glm'] = glm.predict(X_train_scaled[f_features])
    # Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_glm) ** .5

    # repeat usage on validate
    y_validate['tax_value_pred_glm'] = glm.predict(X_validate_scaled[f_features])
    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_validate.tax_value, 
                                       y_validate.tax_value_pred_glm) ** .5
    
    r_2 = explained_variance_score(y_validate.tax_value,
                                       y_validate.tax_value_pred_glm)
    
    print('Tweedie Model')
    print(f'RMSE on training data: {rmse_train:.08}')
    print(f'RMSE on validation data: {rmse_validate:.08}')
    print(f'Difference in RMSE: {rmse_validate - rmse_train:.08}')
    print(f'R^2 value: {r_2:0.4}')

    return glm


def get_polynomial_model(X_train_scaled, y_train, 
                         X_validate_scaled, y_validate,
                         X_test_scaled,
                         f_features):
    #1. Create the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2) #quadratic function

    #1. Fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled[f_features])

    #1. Transform X_validate_scaled & X_test_scaled 
    X_validate_degree2 = pf.fit_transform(X_validate_scaled[f_features])
    X_test_degree2 = pf.fit_transform(X_test_scaled[f_features])

    # make la thing
    lm2 = LinearRegression()
    # fit za thing
    lm2.fit(X_train_degree2, y_train.tax_value)
    # usage of a thing
    y_train['tax_value_pred_lm2'] = lm2.predict(X_train_degree2)
    # Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.tax_value, 
                                    y_train.tax_value_pred_lm2) ** .5

    # repeat usage on validate
    y_validate['tax_value_pred_lm2'] = lm2.predict(X_validate_degree2)
    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_validate.tax_value, 
                                       y_validate.tax_value_pred_lm2) ** .5
    
    r_2 = explained_variance_score(y_validate.tax_value,
                                       y_validate.tax_value_pred_lm2)
    
    print('Polynomial Model')
    print(f'RMSE on training data: {rmse_train:.08}')
    print(f'RMSE on validation data: {rmse_validate:.08}')
    print(f'Difference in RMSE: {rmse_validate - rmse_train:.08}')
    print(f'R^2 value: {r_2:0.4}')

    return lm2, X_test_degree2

def get_polynomial_test(lm2, X_test_degree2, y_test):
    y_test['tax_value_pred_lm2'] = lm2.predict(X_test_degree2)
    # Evaluate: RMSE
    rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm2) ** .5
    
    r_2 = explained_variance_score(y_test.tax_value,
                                       y_test.tax_value_pred_lm2)
    
    print('Polynomial Model on Test Data')
    print(f'RMSE on test data: {rmse_test:.08}')
    print(f'R^2 value: {r_2:0.4}')