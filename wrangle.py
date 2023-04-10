# import libraries to work with arrays and dataframes
import numpy as np
import pandas as pd
# import math functions
import math
# import visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
# import file manipulation tools
import os
import env
# import data prep tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def acquire_zillow_sfr():
    '''
    This function will retrieve zillow home data for 2017 properties. It will only get
    single family residential properties. the function will attempt to open the data from 
    a local csv file, if one is not found, it will download the data from the codeup
    database. An env file is needed in the local directory in order to run this file.
    '''
    if os.path.exists('zillow_2017_sfr.csv'):
        print('opening data from local file')
        df = pd.read_csv('zillow_2017_sfr.csv', index_col=0)
    else:
        # run sql query and write to csv
        print('local file not found')
        print('retrieving data from sql server')
        query = '''
WITH cte_sfr as(
	SELECT * 
    FROM properties_2017
    WHERE propertylandusetypeid IN (
		SELECT propertylandusetypeid
		FROM propertylandusetype
		WHERE propertylandusedesc IN( 
			"Single Family Residential", "Inferred Single Family Residential"))
	AND parcelid IN (
		SELECT parcelid
        FROM predictions_2017)
)

SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,
	garagecarcnt, garagetotalsqft, lotsizesquarefeet, poolcnt,
	yearbuilt, fips, regionidcity, taxvaluedollarcnt
FROM cte_sfr
;
        '''
        connection = env.get_db_url('zillow')
        df = pd.read_sql(query, connection)
        df.to_csv('zillow_2017_sfr.csv')
    
    # renaming column names to one's I like better
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'garagecarcnt':'cars_garage',
                              'garagetotalsqft':'garage_sqft',
                              'lotsizesquarefeet':'lot_size',
                              'poolcnt':'pools',
                              'regionidcity':'region',
                              'yearbuilt':'year_built',
                              'taxvaluedollarcnt':'tax_value'
                              })
    return df

def remove_outliers(df, col_list, k=1.5):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df

def clean_zillow_sfr(df, 
                     out_cols=['bedrooms', 'bathrooms', 'area', 
                              'tax_value']):
    '''
    this function will take in a DataFrame of zillow single family resident data,
    it will then remove rows will null values, then remove rows with 0 bedrooms or 
    0 bathrooms, it will then change dtypes of bedroomcnt, calculatedfinishedsquarefeet,
    taxvaluedollarcnt, yearbuilt, and fips to integer, then return the cleaned df
    '''
    # remove outliers from basic info columns 
    df = remove_outliers(df, out_cols)
    
    # fill in nan values in garage_sqft with 0
    garage_imputer = SimpleImputer(strategy='constant', fill_value=0)
    garage_imputer.fit(df[['garage_sqft']])
    df.garage_sqft = garage_imputer.transform(df[['garage_sqft']])
    # if garage_sqft is 0 then the cars that fit inside it will also be 0
    df.cars_garage = np.where(df.garage_sqft == 0, 0, df.cars_garage)
    
    # fill the pools nan with 0s
    pool_imputer = SimpleImputer(strategy='constant', fill_value=0)
    pool_imputer.fit(df[['pools']])
    df.pools = pool_imputer.transform(df[['pools']])

    # lets put lot_size into bins so there aren't so many outliers
    bins = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,np.inf]
    df['lot_size_binned'] = pd.cut(df.lot_size, bins)
    
    # return the cleaned dataFrame
    return df

def split_zillow(df):
    '''
    this function will take in a cleaned zillow dataFrame and return the data split into
    train, validate and test dataframes in preparation for ml modeling.
    '''
    train_val, test = train_test_split(df,
                                      random_state=1342,
                                      train_size=0.8)
    train, validate = train_test_split(train_val,
                                      random_state=1342,
                                      train_size=0.7)
    return train, validate, test

def wrangle_zillow():
    '''
    This function will acquire the zillow dataset, clean the data, and split it
    and return the data as train, validate, test
    '''
    return split_zillow(
        clean_zillow_sfr(
            acquire_zillow_sfr()))

def impute_region(train, validate, test):
    # fill in the region blanks with the most common region_id
    # by each fips code
    region_imputer = SimpleImputer(strategy='most_frequent')
    
    region_imputer.fit(train[train.fips == 6037])
    train[train.fips == 6037] = region_imputer.transform(
        train[train.fips == 6037])
    validate[validate.fips == 6037] = region_imputer.transform(
        validate[validate.fips == 6037])
    test[test.fips == 6037] = region_imputer.transform(
        test[test.fips == 6037])
    
    region_imputer.fit(train[train.fips == 6059])
    train[train.fips == 6059] = region_imputer.transform(
        train[train.fips == 6059])
    validate[validate.fips == 6059] = region_imputer.transform(
        validate[validate.fips == 6059])
    test[test.fips == 6059] = region_imputer.transform(
        test[test.fips == 6059])
    
    region_imputer.fit(train[train.fips == 6111])
    train[train.fips == 6111] = region_imputer.transform(
        train[train.fips == 6111])
    validate[validate.fips == 6111] = region_imputer.transform(
        validate[validate.fips == 6111])
    test[test.fips == 6111] = region_imputer.transform(
        test[test.fips == 6111])
    
    
    # converting column datatypes
    # change dtypes of columns to int
    int_list = train.drop(columns=['bathrooms', 'lot_size_binned']).columns.to_list()
    for col in int_list:
        train[col] = train[col].astype(int)
        validate[col] = validate[col].astype(int)
        test[col] = test[col].astype(int)


def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedrooms', 'bathrooms', 'area', 
                                 'cars_garage', 'garage_sqft', 'year_built'],
               scaler=MinMaxScaler(),
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(train[columns_to_scale]),
        columns=train[columns_to_scale].columns.values, 
        index = train.index)
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(validate[columns_to_scale]),
        columns=validate[columns_to_scale].columns.values).set_index(
        [validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(
        test[columns_to_scale]), 
        columns=test[columns_to_scale].columns.values).set_index(
        [test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
def impute_and_scale(train, validate, test):
    impute_region(train, validate, test)
    train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test)
    return train_scaled, validate_scaled, test_scaled