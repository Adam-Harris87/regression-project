# import libraries to work with arrays and dataframes
import numpy as np
import pandas as pd
# import math functions
import math
from scipy.stats import pearsonr
# import visualization tools
import matplotlib.pyplot as plt
import seaborn as sns


def check_p(p, r, α=0.05):
    if p < α:
        print('there is sufficient evidence to reject our null hypothesis')
        print(f'the p-value is {p}')
        print(f'the r coeficient is {r}')
    else:
        print('we fail to reject our null hypothesis')
        print(f'the p-value is {p}')

def get_cars_value(train):    
    sns.relplot(data=train, x='cars_garage', y='tax_value', 
                kind='line', ci=False, color='blue')
    plt.axhline(train.tax_value.mean(), color='red')
    plt.annotate('Average Tax Value', (1.8,360000), size=8)
    plt.title('Tax Value by Number of Cars That Fit in Garage', size=14)
    plt.xlabel('Number of Cars That Fit In Garage', size=14)
    plt.ylabel('Tax Valuation (Dollars)', size=14)
    plt.show()

def get_pearson_garage(train):
    r, p = pearsonr(train.garage_sqft, train.tax_value)
    check_p(p,r)

def get_area_value(train):  
    plt.hexbin(data=train, x='area', y='tax_value', gridsize=10, cmap='Blues')
    plt.xlabel('Square Footage of House', size =14)
    plt.ylabel('Tax Value of House (Dollars)', size=14)
    plt.title('Tax Value by Area of House', size=15)
    plt.yticks(ticks=[0,200_000,400_000,600_000,800_000,1_000_000], 
               labels=['0', '200,000', '400,000', '600,000', '800,000', '1,000,000'])
    plt.show()

def get_pearson_area(train):
    r, p = pearsonr(train.area, train.tax_value)
    check_p(p, r)

def get_lot_bin_vis(train):    
    sns.catplot(data=train, x='lot_size_binned', y='tax_value', kind='violin')
    plt.xticks(rotation=90)
    plt.axhline(train.tax_value.mean(), color='red', label='tax value mean')
    plt.title('Tax Value by Property Lot Size', size=15)
    plt.xlabel('Property Lot Size (sqft) in Bins', size=14)
    plt.ylabel('Tax Valuation (Dollars)', size=14)
    plt.legend()
    plt.yticks(ticks=[0,200_000,400_000,600_000,800_000,1_000_000], 
               labels=['0', '200,000', '400,000', '600,000', '800,000', '1,000,000'])
    plt.show()

def get_cars_by_sqft(train):
    sns.relplot(data=train, x='garage_sqft', y='cars_garage', 
                kind='line', ci=False, color='blue')
    plt.title('Number of Cars That Fit in Garage by Garage sqft', size=14)
    plt.ylabel('Number of Cars That Fit In Garage', size=14)
    plt.xlabel('Garage Square Footage', size=14)
    plt.show()