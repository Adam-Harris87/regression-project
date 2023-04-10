# Project description

In this project we will be looking at a dataset of predicted home tax values from houses that had a transaction in 2017. The data was gathered by Zillow, and acquired from the Codeup database. We will be simulating presenting the project to a panel of data scientists from Zillow.


# Project goals

We will be attempting to improve the tax value predictions for properties that had transactions in 2017.


# Project planning

Planning - The project planning will be laid out in this readme

Acquisition - Data will be acquired via SQL function from the codeup server. Once the data is initially acquired, a telco.csv will be created in the user's local directory and reference instead of the SQL when accessed after the first time.

Preparation - .

Exploration - We will explore the data using various graphs, charts and visualizations in order to see if we can visually determine fields which will have a large impact upon predicting churn rate. We will then perform hypothesis testing in order to check if our visual determination of impactful fields is statistically accurate. We will finally choose a number of fields that have the most correlation to our churn target to use with our machine learning models.

Modeling - We will input our 4-5 most impactful fields into various machine learning algorlithms, and check the accuracy and recall scores of each model. We will consider a customer churn to be a positive condition of our prediction models. 

- Since we are looking at a linear regression problem we will be evaluating our models based on the Root Mean Squared Error (RMSE) value, along with the R$^2$ score.

Delivery - Our delivery method will be an interactive Jupyter notebook containing methodology notes including useful exploration visualisations and recall score metrics from our highest performing models. There will also be a verbal presentation of the findings of this project.


# Initial hypotheses and/or questions you have of the data, ideas

Does tax value correlate to the square footage of the house?
Does tax value correlate to the property's lot size?
Does tax value correlate to the number of cars that fit in the garage?
Does the number of cars that fit in the garage correlate to the size of the garage?


# Data dictionary


# Instructions on how to reproduce the project and findings

In order to run the files in this project, the user will need to connect to the Codeup SQL database, in order to do this the user will need to have a file named 'env.py' in the same file directory as the project files. This env.py file will need to contain: 

- user = 'your_username_to_connect_to_the_codeup_database'
- password = 'your_password_for_the_codeup_database'
- host = 'data.codeup.com'

All project files will need to be located together in the same directory. Run the final_report file to get the finished report.


# Key findings, recommendations, and takeaways from your project.

- Square footage of the house has the strongest correlation to the property's tax value
- Square footage of the garage has a very strong correlation to the number of cars that fit in the garage
- Number of cars that fit in the garage has a moderate correlation to the property tax value
- Square footage of the property's lot has a very weak correlation to the tax value