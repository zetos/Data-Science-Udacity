# -*- coding: utf-8 -*-

import numpy as np
import pandas
import scipy
import statsmodels.api as sm
from datetime import datetime

"""
In this optional exercise, you should complete the function called 
predictions(turnstile_weather). This function takes in our pandas 
turnstile weather dataframe, and returns a set of predicted ridership values,
based on the other information in the dataframe.  

In exercise 3.5 we used Gradient Descent in order to compute the coefficients
theta used for the ridership prediction. Here you should attempt to implement 
another way of computing the coeffcients theta. You may also try using a reference implementation such as: 
http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.OLS.html

One of the advantages of the statsmodels implementation is that it gives you
easy access to the values of the coefficients theta. This can help you infer relationships 
between variables in the dataset.

You may also experiment with polynomial terms as part of the input variables.  

The following links might be useful: 
http://en.wikipedia.org/wiki/Ordinary_least_squares
http://en.wikipedia.org/w/index.php?title=Linear_least_squares_(mathematics)
http://en.wikipedia.org/wiki/Polynomial_regression

This is your playground. Go wild!

How does your choice of linear regression compare to linear regression
with gradient descent computed in Exercise 3.5?

You can look at the information contained in the turnstile_weather dataframe below:
https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/turnstile_data_master_with_weather.csv

Note: due to the memory and CPU limitation of our amazon EC2 instance, we will
give you a random subset (~10%) of the data contained in turnstile_data_master_with_weather.csv

If you receive a "server has encountered an error" message, that means you are hitting 
the 30 second limit that's placed on running your program. See if you can optimize your code so it
runs faster.
"""

def predictions(weather_turnstile):
    #
    # Your implementation goes here. Feel free to write additional
    # helper functions
    # 
    
    dummy_units = pandas.get_dummies(weather_turnstile['UNIT'], prefix='unit')
    
    weather_turnstile['WEEKDAYn'] = weather_turnstile['DATEn'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday())
    weather_turnstile['WEEKDAYn'] = weather_turnstile['WEEKDAYn'].apply(lambda x: 0 if x in [5, 6] else 1)
    weather_turnstile['PEAKn'] = weather_turnstile['Hour'].apply(lambda x: 1 if x in [9, 12, 13, 16, 17, 20, 21, 0] else 0)
    
    
    dependent_Y = weather_turnstile[['ENTRIESn_hourly']]
    features_X = weather_turnstile[['rain', 'Hour', 'PEAKn', 'WEEKDAYn',
                           'maxtempi', 'mintempi']].join(dummy_units)
    features_X = sm.add_constant(features_X)
    
    est = sm.OLS(dependent_Y, features_X).fit()
    prediction = est.predict(features_X)
    
    return prediction
