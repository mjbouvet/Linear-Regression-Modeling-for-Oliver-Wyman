import pandas as pd
import numpy as np
import seaborn as sns #visualization
import matplotlib.pyplot as plt #visualization
import statsmodels.api as sm
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew

sns.set(color_codes=True)

import datetime

#Import Files
casesData = pd.read_csv("H:/Documents/Oliver Wyman Interview/COVID Cases.csv")
mobilityIndex = pd.read_csv("H:/Documents/Oliver Wyman Interview/Mobility Index.csv")
smallBusinessData = pd.read_csv("H:/Documents/Oliver Wyman Interview/Small Business Data.csv")
unemploymentData = pd.read_csv("H:/Documents/Oliver Wyman Interview/Unemployment.csv")

#Get Types of Data
print("CasesData:", '\n', casesData.dtypes, '\n')
print("MobilityIndex:", '\n', mobilityIndex.dtypes, '\n')
print("smallBusinessData:", '\n', smallBusinessData.dtypes, '\n')
print("unemploymentData:", '\n', unemploymentData.dtypes, '\n')

#Limit to Entries from North Carolina
casesData_NC = casesData[casesData["State"] == "NC"].copy()
mobilityIndex_NC = mobilityIndex[mobilityIndex["State"] == "NC"].copy()
smallBusinessData_NC = smallBusinessData[smallBusinessData["State"]=="NC"].copy()
unemploymentData_NC = unemploymentData[unemploymentData["State"]=="NC"].copy()

#Check for Duplicate Rows
print("CasesData Shape:", casesData_NC.shape)
duplicate_rows_casesData = casesData_NC[casesData_NC.duplicated()]
print("number of duplicate rows: ", duplicate_rows_casesData.shape, '\n')

print("MobilityIndex Shape:",mobilityIndex_NC.shape)
duplicate_rows_mobilityIndex = mobilityIndex_NC[mobilityIndex_NC.duplicated()]
print("number of duplicate rows: ", duplicate_rows_mobilityIndex.shape, '\n')

print("smallBusinessData Shape:", smallBusinessData_NC.shape)
duplicate_rows_smallBusinessData = smallBusinessData_NC[smallBusinessData_NC.duplicated()]
print("number of duplicate rows: ", duplicate_rows_smallBusinessData.shape, '\n')

print("unemploymentData Shape", unemploymentData_NC.shape)
duplicate_rows_unemploymentData = unemploymentData_NC[unemploymentData_NC.duplicated()]
print("number of duplicate rows: ", duplicate_rows_unemploymentData.shape, '\n')

#Check for Null Values
casesData_NC.loc[casesData_NC.case_rate == '.', 'case_rate'] = np.nan
casesData_NC.loc[casesData_NC.new_case_rate == '.', 'new_case_rate'] = np.nan

print(casesData_NC.isnull().sum())
mean_value = pd.to_numeric(casesData_NC['new_case_rate']).mean()
casesData_NC = casesData_NC.fillna(mean_value)
print(casesData_NC.isnull().sum())

mobilityIndex_NC.loc[mobilityIndex_NC.gps_retail_and_recreation == '.', 'gps_retail_and_recreation'] = np.nan
mobilityIndex_NC.loc[mobilityIndex_NC.gps_grocery_and_pharmacy == '.', 'gps_grocery_and_pharmacy'] = np.nan
mobilityIndex_NC.loc[mobilityIndex_NC.gps_parks == '.', 'gps_parks'] = np.nan
mobilityIndex_NC.loc[mobilityIndex_NC.gps_transit_stations == '.', 'gps_transit_stations'] = np.nan
mobilityIndex_NC.loc[mobilityIndex_NC.gps_workplaces == '.', 'gps_workplaces'] = np.nan
mobilityIndex_NC.loc[mobilityIndex_NC.gps_residential == '.', 'gps_residential'] = np.nan
mobilityIndex_NC.loc[mobilityIndex_NC.gps_away_from_home == '.', 'gps_away_from_home'] = np.nan

print(mobilityIndex_NC.isnull().sum())
mean_value = pd.to_numeric(mobilityIndex_NC['gps_retail_and_recreation']).mean()
mobilityIndex_NC['gps_retail_and_recreation'] = mobilityIndex_NC['gps_retail_and_recreation'].fillna(mean_value)
mean_value = pd.to_numeric(mobilityIndex_NC['gps_grocery_and_pharmacy']).mean()
mobilityIndex_NC['gps_grocery_and_pharmacy'] = mobilityIndex_NC['gps_grocery_and_pharmacy'].fillna(mean_value)
mean_value = pd.to_numeric(mobilityIndex_NC['gps_parks']).mean()
mobilityIndex_NC['gps_parks'] = mobilityIndex_NC['gps_parks'].fillna(mean_value)
mean_value = pd.to_numeric(mobilityIndex_NC['gps_transit_stations']).mean()
mobilityIndex_NC['gps_transit_stations'] = mobilityIndex_NC['gps_transit_stations'].fillna(mean_value)
print(mobilityIndex_NC.isnull().sum())

smallBusinessData_NC.loc[smallBusinessData_NC['RevenueChange'] == '.', 'RevenueChange'] = np.nan
print(smallBusinessData_NC.isnull().sum())

unemploymentData_NC.loc[unemploymentData_NC['initclaims_rate'] == '.', 'initclaims_rate'] = np.nan
unemploymentData_NC.loc[unemploymentData_NC['contclaims_rate'] == '.', 'contclaims_rate'] = np.nan
print(unemploymentData_NC.isnull().sum())
mean_value = pd.to_numeric(unemploymentData_NC['contclaims_rate']).mean()
unemploymentData_NC['contclaims_rate'] = unemploymentData_NC['contclaims_rate'].fillna(mean_value)
print(unemploymentData_NC.isnull().sum())


##START OF REGRESSION MODEL
smallBusinessData_NC.index = smallBusinessData_NC.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day']), "%Y %m %d %H:%M:%S"),axis=1)
casesData_NC.index = casesData_NC.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day']), "%Y %m %d %H:%M:%S"),axis=1)
mobilityIndex_NC.index = mobilityIndex_NC.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day']), "%Y %m %d %H:%M:%S"),axis=1)
unemploymentData_NC.index = unemploymentData_NC.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day_endofweek']), "%Y %m %d %H:%M:%S"),axis=1)

print(smallBusinessData_NC.head())
print(casesData_NC.head())

for i in smallBusinessData_NC.index:
    if i not in casesData_NC.index:
        smallBusinessData_NC = smallBusinessData_NC.drop(i)

for i in casesData_NC.index:
    if i not in smallBusinessData_NC.index:
        casesData_NC = casesData_NC.drop(i)

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(casesData_NC['case_rate'])]
headers = ['RevenueChange', 'case_rate']
combinedDataCaseRate = pd.concat(data, axis = 1, keys=headers) #.188412 (WEAK)

print(combinedDataCaseRate.corr())

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(casesData_NC['new_case_rate'])]
headers = ['RevenueChange', 'new_case_rate']
combinedDataNewCaseRate = pd.concat(data, axis = 1, keys=headers) #.156122 (WEAK)

print(combinedDataNewCaseRate.corr())

#RESET
smallBusinessData_NC = smallBusinessData[smallBusinessData["State"]=="NC"].copy()
smallBusinessData_NC.index = smallBusinessData_NC.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day']), "%Y %m %d %H:%M:%S"),axis=1)
##

for i in smallBusinessData_NC.index:
    if i not in mobilityIndex_NC.index:
        smallBusinessData_NC = smallBusinessData_NC.drop(i)

for i in mobilityIndex_NC.index:
    if i not in smallBusinessData_NC.index:
        mobilityIndex_NC = mobilityIndex_NC.drop(i)

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(mobilityIndex_NC['gps_retail_and_recreation'])]
headers = ['RevenueChange', 'gps_retail_and_recreation']
combinedDataRetail = pd.concat(data, axis = 1, keys=headers) #.8889222 (STRONG)

print(combinedDataRetail.corr())

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(mobilityIndex_NC['gps_grocery_and_pharmacy'])]
headers = ['RevenueChange', 'gps_grocery_and_pharmacy']
combinedDataGrocery = pd.concat(data, axis = 1, keys=headers) #.689071 (STRONG)

print(combinedDataGrocery.corr())

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(mobilityIndex_NC['gps_parks'])]
headers = ['RevenueChange', 'gps_parks']
combinedDataGrocery = pd.concat(data, axis = 1, keys=headers) #.591387 (OK)

print(combinedDataGrocery.corr())

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(mobilityIndex_NC['gps_transit_stations'])]
headers = ['RevenueChange', 'gps_transit_stations']
combinedDataTransit = pd.concat(data, axis = 1, keys=headers) #.845719 (STRONG)

print(combinedDataTransit.corr())

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(mobilityIndex_NC['gps_workplaces'])]
headers = ['RevenueChange', 'gps_workplaces']
combinedDataWorkplaces = pd.concat(data, axis = 1, keys=headers) #.70181 (STRONG)

print(combinedDataWorkplaces.corr())

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(mobilityIndex_NC['gps_residential'])]
headers = ['RevenueChange', 'gps_residential']
combinedDataResidential = pd.concat(data, axis = 1, keys=headers) #-.875719 (STRONG)

print(combinedDataResidential.corr())

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(mobilityIndex_NC['gps_away_from_home'])]
headers = ['RevenueChange', 'gps_away_from_home']
combinedDataAway = pd.concat(data, axis = 1, keys=headers) #.894721 (STRONG)

print(combinedDataAway.corr())

#RESET
smallBusinessData_NC = smallBusinessData[smallBusinessData["State"]=="NC"].copy()
smallBusinessData_NC.index = smallBusinessData_NC.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day']), "%Y %m %d %H:%M:%S"),axis=1)
##

for i in smallBusinessData_NC.index:
    if i not in unemploymentData_NC.index:
        smallBusinessData_NC = smallBusinessData_NC.drop(i)

for i in unemploymentData_NC.index:
    if i not in smallBusinessData_NC.index:
        unemploymentData_NC = unemploymentData_NC.drop(i)

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(unemploymentData_NC['initclaims_rate'])]
headers = ['RevenueChange', 'initclaims_rate']
combinedDataInitClaims = pd.concat(data, axis = 1, keys=headers) #-.906872 (STRONGEST)

print(combinedDataInitClaims.corr())

data = [smallBusinessData_NC['RevenueChange'], pd.to_numeric(unemploymentData_NC['contclaims_rate'])]
headers = ['RevenueChange', 'contclaims_rate']
combinedDataContClaims = pd.concat(data, axis = 1, keys=headers) #-.272933 (WEAK)

print(combinedDataContClaims.corr())

#WE CHOOSE TO USE CASE_RATE FOR CASE_DATA
#WE CHOSE TO USE GPS_AWAY_FROM_HOME, GPS_RETAIL_AND_RECREATION, AND GPS_TRANSIT_STATION FROM MOBILITYINDEX
#WE CHOOSE TO USE INIT_CLAIMS_RATE FOR UNEMPLOYMENT

#DESCRIBING DATA SETS
print(combinedDataCaseRate.describe())

print(combinedDataAway.describe())
print(combinedDataRetail.describe())
print(combinedDataTransit.describe())

print(combinedDataInitClaims.describe())

#CHECK FOR OUTLIERS AND SKEWNESS
combinedDataInitClaims.hist(grid = False, color = 'cadetblue')
plt.show()

#CALCULATE THE KURTOSIS
initClaimsKurtosis = kurtosis(combinedDataInitClaims['initclaims_rate'], fisher = True)
revenueChangeKurtosis = kurtosis(combinedDataInitClaims['RevenueChange'], fisher = True)
print("The Init_Claims Kurtosis: ", initClaimsKurtosis)
print("The Revenue Change Kurtosis: ", revenueChangeKurtosis)

#CALCULATE THE SKEW
initClaimsSkew = skew(combinedDataInitClaims['initclaims_rate'])
revenueChangeSkew = skew(combinedDataInitClaims['RevenueChange'])
print("The Init_Claims Skew: ", initClaimsSkew)
print("The Revenue Change Skew: ", revenueChangeSkew) #BOTH ARE MODERATELY SKEWED 1/2 TO 1 (abs)

#BUILD THE MODEL

#DEFINE OUR INPUT VARIABLE (X) and OUT VARIABLE
Y = combinedDataInitClaims.drop('initclaims_rate', axis =1)
X = combinedDataInitClaims.drop('RevenueChange', axis=1)

#SPLIT X and Y into X_
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .20, random_state = 1)

#CREATE A LINEAR REGRESSION MODEL OBJECT
regression_model = LinearRegression()

#Pass through the X_train and Y_train data set
regression_model.fit(X_train, y_train)
intercept = regression_model.intercept_[0]
coefficent = regression_model.coef_[0][0]
print("The coefficent for our model is {:.2}".format(coefficent))
print("The intercept for our model is {:.4}".format(intercept))

#Taking a Single Prediction
prediction = regression_model.predict([[1.07]])
predicted_value = prediction[0][0]
print("The predicted value is {:.4}".format(predicted_value))

#Get Multiple Predictions
y_predict = regression_model.predict(X_test)

#Show the first 5 predictions
print(y_predict[:5])

#EVALUATING THE MODEL

#define our input
X2 = sm.add_constant(X)

#create an OLS model
model = sm.OLS(Y, X2)

#fit the data
est = model.fit()

#CONFIDENCE INTERVAL

#95 percent by default
print(est.conf_int())

#Hypothesis Testings
#Null: There is no relationship between the revenue change and init_Claims, and coef does equal 0
#ALternative: There is a relationship, and coefficient does not equal 0

print(est.pvalues)
#Both pvalues are less than .05, more specifically the coefficient for initclaims is close to 0 so less than .05 so we can reject our null hypothesis meaning there is a relationship

#Calculate the Mean Sqaured Error
model_mse = mean_squared_error(y_test, y_predict)

#Calculate the Mean Absolute Error
model_mae = mean_absolute_error(y_test, y_predict)

#Calculate the Root Mean Squared Error
model_rmse = math.sqrt(model_mse)

#display
print("MSE {:.3}".format(model_mse))
print("MSE {:.3}".format(model_mae))
print("MSE {:.3}".format(model_rmse))


#PRINT OUT A SUMMARY
print(est.summary())

#PLOT THE RESIDUALS
(y_test - y_predict).hist(grid = False, color = 'royalblue')
plt.title("Model Residuals")
plt.show()

#PLOTTING OUR LINE

#Plot Outputs
plt.scatter(X_test, y_test, color = 'red', label = 'InitClaim_rate')
plt.plot(X_test,y_predict, color = 'royalBlue', linewidth = 3, linestyle = '-', label = 'Regression Line')

plt.title("Linear Regression Model InitClaims_Rate vs. Revenue Change")
plt.xlabel("initclaims_rate")
plt.ylabel("Revenue Change")
plt.legend()
plt.show()


#PICKLE THE MODEL
import pickle
with open('my_linear_regression.sav', 'wb') as f:
    pickle.dump(regression_model,f)

with open('my_linear_regression.sav', 'rb') as f:
    regression_model_2 = pickle.load(f)

#make a prediction
print(regression_model_2.predict([[-.1308]]))


#DESCRIBING DATA SETS PART 2
print(combinedDataCaseRate.describe())

print(combinedDataAway.describe())
print(combinedDataRetail.describe())
print(combinedDataTransit.describe())

print(combinedDataInitClaims.describe())

#CHECK FOR OUTLIERS AND SKEWNESS PART 2
combinedDataAway.hist(grid = False, color = 'cadetblue')
plt.show()

#CALCULATE THE KURTOSIS PART 2
dataAwayKurtosis = kurtosis(combinedDataAway['gps_away_from_home'], fisher = True)
revenueChangeKurtosis = kurtosis(combinedDataInitClaims['RevenueChange'], fisher = True)
print("The gps_away_from_home Kurtosis: ", dataAwayKurtosis)
print("The Revenue Change Kurtosis: ", revenueChangeKurtosis)

#CALCULATE THE SKEW PART 2
dataAwaySkew = skew(combinedDataAway['gps_away_from_home'])
revenueChangeSkew = skew(combinedDataInitClaims['RevenueChange'])
print("The Data Away Skew: ",dataAwaySkew) #ESSENTIALLY NOT SKEWED (CLOSE TO 0)
print("The Revenue Change Skew: ", revenueChangeSkew) #BOTH ARE MODERATELY SKEWED 1/2 TO 1 (abs)

#BUILD THE MODEL PART 2

#DEFINE OUR INPUT VARIABLE (X) and OUT VARIABLE
Y = combinedDataAway.drop('gps_away_from_home', axis =1)
X = combinedDataAway.drop('RevenueChange', axis=1)

#SPLIT X and Y into X_
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .20, random_state = 1)

#CREATE A LINEAR REGRESSION MODEL OBJECT
regression_model = LinearRegression()

#Pass through the X_train and Y_train data set
regression_model.fit(X_train, y_train)
intercept = regression_model.intercept_[0]
coefficent = regression_model.coef_[0][0]
print("The coefficent for our model is {:.2}".format(coefficent))
print("The intercept for our model is {:.4}".format(intercept))

#Taking a Single Prediction
prediction = regression_model.predict([[1.07]])
predicted_value = prediction[0][0]
print("The predicted value is {:.4}".format(predicted_value))

#Get Multiple Predictions
y_predict = regression_model.predict(X_test)

#Show the first 5 predictions
print(y_predict[:5])

#EVALUATING THE MODEL

#define our input
X2 = sm.add_constant(X)

#create an OLS model
model = sm.OLS(Y, X2)

#fit the data
est = model.fit()

#CONFIDENCE INTERVAL

#95 percent by default
print(est.conf_int())

#Hypothesis Testings
#Null: There is no relationship between the revenue change and init_Claims, and coef does equal 0
#ALternative: There is a relationship, and coefficient does not equal 0

print(est.pvalues)
#Both pvalues are less than .05, more specifically the coefficient for initclaims is close to 0 so less than .05 so we can reject our null hypothesis meaning there is a relationship

#Calculate the Mean Sqaured Error
model_mse = mean_squared_error(y_test, y_predict)

#Calculate the Mean Absolute Error
model_mae = mean_absolute_error(y_test, y_predict)

#Calculate the Root Mean Squared Error
model_rmse = math.sqrt(model_mse)

#display
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


model_r2 = r2_score(y_test, y_predict)
print("R2 {:.2}".format(model_r2))

#PRINT OUT A SUMMARY
print(est.summary())

#PLOT THE RESIDUALS
(y_test - y_predict).hist(grid = False, color = 'royalblue')
plt.title("Model Residuals")
plt.show()

#PLOTTING OUR LINE

#Plot Outputs
plt.scatter(X_test, y_test, color = 'red', label = 'gps_away_from_home')
plt.plot(X_test,y_predict, color = 'royalBlue', linewidth = 3, linestyle = '-', label = 'Regression Line')

plt.title("Linear Regression Model gps_away_from_home vs. Revenue Change")
plt.xlabel("gps_away_from_home")
plt.ylabel("Revenue Change")
plt.legend()
plt.show()


#PICKLE THE MODEL
import pickle
with open('my_linear_regression.sav', 'wb') as f:
    pickle.dump(regression_model,f)

with open('my_linear_regression.sav', 'rb') as f:
    regression_model_2 = pickle.load(f)

#make a prediction
print(regression_model_2.predict([[2.1]]))