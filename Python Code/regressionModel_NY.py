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
casesData_NY = casesData[casesData["State"] == "NY"].copy()
mobilityIndex_NY = mobilityIndex[mobilityIndex["State"] == "NY"].copy()
smallBusinessData_NY = smallBusinessData[smallBusinessData["State"]=="NY"].copy()
unemploymentData_NY = unemploymentData[unemploymentData["State"]=="NY"].copy()

#Check for Duplicate Rows
print("CasesData Shape:", casesData_NY.shape)
duplicate_rows_casesData = casesData_NY[casesData_NY.duplicated()]
print("number of duplicate rows: ", duplicate_rows_casesData.shape, '\n')

print("MobilityIndex Shape:",mobilityIndex_NY.shape)
duplicate_rows_mobilityIndex = mobilityIndex_NY[mobilityIndex_NY.duplicated()]
print("number of duplicate rows: ", duplicate_rows_mobilityIndex.shape, '\n')

print("smallBusinessData Shape:", smallBusinessData_NY.shape)
duplicate_rows_smallBusinessData = smallBusinessData_NY[smallBusinessData_NY.duplicated()]
print("number of duplicate rows: ", duplicate_rows_smallBusinessData.shape, '\n')

print("unemploymentData Shape", unemploymentData_NY.shape)
duplicate_rows_unemploymentData = unemploymentData_NY[unemploymentData_NY.duplicated()]
print("number of duplicate rows: ", duplicate_rows_unemploymentData.shape, '\n')

#Check for Null Values
print(casesData_NY.head(10))

casesData_NY.loc[casesData_NY.case_rate == '.', 'case_rate'] = np.nan
casesData_NY.loc[casesData_NY.new_case_rate == '.', 'new_case_rate'] = np.nan

print(casesData_NY.isnull().sum())
print(casesData_NY.head(10))
mean_value = pd.to_numeric(casesData_NY['new_case_rate']).mean()
casesData_NY = casesData_NY.fillna(mean_value)
print(casesData_NY.isnull().sum())
print(casesData_NY.head(10))

mobilityIndex_NY.loc[mobilityIndex_NY.gps_retail_and_recreation == '.', 'gps_retail_and_recreation'] = np.nan
mobilityIndex_NY.loc[mobilityIndex_NY.gps_grocery_and_pharmacy == '.', 'gps_grocery_and_pharmacy'] = np.nan
mobilityIndex_NY.loc[mobilityIndex_NY.gps_parks == '.', 'gps_parks'] = np.nan
mobilityIndex_NY.loc[mobilityIndex_NY.gps_transit_stations == '.', 'gps_transit_stations'] = np.nan
mobilityIndex_NY.loc[mobilityIndex_NY.gps_workplaces == '.', 'gps_workplaces'] = np.nan
mobilityIndex_NY.loc[mobilityIndex_NY.gps_residential == '.', 'gps_residential'] = np.nan
mobilityIndex_NY.loc[mobilityIndex_NY.gps_away_from_home == '.', 'gps_away_from_home'] = np.nan

print(mobilityIndex_NY.isnull().sum())
mean_value = pd.to_numeric(mobilityIndex_NY['gps_retail_and_recreation']).mean()
mobilityIndex_NY['gps_retail_and_recreation'] = mobilityIndex_NY['gps_retail_and_recreation'].fillna(mean_value)
mean_value = pd.to_numeric(mobilityIndex_NY['gps_grocery_and_pharmacy']).mean()
mobilityIndex_NY['gps_grocery_and_pharmacy'] = mobilityIndex_NY['gps_grocery_and_pharmacy'].fillna(mean_value)
mean_value = pd.to_numeric(mobilityIndex_NY['gps_parks']).mean()
mobilityIndex_NY['gps_parks'] = mobilityIndex_NY['gps_parks'].fillna(mean_value)
mean_value = pd.to_numeric(mobilityIndex_NY['gps_transit_stations']).mean()
mobilityIndex_NY['gps_transit_stations'] = mobilityIndex_NY['gps_transit_stations'].fillna(mean_value)
print(mobilityIndex_NY.isnull().sum())

smallBusinessData_NY.loc[smallBusinessData_NY['RevenueChange'] == '.', 'RevenueChange'] = np.nan
print(smallBusinessData_NY.isnull().sum())

unemploymentData_NY.loc[unemploymentData_NY['initclaims_rate'] == '.', 'initclaims_rate'] = np.nan
unemploymentData_NY.loc[unemploymentData_NY['contclaims_rate'] == '.', 'contclaims_rate'] = np.nan
print(unemploymentData_NY.isnull().sum())
mean_value = pd.to_numeric(unemploymentData_NY['contclaims_rate']).mean()
unemploymentData_NY['contclaims_rate'] = unemploymentData_NY['contclaims_rate'].fillna(mean_value)
print(unemploymentData_NY.isnull().sum())

smallBusinessData_NY.index = smallBusinessData_NY.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day']), "%Y %m %d %H:%M:%S"),axis=1)
casesData_NY.index = casesData_NY.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day']), "%Y %m %d %H:%M:%S"),axis=1)
mobilityIndex_NY.index = mobilityIndex_NY.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day']), "%Y %m %d %H:%M:%S"),axis=1)
unemploymentData_NY.index = unemploymentData_NY.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day_endofweek']), "%Y %m %d %H:%M:%S"),axis=1)

print(smallBusinessData_NY.head())
print(casesData_NY.head())

for i in smallBusinessData_NY.index:
    if i not in casesData_NY.index:
        smallBusinessData_NY = smallBusinessData_NY.drop(i)

for i in casesData_NY.index:
    if i not in smallBusinessData_NY.index:
        casesData_NY = casesData_NY.drop(i)

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(casesData_NY['case_rate'])]
headers = ['RevenueChange', 'case_rate']
combinedDataCaseRate = pd.concat(data, axis = 1, keys=headers) #.511935 (OK)

print(combinedDataCaseRate.corr())

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(casesData_NY['new_case_rate'])]
headers = ['RevenueChange', 'new_case_rate']
combinedDataNewCaseRate = pd.concat(data, axis = 1, keys=headers) #-.691218 (STRONG)

print(combinedDataNewCaseRate.corr())

#RESET
smallBusinessData_NY = smallBusinessData[smallBusinessData["State"]=="NY"].copy()
smallBusinessData_NY.index = smallBusinessData_NY.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day']), "%Y %m %d %H:%M:%S"),axis=1)
##

for i in smallBusinessData_NY.index:
    if i not in mobilityIndex_NY.index:
        smallBusinessData_NY = smallBusinessData_NY.drop(i)

for i in mobilityIndex_NY.index:
    if i not in smallBusinessData_NY.index:
        mobilityIndex_NY = mobilityIndex_NY.drop(i)

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(mobilityIndex_NY['gps_retail_and_recreation'])]
headers = ['RevenueChange', 'gps_retail_and_recreation']
combinedDataRetail = pd.concat(data, axis = 1, keys=headers) #955133 (STRONG)

print(combinedDataRetail.corr())

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(mobilityIndex_NY['gps_grocery_and_pharmacy'])]
headers = ['RevenueChange', 'gps_grocery_and_pharmacy']
combinedDataGrocery = pd.concat(data, axis = 1, keys=headers) #.787008 (STRONG)

print(combinedDataGrocery.corr())

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(mobilityIndex_NY['gps_parks'])]
headers = ['RevenueChange', 'gps_parks']
combinedDataGrocery = pd.concat(data, axis = 1, keys=headers) #.418492 (OK)

print(combinedDataGrocery.corr())

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(mobilityIndex_NY['gps_transit_stations'])]
headers = ['RevenueChange', 'gps_transit_stations']
combinedDataTransit = pd.concat(data, axis = 1, keys=headers) #.960822 (STRONG)

print(combinedDataTransit.corr())

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(mobilityIndex_NY['gps_workplaces'])]
headers = ['RevenueChange', 'gps_workplaces']
combinedDataWorkplaces = pd.concat(data, axis = 1, keys=headers) #.962474 (STRONG)

print(combinedDataWorkplaces.corr())

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(mobilityIndex_NY['gps_residential'])]
headers = ['RevenueChange', 'gps_residential']
combinedDataResidential = pd.concat(data, axis = 1, keys=headers) #-.968317 (STRONGEST)

print(combinedDataResidential.corr())

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(mobilityIndex_NY['gps_away_from_home'])]
headers = ['RevenueChange', 'gps_away_from_home']
combinedDataAway = pd.concat(data, axis = 1, keys=headers) #.9623072 (STRONG)

print(combinedDataAway.corr())

#RESET
smallBusinessData_NY = smallBusinessData[smallBusinessData["State"]=="NY"].copy()
smallBusinessData_NY.index = smallBusinessData_NY.apply(lambda x:datetime.datetime.strptime("{0} {1} {2} 00:00:00".format(x['year'],x['month'], x['day']), "%Y %m %d %H:%M:%S"),axis=1)
##

for i in smallBusinessData_NY.index:
    if i not in unemploymentData_NY.index:
        smallBusinessData_NY = smallBusinessData_NY.drop(i)

for i in unemploymentData_NY.index:
    if i not in smallBusinessData_NY.index:
        unemploymentData_NY = unemploymentData_NY.drop(i)

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(unemploymentData_NY['initclaims_rate'])]
headers = ['RevenueChange', 'initclaims_rate']
combinedDataInitClaims = pd.concat(data, axis = 1, keys=headers) #-.831587 (STRONG)

print(combinedDataInitClaims.corr())

data = [smallBusinessData_NY['RevenueChange'], pd.to_numeric(unemploymentData_NY['contclaims_rate'])]
headers = ['RevenueChange', 'contclaims_rate']
combinedDataContClaims = pd.concat(data, axis = 1, keys=headers) #-.506223 (OK)

print(combinedDataContClaims.corr())

#DESCRIBING DATA SETS PART 2
print(combinedDataResidential.describe())

#CHECK FOR OUTLIERS AND SKEWNESS PART 2
combinedDataResidential.hist(grid = False, color = 'cadetblue')
plt.show()

#CALCULATE THE KURTOSIS PART 2
dataResidentialKurtosis = kurtosis(combinedDataResidential['gps_residential'], fisher = True)
revenueChangeKurtosis = kurtosis(combinedDataInitClaims['RevenueChange'], fisher = True)
print("The gps_residential Kurtosis: ", dataResidentialKurtosis)
print("The Revenue Change Kurtosis: ", revenueChangeKurtosis)

#CALCULATE THE SKEW PART 2
dataResidentialSkew = skew(combinedDataResidential['gps_residential'])
revenueChangeSkew = skew(combinedDataInitClaims['RevenueChange'])
print("The Data Away Skew: ",dataResidentialSkew) #ESSENTIALLY NOT SKEWED (CLOSE TO 0)
print("The Revenue Change Skew: ", revenueChangeSkew) #BOTH ARE MODERATELY SKEWED 1/2 TO 1 (abs)

#BUILD THE MODEL PART 2

#DEFINE OUR INPUT VARIABLE (X) and OUT VARIABLE
Y = combinedDataResidential.drop('gps_residential', axis =1)
X = combinedDataResidential.drop('RevenueChange', axis=1)

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
plt.scatter(X_test, y_test, color = 'red', label = 'gps_residential')
plt.plot(X_test,y_predict, color = 'royalBlue', linewidth = 3, linestyle = '-', label = 'Regression Line')

plt.title("Linear Regression Model gps_residential vs. Revenue Change")
plt.xlabel("gps_residential")
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