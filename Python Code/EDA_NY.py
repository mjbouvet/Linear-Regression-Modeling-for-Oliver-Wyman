import pandas as pd
import numpy as np
import seaborn as sns #visualization
import matplotlib.pyplot as plt #visualization
sns.set(color_codes=True)

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
casesData_NC = casesData_NY.fillna(mean_value)
print(casesData_NC.isnull().sum())
print(casesData_NC.head(10))

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


#Looking for Outliers
#Cases Data
sns.boxplot(x = casesData_NY['case_rate'])
plt.show()

Q1 = casesData_NY['case_rate'].quantile(0.25)
Q3 = casesData_NY['case_rate'].quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR CASE_RATE IS:",IQR)

sns.boxplot(x = pd.to_numeric(casesData_NY['new_case_rate']))
plt.show()

Q1 = pd.to_numeric(casesData_NY['new_case_rate']).quantile(0.25)
Q3 = pd.to_numeric(casesData_NY['new_case_rate']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR NEW_CASE_RATE IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NY['gps_retail_and_recreation']))
plt.show()
Q1 = pd.to_numeric(mobilityIndex_NY['gps_retail_and_recreation']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NY['gps_retail_and_recreation']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_RETAIL_AND_RECREATION IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NY['gps_grocery_and_pharmacy']))
plt.show()
Q1 = pd.to_numeric(mobilityIndex_NY['gps_grocery_and_pharmacy']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NY['gps_grocery_and_pharmacy']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_GROCERY_AND_PHARMACY IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NY['gps_parks']))
plt.show()
Q1 = pd.to_numeric(mobilityIndex_NY['gps_parks']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NY['gps_parks']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_PARKS IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NY['gps_transit_stations']))
plt.show()
Q1 = pd.to_numeric(mobilityIndex_NY['gps_transit_stations']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NY['gps_transit_stations']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_TRANSIT_STATIONS IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NY['gps_workplaces']))
plt.show()

Q1 = pd.to_numeric(mobilityIndex_NY['gps_workplaces']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NY['gps_workplaces']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_WORKPLACES IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NY['gps_residential']))
plt.show()

Q1 = pd.to_numeric(mobilityIndex_NY['gps_residential']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NY['gps_residential']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_RESIDENTIAL IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NY['gps_away_from_home']))
plt.show()

Q1 = pd.to_numeric(mobilityIndex_NY['gps_away_from_home']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NY['gps_away_from_home']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_AWAY_FROM_HOME IS:", IQR)

sns.boxplot(x = pd.to_numeric(smallBusinessData_NY['RevenueChange']))
plt.show()

Q1 = pd.to_numeric(smallBusinessData_NY['RevenueChange']).quantile(0.25)
Q3 = pd.to_numeric(smallBusinessData_NY['RevenueChange']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR REVENUE CHANGE IS:", IQR)

sns.boxplot(x = pd.to_numeric(unemploymentData_NY['initclaims_rate']))
plt.show()

Q1 = pd.to_numeric(unemploymentData_NY['initclaims_rate']).quantile(0.25)
Q3 = pd.to_numeric(unemploymentData_NY['initclaims_rate']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR INITCLAIMS_RATE CHANGE IS:", IQR)

sns.boxplot(x = pd.to_numeric(unemploymentData_NY['contclaims_rate']))
plt.show()

Q1 = pd.to_numeric(unemploymentData_NY['contclaims_rate']).quantile(0.25)
Q3 = pd.to_numeric(unemploymentData_NY['contclaims_rate']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR CONTCLAIMS_RATE CHANGE IS:", IQR)

#PLOTTING AGAINST REVENUE CHANGE
indexingCaseData = []
indexingRevenueChange = []
indexingMobilityIndex = []
indexingUnemploymentData = []

for i in range(0, casesData_NC['year'].size):
    indexingCaseData.append(i)

for i in range(0, smallBusinessData_NY['year'].size):
    indexingRevenueChange.append(i)

for i in range(0, mobilityIndex_NY['year'].size):
    indexingMobilityIndex.append(i)

for i in range(0, unemploymentData_NY['year'].size):
    indexingUnemploymentData.append(i)

#CASE RATE
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingCaseData, casesData_NC['case_rate'])
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('Case Rate')
ax1.set_xlabel('Day')
ax1.set_ylabel('Confirmed Cases per 100k People')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()

#NEW CASE DATA
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingCaseData, casesData_NC['new_case_rate'])
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('New Case Rate')
ax1.set_xlabel('Day')
ax1.set_ylabel('New Confirmed Cases per 100k People')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()

#GPS_RETAIL_AND_RECREATION
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NY['gps_retail_and_recreation']))
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('gps_retail_and_recreation')
ax1.set_xlabel('Day')
ax1.set_ylabel('Time spent at retail and recreation locations')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()

#GPS_GROCERY_AND_PHARMACY
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NY['gps_grocery_and_pharmacy']))
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('gps_grocery_and_pharmacy')
ax1.set_xlabel('Day')
ax1.set_ylabel('Time spent at grocery and pharmacy')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()

#GPS_PARKS
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NY['gps_parks']))
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('gps_parks')
ax1.set_xlabel('Day')
ax1.set_ylabel('Time spent at Parks')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()

#GPS_TRANSIT_STATIONS
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NY['gps_transit_stations']))
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('gps_transit_stations')
ax1.set_xlabel('Day')
ax1.set_ylabel('Time spent at Transit Stations')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()

#GPS_WORKPLACES
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NY['gps_workplaces']))
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('gps_workplaces')
ax1.set_xlabel('Day')
ax1.set_ylabel('Time spent at Workplaces')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()

#GPS_RESIDENTIAL
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NY['gps_residential']))
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('gps_residential')
ax1.set_xlabel('Day')
ax1.set_ylabel('Time spent at Residential Locations')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()

#GPS_AWAY_FROM_HOME
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NY['gps_away_from_home']))
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('gps_away_from_home')
ax1.set_xlabel('Day')
ax1.set_ylabel('Time Spent Away From Home')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()

#INITCLAIMS_RATE
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingUnemploymentData, pd.to_numeric(unemploymentData_NY['initclaims_rate']))
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('initclaims_rate')
ax1.set_xlabel('Week')
ax1.set_ylabel('Weekly Inital Unemployment Insurance Claims per 100 People')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()

#CONTCLAIMS_RATE
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingUnemploymentData, pd.to_numeric(unemploymentData_NY['contclaims_rate']))
ax2.plot(indexingRevenueChange, smallBusinessData_NY['RevenueChange'])

ax1.set_title('contclaims_rate')
ax1.set_xlabel('Week')
ax1.set_ylabel('Weekly Continuing Unemployment Insurance Claims per 100 People')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()