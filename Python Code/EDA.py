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
print(casesData_NC.head(10))

casesData_NC.loc[casesData_NC.case_rate == '.', 'case_rate'] = np.nan
casesData_NC.loc[casesData_NC.new_case_rate == '.', 'new_case_rate'] = np.nan

print(casesData_NC.isnull().sum())
print(casesData_NC.head(10))
mean_value = pd.to_numeric(casesData_NC['new_case_rate']).mean()
casesData_NC = casesData_NC.fillna(mean_value)
print(casesData_NC.isnull().sum())
print(casesData_NC.head(10))

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


#Looking for Outliers
#Cases Data
sns.boxplot(x = casesData_NC['case_rate'])
plt.show()

Q1 = casesData_NC['case_rate'].quantile(0.25)
Q3 = casesData_NC['case_rate'].quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR CASE_RATE IS:",IQR)

sns.boxplot(x = pd.to_numeric(casesData_NC['new_case_rate']))
plt.show()

Q1 = pd.to_numeric(casesData_NC['new_case_rate']).quantile(0.25)
Q3 = pd.to_numeric(casesData_NC['new_case_rate']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR NEW_CASE_RATE IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NC['gps_retail_and_recreation']))
plt.show()
Q1 = pd.to_numeric(mobilityIndex_NC['gps_retail_and_recreation']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NC['gps_retail_and_recreation']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_RETAIL_AND_RECREATION IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NC['gps_grocery_and_pharmacy']))
plt.show()
Q1 = pd.to_numeric(mobilityIndex_NC['gps_grocery_and_pharmacy']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NC['gps_grocery_and_pharmacy']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_GROCERY_AND_PHARMACY IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NC['gps_parks']))
plt.show()
Q1 = pd.to_numeric(mobilityIndex_NC['gps_parks']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NC['gps_parks']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_PARKS IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NC['gps_transit_stations']))
plt.show()
Q1 = pd.to_numeric(mobilityIndex_NC['gps_transit_stations']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NC['gps_transit_stations']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_TRANSIT_STATIONS IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NC['gps_workplaces']))
plt.show()

Q1 = pd.to_numeric(mobilityIndex_NC['gps_workplaces']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NC['gps_workplaces']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_WORKPLACES IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NC['gps_residential']))
plt.show()

Q1 = pd.to_numeric(mobilityIndex_NC['gps_residential']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NC['gps_residential']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_RESIDENTIAL IS:", IQR)

sns.boxplot(x = pd.to_numeric(mobilityIndex_NC['gps_away_from_home']))
plt.show()

Q1 = pd.to_numeric(mobilityIndex_NC['gps_away_from_home']).quantile(0.25)
Q3 = pd.to_numeric(mobilityIndex_NC['gps_away_from_home']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR GPS_AWAY_FROM_HOME IS:", IQR)

sns.boxplot(x = pd.to_numeric(smallBusinessData_NC['RevenueChange']))
plt.show()

Q1 = pd.to_numeric(smallBusinessData_NC['RevenueChange']).quantile(0.25)
Q3 = pd.to_numeric(smallBusinessData_NC['RevenueChange']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR REVENUE CHANGE IS:", IQR)

sns.boxplot(x = pd.to_numeric(unemploymentData_NC['initclaims_rate']))
plt.show()

Q1 = pd.to_numeric(unemploymentData_NC['initclaims_rate']).quantile(0.25)
Q3 = pd.to_numeric(unemploymentData_NC['initclaims_rate']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR INITCLAIMS_RATE CHANGE IS:", IQR)

sns.boxplot(x = pd.to_numeric(unemploymentData_NC['contclaims_rate']))
plt.show()

Q1 = pd.to_numeric(unemploymentData_NC['contclaims_rate']).quantile(0.25)
Q3 = pd.to_numeric(unemploymentData_NC['contclaims_rate']).quantile(0.75)
IQR = Q3 - Q1
print("THE IQR FOR CONTCLAIMS_RATE CHANGE IS:", IQR)

#PLOTTING AGAINST REVENUE CHANGE
indexingCaseData = []
indexingRevenueChange = []
indexingMobilityIndex = []
indexingUnemploymentData = []

for i in range(0, casesData_NC['year'].size):
    indexingCaseData.append(i)

for i in range(0, smallBusinessData_NC['year'].size):
    indexingRevenueChange.append(i)

for i in range(0, mobilityIndex_NC['year'].size):
    indexingMobilityIndex.append(i)

for i in range(0, unemploymentData_NC['year'].size):
    indexingUnemploymentData.append(i)

#CASE RATE
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(indexingCaseData, casesData_NC['case_rate'])
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

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
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

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

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NC['gps_retail_and_recreation']))
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

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

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NC['gps_grocery_and_pharmacy']))
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

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

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NC['gps_parks']))
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

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

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NC['gps_transit_stations']))
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

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

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NC['gps_workplaces']))
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

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

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NC['gps_residential']))
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

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

ax1.plot(indexingMobilityIndex, pd.to_numeric(mobilityIndex_NC['gps_away_from_home']))
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

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

ax1.plot(indexingUnemploymentData, pd.to_numeric(unemploymentData_NC['initclaims_rate']))
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

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

ax1.plot(indexingUnemploymentData, pd.to_numeric(unemploymentData_NC['contclaims_rate']))
ax2.plot(indexingRevenueChange, smallBusinessData_NC['RevenueChange'])

ax1.set_title('contclaims_rate')
ax1.set_xlabel('Week')
ax1.set_ylabel('Weekly Continuing Unemployment Insurance Claims per 100 People')

ax2.set_title('Revenue Change')
ax2.set_xlabel('Day')
ax2.set_ylabel('Percent Change in Net Revenue')
plt.show()