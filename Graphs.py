import ast
import warnings

import geojsonio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


pd.set_option('display.max_columns', None)
crimes = pd.read_csv(
    '/content/drive/MyDrive/Colab Notebooks/data_files/cleaned_crime.csv')
# crimes = crimes[:100]

crimes[crimes['Location'] == '(0.00000000, 0.00000000)']

offences = pd.read_excel(
    '/content/drive/MyDrive/Colab Notebooks/data_files/rmsoffensecodes.xlsx')
offences.head(5)

crimes = crimes.merge(offences, left_on='OFFENSE_CODE', right_on='CODE', how='inner')
crimes = crimes.drop(columns=['CODE']).rename(columns={'NAME': 'OFFENSE_NAME'})

crimes['SHOOTING'] = crimes['SHOOTING'].apply(
    lambda x: 1 if x == 'Y' or x == 1.0 else 0 if x == 0 or x == 0.0 else 'Unknown')

crimes['DISTRICT'] = crimes['DISTRICT'].astype(str).apply(lambda x: 'Unknown' if x == 'nan' else x)
latlong = crimes.loc[crimes['Lat'].isnull() & crimes['Long'].isnull()]

crimes['OFFENSE_CATEGORY'] = crimes['OFFENSE_DESCRIPTION'].apply(lambda x: x.split('-')[0])
crimes['OFFENSE_ACTIVITY'] = crimes['OFFENSE_NAME'].apply(lambda x: x.split('-')[-1:][0])
crimes = crimes.drop('OFFENSE_NAME', axis=1)

crimes['OCCURRED_ON_DATE'] = crimes['OCCURRED_ON_DATE'].apply(
    lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))

crimes['date'] = crimes['OCCURRED_ON_DATE'].dt.strftime('%Y-%m-%d')
crimes['year_month'] = crimes['OCCURRED_ON_DATE'].dt.strftime('%B %Y')
crimes['hour'] = crimes['OCCURRED_ON_DATE'].dt.hour
crimes['day'] = crimes['OCCURRED_ON_DATE'].dt.day_name()
crimes['month'] = crimes['OCCURRED_ON_DATE'].dt.month_name()
crimes['year'] = crimes['OCCURRED_ON_DATE'].dt.year

crimes['time_period'] = crimes['hour'].apply(lambda x: period(x))
crimes['weekend'] = crimes['day'].apply(lambda x: weekend(x))

sns.set(rc={'figure.figsize': (20, 10)})

def weekend(day):
    if day == 'Saturday' or day == 'Sunday':
        return 1
    else:
        return 0

def period(hour):
    if hour >= 5 and hour < 12:
        return 'Morning'
    elif hour > 12 and hour < 17:
        return 'Afternoon'
    elif hour > 17 and hour < 22:
        return 'Evening'
    else:
        return 'Midnight'

def series(df, col, threshold, string_other='OTHER'):
    count_incidents = df[col].value_counts()
    mask = count_incidents > threshold
    tail_prob = count_incidents.loc[~mask].sum()
    count_incidents = count_incidents.loc[mask]
    count_incidents[string_other] = tail_prob
    return count_incidents
    
#Boston common Crimes
incidents = series(crimes, 'OFFENSE_CATEGORY', 25000)
sns.set(font_scale=1.25)
ax = sns.barplot(y=incidents.index, x=incidents.values, color="orange")
ax.set_title('Most Common Crime Types in Boston', fontweight="bold")
plt.savefig("/content/drive/MyDrive/Colab Notebooks/outputs/CommonCrimeTypes.png")
plt.show()

#Monthly crimes
crimes_monthly = crimes[~crimes['year_month'].isin(['June 2015', 'August 2020'])]
monthly_crimes = crimes_monthly.groupby('year_month').size().reset_index(name='COUNT')
monthly_crimes['year_month_ordered'] = pd.to_datetime(monthly_crimes['year_month'], format='%B %Y')
monthly_crimes = monthly_crimes.sort_values('year_month_ordered')
ax = sns.lineplot(x='year_month', y="COUNT", data=monthly_crimes, sort=False)
ax1 = plt.setp(ax.get_xticklabels(), rotation=90)
ax2 = ax.set_title('Total Crimes per Month:', fontweight="bold")
plt.savefig("/content/drive/MyDrive/Colab Notebooks/outputs/CrimesPerMonth.png")
plt.show()


daily_crimes = crimes.groupby(['date']).size().reset_index(name='count')
print("Crime Reports per Day: ", int(daily_crimes['count'].mean()))


#Average crimes daily
daily_average = crimes.groupby(['date', 'day']).size().reset_index(name='count')
daily_average = daily_average.groupby('day').mean().reset_index()
order_by_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ax = sns.barplot(x='day', y='count', data=daily_average, color='orange', order=order_by_day)
ax = ax.set_title('Average Number of Crimes Daily 2015-2020', fontweight="bold")
plt.savefig("/content/drive/MyDrive/Colab Notebooks/outputs/AvgDailyCrimes.png")
plt.show()

# crimes = crimes.assign(time_period=crimes['hour'])
#Proportion of crimes per day
daily_propor = crimes.groupby(['date', 'time_period', 'day']).size().reset_index(name='Percentage')
daily_propor = daily_propor.groupby(['time_period', 'day']).mean().reset_index()
daily_propor['count'] = daily_propor.groupby('day')['Percentage'].apply(lambda x: 100 * x / float(x.sum()))
time_order = ['Morning', 'Afternoon', 'Evening', 'Midnight']
ax = sns.barplot(x='day', hue='time_period', y='Percentage', data=daily_propor, color='orange', order=order_by_day,
                 hue_order=time_order)
ax = ax.set_title('Proportion of Crime Reports in a Day 2015-2020', fontweight="bold")
plt.savefig("/content/drive/MyDrive/Colab Notebooks/outputs/DailyCrimeProportion.png")
plt.show()


#top 5 offences per day
daily_offences = crimes.groupby(['year', 'OFFENSE_CATEGORY']).size().reset_index(name='count')
daily_offences = daily_offences.groupby('year').apply(
    lambda x: x.sort_values(["count"], ascending=False)).reset_index(
    drop=True)
daily_offences = daily_offences.groupby('year').head(5)


monthly_crimes = crimes.groupby(['day', 'OFFENSE_CATEGORY']).size().reset_index(name='count')
monthly_crimes = monthly_crimes.groupby('day').apply(lambda x: x.sort_values(["count"], ascending=False)).reset_index(
    drop=True)
monthly_crimes = monthly_crimes.groupby('day').head(5)
ax = sns.barplot(x='day', hue='OFFENSE_CATEGORY', y='count', data=monthly_crimes, color='orange', order=order_by_day)
ax = ax.set_title('Top 5 Offense Category per Day 2015-2020', fontweight="bold")
plt.savefig("/content/drive/MyDrive/Colab Notebooks/outputs/TopOffenceCategoryPerDay.png")
plt.show()
