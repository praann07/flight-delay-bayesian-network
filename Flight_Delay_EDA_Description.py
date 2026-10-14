import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['figure.dpi'] = 100

BASE = '/kaggle/input/datasets/organizations/usdot/flight-delays'
PLOT_DIR = 'EDA_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

airlines = pd.read_csv(f'{BASE}/airlines.csv').rename(columns={'AIRLINE': 'AIRLINE_NAME'})
airports = pd.read_csv(f'{BASE}/airports.csv')

DTYPES = {
    'YEAR': 'int16', 'MONTH': 'int8', 'DAY': 'int8', 'DAY_OF_WEEK': 'int8',
    'FLIGHT_NUMBER': 'int16',
    'SCHEDULED_DEPARTURE': 'int16', 'DEPARTURE_TIME': 'float32',
    'DEPARTURE_DELAY': 'float32', 'TAXI_OUT': 'float32', 'WHEELS_OFF': 'float32',
    'SCHEDULED_TIME': 'float32', 'ELAPSED_TIME': 'float32', 'AIR_TIME': 'float32',
    'DISTANCE': 'float32', 'WHEELS_ON': 'float32', 'TAXI_IN': 'float32',
    'SCHEDULED_ARRIVAL': 'int16', 'ARRIVAL_TIME': 'float32', 'ARRIVAL_DELAY': 'float32',
    'DIVERTED': 'int8', 'CANCELLED': 'int8',
    'AIR_SYSTEM_DELAY': 'float32', 'SECURITY_DELAY': 'float32',
    'AIRLINE_DELAY': 'float32', 'LATE_AIRCRAFT_DELAY': 'float32',
    'WEATHER_DELAY': 'float32',
}
flights = pd.read_csv(f'{BASE}/flights.csv', dtype=DTYPES)


def section(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, name), bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot: {PLOT_DIR}/{name}')


section('1. DATASET OVERVIEW')

print(f'AIRLINES: {airlines.shape[0]} rows, {airlines.shape[1]} columns')
print(airlines.to_string(index=False))
print()
print(f'AIRPORTS: {airports.shape[0]} rows, {airports.shape[1]} columns')
print(f'Unique airport codes: {airports["IATA_CODE"].nunique()}')
print(f'Countries represented: {airports["COUNTRY"].nunique()}')
print()
print(f'FLIGHTS: {flights.shape[0]} rows, {flights.shape[1]} columns')
print(f'Year(s) covered: {sorted(flights["YEAR"].unique())}')
print(f'Months covered: {sorted(flights["MONTH"].unique())}')
print(f'Airlines operating: {flights["AIRLINE"].nunique()}')
print(f'Unique aircraft (tail numbers): {flights["TAIL_NUMBER"].nunique()}')
print(f'Unique origin airports: {flights["ORIGIN_AIRPORT"].nunique()}')
print(f'Unique destination airports: {flights["DESTINATION_AIRPORT"].nunique()}')

section('2. FLIGHTS DATA DICTIONARY')

dtype_map = {
    'YEAR': 'int', 'MONTH': 'int', 'DAY': 'int', 'DAY_OF_WEEK': 'int',
    'AIRLINE': 'str', 'FLIGHT_NUMBER': 'int', 'TAIL_NUMBER': 'str',
    'ORIGIN_AIRPORT': 'str', 'DESTINATION_AIRPORT': 'str',
    'SCHEDULED_DEPARTURE': 'int (HHMM)', 'DEPARTURE_TIME': 'int (HHMM)',
    'DEPARTURE_DELAY': 'int (minutes)', 'TAXI_OUT': 'int (minutes)',
    'WHEELS_OFF': 'int (HHMM)', 'SCHEDULED_TIME': 'int (minutes)',
    'ELAPSED_TIME': 'int (minutes)', 'AIR_TIME': 'int (minutes)',
    'DISTANCE': 'int (miles)', 'WHEELS_ON': 'int (HHMM)',
    'TAXI_IN': 'int (minutes)', 'SCHEDULED_ARRIVAL': 'int (HHMM)',
    'ARRIVAL_TIME': 'int (HHMM)', 'ARRIVAL_DELAY': 'int (minutes)',
    'DIVERTED': 'int (0/1)', 'CANCELLED': 'int (0/1)',
    'CANCELLATION_REASON': 'str (A/B/C/D)', 'AIR_SYSTEM_DELAY': 'int (minutes)',
    'SECURITY_DELAY': 'int (minutes)', 'AIRLINE_DELAY': 'int (minutes)',
    'LATE_AIRCRAFT_DELAY': 'int (minutes)', 'WEATHER_DELAY': 'int (minutes)',
}

non_null_counts = flights.notna().sum()
total_rows = len(flights)

dictionary = pd.DataFrame({
    'Column': flights.columns,
    'Type': [dtype_map.get(c, str(flights[c].dtype)) for c in flights.columns],
    'Non-Null': non_null_counts.values,
    'Null %': (100 - non_null_counts.values / total_rows * 100).round(2),
    'Example': [str(flights[c].loc[flights[c].notna()].iloc[0]) if non_null_counts[c] else 'NA'
                for c in flights.columns],
})
print(dictionary.to_string(index=False))

section('3. DELAY SIGNAL & MISSING VALUES')

delay_cols = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'AIR_SYSTEM_DELAY',
              'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
print(flights[delay_cols].describe().T.round(2).to_string())

print('\nDelay-cause column coverage (minutes present when a cause was recorded):')
coverage = pd.DataFrame({
    'Cause': delay_cols[2:],
    'Rows with value': flights[delay_cols[2:]].notna().sum().values,
    'Share of flights': (flights[delay_cols[2:]].notna().mean() * 100).round(2).values,
})
print(coverage.to_string(index=False))

print('\nCancelled flights:')
print(flights['CANCELLED'].value_counts().to_string())
print('Diverted flights:', int((flights['DIVERTED'] == 1).sum()))

section('4. DEPARTURE DELAY DISTRIBUTION')

delayed_mask = flights['DEPARTURE_DELAY'] > 15
overall_delay_rate = delayed_mask.mean()
mean_delay = flights['DEPARTURE_DELAY'].mean()
median_delay = flights['DEPARTURE_DELAY'].median()
p75_delay = flights['DEPARTURE_DELAY'].quantile(0.75)

print(f'Departure delay - mean: {mean_delay:.2f} min | median: {median_delay:.2f} min | 75th pct: {p75_delay:.2f} min')
print(f'On-time (<=15 min): {(1 - overall_delay_rate) * 100:.2f}%')
print(f'Delayed (>15 min): {overall_delay_rate * 100:.2f}%')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(flights['DEPARTURE_DELAY'].clip(-60, 300), bins=60, color='#1f77b4', edgecolor='white')
axes[0].set_title('Departure Delay Distribution (clipped [-60, 300] min)')
axes[0].set_xlabel('Departure delay (minutes)')
axes[0].set_ylabel('Number of flights')
axes[0].axvline(15, color='red', ls='--', label='15-min on-time threshold')
axes[0].legend()
axes[1].bar(['On-time (<=15 min)', 'Delayed (>15 min)'],
            [1 - overall_delay_rate, overall_delay_rate],
            color=['#2ca02c', '#d62728'])
axes[1].set_title('On-time vs Delayed Flights')
axes[1].set_ylabel('Share of flights')
for i, v in enumerate([1 - overall_delay_rate, overall_delay_rate]):
    axes[1].text(i, v + 0.01, f'{v * 100:.2f}%', ha='center')
save(fig, '01_delay_distribution.png')

section('5. DELAY RATE BY AIRLINE')

airline_stats = (
    flights.groupby('AIRLINE')
    .agg(flights_count=('FLIGHT_NUMBER', 'count'), delay_rate=('DEPARTURE_DELAY', lambda s: (s > 15).mean()))
    .reset_index()
    .merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')
    .sort_values('delay_rate', ascending=False)
)
airline_stats['AIRLINE_NAME'] = airline_stats['AIRLINE_NAME'].fillna('Unknown')
print(airline_stats[['AIRLINE', 'AIRLINE_NAME', 'flights_count', 'delay_rate']].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=airline_stats, x='AIRLINE', y='delay_rate', order=airline_stats['AIRLINE'], ax=ax)
ax.set_title('Delay Rate by Airline (>15 min)')
ax.set_xlabel('Airline (IATA code)')
ax.set_ylabel('Delay rate')
save(fig, '02_delay_by_airline.png')

best_airline = airline_stats.iloc[-1]
worst_airline = airline_stats.iloc[0]

section('6. DELAY RATE BY MONTH & SEASON')

month_stats = (
    flights.groupby('MONTH')['DEPARTURE_DELAY']
    .agg(mean_delay='mean', delay_rate=lambda s: (s > 15).mean())
    .reset_index()
)
print(month_stats.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(month_stats['MONTH'], month_stats['delay_rate'] * 100, marker='o', color='#ff7f0e')
ax.set_title('Monthly Delay Rate')
ax.set_xlabel('Month')
ax.set_ylabel('Delay rate (%)')
ax.set_xticks(month_stats['MONTH'])
save(fig, '03_delay_by_month.png')

month_delay_rate = dict(zip(month_stats['MONTH'], month_stats['delay_rate']))
best_month = month_stats.loc[month_stats['delay_rate'].idxmin()]
worst_month = month_stats.loc[month_stats['delay_rate'].idxmax()]

section('7. DELAY RATE BY DAY OF WEEK & DAY TYPE')

day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday',
             5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
day_stats = (
    flights.groupby('DAY_OF_WEEK')['DEPARTURE_DELAY']
    .agg(mean_delay='mean', delay_rate=lambda s: (s > 15).mean())
    .reset_index()
)
day_stats['DAY_NAME'] = day_stats['DAY_OF_WEEK'].map(day_names)
print(day_stats.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(day_stats['DAY_OF_WEEK'], day_stats['delay_rate'] * 100, marker='o', color='#2ca02c')
ax.set_title('Delay Rate by Day of Week')
ax.set_xlabel('Day of week')
ax.set_ylabel('Delay rate (%)')
ax.set_xticks(list(day_names.keys()))
ax.set_xticklabels([day_names[d] for d in day_stats['DAY_OF_WEEK']], rotation=45)
save(fig, '04_delay_by_day.png')

flights['DAY_TYPE'] = np.where(flights['DAY_OF_WEEK'].isin([6, 7]), 'Weekend', 'Weekday')
weekday_rate = (flights.loc[flights['DAY_TYPE'] == 'Weekday', 'DEPARTURE_DELAY'] > 15).mean()
weekend_rate = (flights.loc[flights['DAY_TYPE'] == 'Weekend', 'DEPARTURE_DELAY'] > 15).mean()
print(f'Weekday delay rate: {weekday_rate * 100:.2f}% | Weekend delay rate: {weekend_rate * 100:.2f}%')

section('8. DELAY RATE BY TIME OF DAY')


hour = flights['SCHEDULED_DEPARTURE'] // 100
flights['TIME_PERIOD'] = np.select(
    [(hour >= 5) & (hour < 12), (hour >= 12) & (hour < 17), (hour >= 17) & (hour < 21)],
    ['Morning', 'Afternoon', 'Evening'],
    default='Night',
)
time_stats = (
    flights.groupby('TIME_PERIOD')['DEPARTURE_DELAY']
    .agg(flights_count='count', mean_delay='mean', delay_rate=lambda s: (s > 15).mean())
    .reindex(['Morning', 'Afternoon', 'Evening', 'Night'])
)
print(time_stats.round(4).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=time_stats.index, y=time_stats['delay_rate'] * 100, ax=ax, palette='viridis')
ax.set_title('Delay Rate by Time of Day')
ax.set_xlabel('Time period')
ax.set_ylabel('Delay rate (%)')
save(fig, '05_delay_by_time.png')

time_delay_rate = time_stats['delay_rate'].to_dict()
best_time = time_stats['delay_rate'].idxmin()
worst_time = time_stats['delay_rate'].idxmax()

section('9. AIRPORT ANALYSIS (TOP 15 ORIGINS)')

origin_stats = (
    flights.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY']
    .agg(flights_count='count', delay_rate=lambda s: (s > 15).mean())
    .reset_index()
    .sort_values('flights_count', ascending=False)
)
top15 = origin_stats.head(15).merge(airports, left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')
print(top15[['ORIGIN_AIRPORT', 'AIRPORT', 'CITY', 'STATE', 'flights_count', 'delay_rate']].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=top15, x='ORIGIN_AIRPORT', y='delay_rate', order=top15['ORIGIN_AIRPORT'], ax=ax)
ax.set_title('Delay Rate by Top 15 Origin Airports')
ax.set_xlabel('Origin airport (IATA)')
ax.set_ylabel('Delay rate')
save(fig, '06_delay_by_airport.png')

busiest_airport = top15.iloc[0]

section('10. CASCADE PROPAGATION DATASET DESCRIPTION')

flights_seq = flights[flights['CANCELLED'] == 0][
    ['TAIL_NUMBER', 'MONTH', 'DAY', 'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY']
].dropna()
flights_seq = flights_seq.sort_values(['TAIL_NUMBER', 'MONTH', 'DAY', 'SCHEDULED_DEPARTURE'])
flights_seq['DELAY_CAT'] = np.where(flights_seq['DEPARTURE_DELAY'] > 15, 'Delayed', 'OnTime')
flights_seq['NEXT_DELAY_CAT'] = flights_seq.groupby('TAIL_NUMBER')['DELAY_CAT'].shift(-1)
cascade_df = flights_seq.dropna(subset=['NEXT_DELAY_CAT'])

print(f'Total chained flight pairs (same aircraft, consecutive): {len(cascade_df):,}')
print(f'Aircraft used in cascade dataset: {cascade_df["TAIL_NUMBER"].nunique()}')
print(f'Average chain length per aircraft: {len(cascade_df) / cascade_df["TAIL_NUMBER"].nunique():.1f} flights')

cascade_table = pd.crosstab(cascade_df['DELAY_CAT'], cascade_df['NEXT_DELAY_CAT'], normalize='index')
print('\nP(CURRENT delay -> NEXT flight delay):')
print((cascade_table * 100).round(1).to_string())

delayed_next_if_delayed = cascade_table.loc['Delayed', 'Delayed']
delayed_next_if_ontime = cascade_table.loc['OnTime', 'Delayed']
cascade_ratio = delayed_next_if_delayed / delayed_next_if_ontime

section('11. AUTO-GENERATED REPORT DESCRIPTION')

summary = f"""
PROJECT DATASET DESCRIPTION — FLIGHT DELAY RISK ASSESSMENT USING BAYESIAN NETWORKS

Data source: US Department of Transportation 2015 Flight Delays and Cancellations
dataset (Kaggle: usdot/flight-delays). The dataset consists of three files:
airlines.csv ({airlines.shape[0]} airlines), airports.csv ({airports.shape[0]} airports),
and flights.csv containing {flights.shape[0]:,} domestic flight records across
{flights['YEAR'].nunique()} year, {flights['MONTH'].nunique()} months,
{flights['AIRLINE'].nunique()} airlines, {flights['TAIL_NUMBER'].nunique():,} unique
aircraft, and {flights['ORIGIN_AIRPORT'].nunique()} origin airports. Each flight record
captures scheduled and actual departure/arrival times, delays in minutes, taxi and
airborne time, distance, cancellation/diverting flags, and five delay-cause components
(air system, security, airline, late aircraft, and weather).

Overall, {overall_delay_rate * 100:.2f}% of flights departed more than 15 minutes late.
The mean departure delay was {mean_delay:.2f} minutes (median {median_delay:.2f} minutes,
75th percentile {p75_delay:.2f} minutes). Delay propensity varies strongly by condition:
by airline it ranges from {best_airline['delay_rate'] * 100:.1f}% for
{best_airline['AIRLINE_NAME']} ({best_airline['AIRLINE']}) to
{worst_airline['delay_rate'] * 100:.1f}% for {worst_airline['AIRLINE_NAME']}
({worst_airline['AIRLINE']}); by month it ranges from {best_month['delay_rate'] * 100:.1f}%
in month {int(best_month['MONTH'])} to {worst_month['delay_rate'] * 100:.1f}% in month
{int(worst_month['MONTH'])}. By time of day, delay rate rises from {time_delay_rate[best_time] * 100:.1f}%
in the {best_time.lower()} to {time_delay_rate[worst_time] * 100:.1f}% in the
{worst_time.lower()} ({worst_time} being the highest-risk period). Weekday flights are
delayed {weekday_rate * 100:.1f}% of the time versus {weekend_rate * 100:.1f}% on weekends,
and the busiest origin airport ({busiest_airport['ORIGIN_AIRPORT']} -
{busiest_airport['CITY']}, {busiest_airport['STATE']}) handled
{busiest_airport['flights_count']:,} departures.

For the cascade layer, consecutive flights flown by the same aircraft were chained by
tail number, yielding {len(cascade_df):,} real same-aircraft flight pairs. A current
flight that is delayed makes the next flight delayed {delayed_next_if_delayed * 100:.1f}%
of the time, versus only {delayed_next_if_ontime * 100:.1f}% when the current flight is
on time — a {cascade_ratio:.1f}x increase in next-leg delay risk, demonstrating
measurable delay propagation through the aircraft schedule.
"""
print(summary.strip())

print('\nAll plot files are saved in the EDA_plots/ directory.')
