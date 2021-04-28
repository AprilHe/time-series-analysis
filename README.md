

<h1 align="center"><a href=>Time Series</a></h1>

## Context

Forecasting how many bikes are rented from the TFL (Transport for London) Cycle Hire scheme.

**Task**
- Using ONLY the TFL dataset
- Using the TFL and electrical power generation dataset, check if there are any relationship between electrical power and rented bikes.

**Data**
- [tfl-daily-cycle-hires.xlsx ](https://data.london.gov.uk/dataset/number-bicycle-hires): the daily number of hired bikes
- [electrical_power_data.csv](https://www.ref.org.uk/fuel/index.php?valdate=2009&tab=dp&share=N): eletrical power consumption
- ukbankholidays.csv: uk bank holidays

**Method**
1. ARIMA models
2. [Prophet by Facebook](https://facebook.github.io/prophet/)
3. XGboost
