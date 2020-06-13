# This script perfrom the statistical analysis for the pollution growth paper

# Importing required modules

import pandas as pd
import statsmodels.api as stats
from ToTeX import restab

# Reading in the data

data = pd.read_csv('C:/Users/User/Documents/Data/Pollution/pollution_data.csv')

# Prepping data for pollution regression

# Data sets for ndividual pollutants

co2_data = data[['ln_co2', 'ln_co2_lag', 'ln_sk', 'ln_n', 'ln_co2_intensity_ratio', 'Country', 'Year']].dropna()
ch4_data = data[['ln_ch4', 'ln_ch4_lag', 'ln_sk', 'ln_n', 'ln_ch4_intensity_ratio', 'Country', 'Year']].dropna()
nox_data = data[['ln_nox', 'ln_nox_lag', 'ln_sk', 'ln_n', 'ln_nox_intensity_ratio', 'Country', 'Year']].dropna()
ghg_data = data[['ln_ghg', 'ln_ghg_lag', 'ln_sk', 'ln_n', 'ln_ghg_intensity_ratio', 'Country', 'Year']].dropna()

# Creating dummy variables for each pollutant

co2_national_dummies = pd.get_dummies(co2_data['Country'])
co2_year_dummies = pd.get_dummies(co2_data['Year'])
ch4_national_dummies = pd.get_dummies(ch4_data['Country'])
ch4_year_dummies = pd.get_dummies(ch4_data['Year'])
nox_national_dummies = pd.get_dummies(nox_data['Country'])
nox_year_dummies = pd.get_dummies(nox_data['Year'])
ghg_national_dummies = pd.get_dummies(ghg_data['Country'])
ghg_year_dummies = pd.get_dummies(ghg_data['Year'])

# Replacing Country and Year with fixed effects

co2_data = pd.concat([co2_data, co2_national_dummies, co2_year_dummies], axis = 1)
ch4_data = pd.concat([ch4_data, ch4_national_dummies, ch4_year_dummies], axis = 1)
nox_data = pd.concat([nox_data, nox_national_dummies, nox_year_dummies], axis = 1)
ghg_data = pd.concat([ghg_data, ghg_national_dummies, ghg_year_dummies], axis = 1)

co2_data = co2_data.drop(['Country', 'Year', 1971, 'United States'], axis = 1)
ch4_data = ch4_data.drop(['Country', 'Year', 1971, 'United States'], axis = 1)
nox_data = nox_data.drop(['Country', 'Year', 1971, 'United States'], axis = 1)
ghg_data = ghg_data.drop(['Country', 'Year', 1971, 'United States'], axis = 1)

# Create the Y and X matrices

CO2 = co2_data['ln_co2']
CH4 = ch4_data['ln_ch4']
NOX = nox_data['ln_nox']
GHG = ghg_data['ln_ghg']

X_CO2 = co2_data.drop(['ln_co2'], axis = 1)
X_CH4 = ch4_data.drop(['ln_ch4'], axis = 1)
X_NOX = nox_data.drop(['ln_nox'], axis = 1)
X_GHG = ghg_data.drop(['ln_ghg'], axis = 1)

# Running pollution regressions

co2_mod = stats.OLS(CO2, X_CO2)
ch4_mod = stats.OLS(CH4, X_CH4)
nox_mod = stats.OLS(NOX, X_NOX)
ghg_mod = stats.OLS(GHG, X_GHG)

models = [co2_mod, ch4_mod, nox_mod, ghg_mod]
names = ['CO2', 'CH4', 'NOx', 'GHG']
res_list = []

for mod in models:
    
    res = mod.fit(cov_type = 'HC1')
    res_list.append(res)
    print(res.summary())
    file = open('C:/Users/User/Documents/Data/Pollution/Baseline/' + names[models.index(mod)] + '.txt', 'w')
    file.write(res.summary().as_text())
    file.close()
    
restab(res_list, 'C:/Users/User/Documents/Data/Pollution/Baseline/restab.txt')

# Next we run gdp models to check coefficients

gdp_data = data[['ln_Income', 'ln_Income_lag', 'ln_sk', 'ln_n', 'Country', 'Year']].dropna()

gdp_national_dummies = pd.get_dummies(gdp_data['Country'])
gdp_year_dummies = pd.get_dummies(gdp_data['Year'])

gdp_data = pd.concat([gdp_data, gdp_national_dummies, gdp_year_dummies], axis = 1)
gdp_data = gdp_data.drop(['Country', 'Year', 1971, 'United States'], axis = 1)

Y = gdp_data['ln_Income']
X = gdp_data.drop(['ln_Income'], axis = 1)

mod = stats.OLS(Y, X)
res = mod.fit(cov_type = 'HC1')
print(res.summary())
file = open('C:/Users/User/Documents/Data/Pollution/Baseline/Y.txt', 'w')
file.write(res.summary().as_text())
file.close()

res_list.append(res)    
restab(res_list, 'C:/Users/User/Documents/Data/Pollution/Baseline/restab_Y.txt')

# Now we estimate \alpha for each model

a1_co2 = co2_mod.fit().params['ln_sk'] / (co2_mod.fit().params['ln_sk'] + 1 - co2_mod.fit().params['ln_co2_lag'])
a2_co2 = -1*co2_mod.fit().params['ln_n'] / (-1*co2_mod.fit().params['ln_n'] + 1 - co2_mod.fit().params['ln_co2_lag'])
a1_ch4 = ch4_mod.fit().params['ln_sk'] / (ch4_mod.fit().params['ln_sk'] + 1 - ch4_mod.fit().params['ln_ch4_lag'])
a2_ch4 = -1*ch4_mod.fit().params['ln_n'] / (-1*ch4_mod.fit().params['ln_n'] + 1 - ch4_mod.fit().params['ln_ch4_lag'])
a1_nox = nox_mod.fit().params['ln_sk'] / (nox_mod.fit().params['ln_sk'] + 1 - nox_mod.fit().params['ln_nox_lag'])
a2_nox = -1*nox_mod.fit().params['ln_n'] / (-1*nox_mod.fit().params['ln_n'] + 1 - nox_mod.fit().params['ln_nox_lag'])
a1_ghg = ghg_mod.fit().params['ln_sk'] / (ghg_mod.fit().params['ln_sk'] + 1 - ghg_mod.fit().params['ln_ghg_lag'])
a2_ghg = -1*ghg_mod.fit().params['ln_n'] / (-1*ghg_mod.fit().params['ln_n'] + 1 - ghg_mod.fit().params['ln_ghg_lag'])
a1_gdp = mod.fit().params['ln_sk'] / (mod.fit().params['ln_sk'] + 1 - mod.fit().params['ln_Income_lag'])
a2_gdp = -1*mod.fit().params['ln_n'] / (-1*mod.fit().params['ln_n'] + 1 - mod.fit().params['ln_Income_lag'])

a1 = [a1_co2, a1_ch4, a1_nox, a1_ghg, a1_gdp]
a2 = [a2_co2, a2_ch4, a2_nox, a2_ghg, a2_gdp]

