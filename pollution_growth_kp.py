# This script performs the statistical analysis for the pollution growth paper

# Importing required modules

import pandas as pd
import numpy as np
import statsmodels.api as stats
from ToTeX import restab

# Reading in the data

data = pd.read_csv('C:/Users/User/Documents/Data/Pollution/pollution_data_kp.csv')

# Prepping data for pollution regression

# Data sets for ndividual pollutants

co2_data = data[['ln_co2', 'ln_co2_lag', 'ln_sk', 'ln_n5', 'ln_co2_intensity_rate', 'Country', 'Year']].dropna()#, 'ln_co2_intensity_lag']].dropna()
ch4_data = data[['ln_ch4', 'ln_ch4_lag3', 'ln_sk', 'ln_n5', 'ln_ch4_intensity_rate', 'Country', 'Year']].dropna()#, 'ln_ch4_intensity_lag3']].dropna()
nox_data = data[['ln_nox', 'ln_nox_lag', 'ln_sk', 'ln_n5', 'ln_nox_intensity_rate', 'Country', 'Year']].dropna()#, 'ln_nox_intensity_lag']].dropna()
ghg_data = data[['ln_ghg', 'ln_ghg_lag', 'ln_sk', 'ln_n5', 'ln_ghg_intensity_rate', 'Country', 'Year']].dropna()#, 'ln_ghg_intensity_lag']].dropna()

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

co2_data = co2_data.drop(['Country', 'Year', 1993, 'United States'], axis = 1)
ch4_data = ch4_data.drop(['Country', 'Year', 1993, 'United States'], axis = 1)
nox_data = nox_data.drop(['Country', 'Year', 1993, 'United States'], axis = 1)
ghg_data = ghg_data.drop(['Country', 'Year', 1993, 'United States'], axis = 1)

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
    file = open('C:/Users/User/Documents/Data/Pollution/' + names[models.index(mod)] + '_kp.txt', 'w')
    file.write(res.summary().as_text())
    file.close()
    
restab(res_list, 'C:/Users/User/Documents/Data/Pollution/restab_kp.txt')

# Next we run gdp models to check coefficients

gdp_data = data[['ln_Income', 'ln_Income_lag', 'ln_sk', 'ln_n5', 'Country', 'Year']].dropna()

gdp_national_dummies = pd.get_dummies(gdp_data['Country'])
gdp_year_dummies = pd.get_dummies(gdp_data['Year'])

gdp_data = pd.concat([gdp_data, gdp_national_dummies, gdp_year_dummies], axis = 1)
gdp_data = gdp_data.drop(['Country', 'Year', 1993, 'Canada'], axis = 1)

Y = gdp_data['ln_Income']
X = gdp_data.drop(['ln_Income'], axis = 1)

mod = stats.OLS(Y, X)
res = mod.fit(cov_type = 'HC1')
print(res.summary())
file = open('C:/Users/User/Documents/Data/Pollution/Y_kp.txt', 'w')
file.write(res.summary().as_text())
file.close()

res_list.append(res)    
restab(res_list, 'C:/Users/User/Documents/Data/Pollution/restab_Y_kp.txt')

# Now we estimate \alpha for each model

alpha_co2 = co2_mod.fit().params['ln_sk'] / (co2_mod.fit().params['ln_sk'] + 1 - co2_mod.fit().params['ln_co2_lag'])
alpha_ch4 = -1*ch4_mod.fit().params['ln_n5'] / (-1*ch4_mod.fit().params['ln_n5'] + 1 - ch4_mod.fit().params['ln_ch4_lag3'])
alpha_nox = nox_mod.fit().params['ln_sk'] / (nox_mod.fit().params['ln_sk'] + 1 - nox_mod.fit().params['ln_nox_lag'])
alpha_ghg = ghg_mod.fit().params['ln_sk'] / (ghg_mod.fit().params['ln_sk'] + 1 - ghg_mod.fit().params['ln_ghg_lag'])
alpha_gdp = mod.fit().params['ln_sk'] / (mod.fit().params['ln_sk'] + 1 - mod.fit().params['ln_Income_lag'])

alpha = pd.Series([alpha_co2, alpha_ch4, alpha_nox, alpha_ghg, alpha_gdp], name = 'alpha')
a_names = pd.Series(['alpha_co2', 'alpha_ch4', 'alpha_nox', 'alpha_ghg', 'alpha_gdp'], name = 'Variable')
alpha = pd.concat([a_names, alpha], axis = 1)
alpha.to_csv('C:/Users/User/Documents/Data/Pollution/alphas_kp.txt', index = False)

# Calculating convergence rates

con_co2 = (1 - alpha_co2) * (np.exp(np.mean(co2_data['ln_n5'])))
con_ch4 = (1 - alpha_ch4) * (np.exp(np.mean(ch4_data['ln_n5'])))
con_nox = (1 - alpha_nox) * (np.exp(np.mean(nox_data['ln_n5'])))
con_ghg = (1 - alpha_ghg) * (np.exp(np.mean(ghg_data['ln_n5'])))
con_gdp = (1 - alpha_gdp) * (np.exp(np.mean(gdp_data['ln_n5'])))

cons = pd.Series([con_co2, con_ch4, con_nox, con_ghg, con_gdp], name = 'convergence_rate')
c_names = pd.Series(['co2', 'ch4', 'nox', 'ghg', 'gdp'], name = 'Variable')
alpha = pd.concat([c_names, cons], axis = 1)
cons.to_csv('C:/Users/User/Documents/Data/Pollution/convergence_rates_kp.txt', index = False)

# Next we perform unit root tests to make sure our results are good

co2 = data[['ln_co2', 'ln_co2_lag']].dropna()
ch4 = data[['ln_ch4', 'ln_ch4_lag3']].dropna()
nox = data[['ln_nox', 'ln_nox_lag']].dropna()
ghg = data[['ln_ghg', 'ln_ghg_lag']].dropna()

co2Xt = co2['ln_co2_lag']
co2Xct = stats.add_constant(co2Xt)
co2Y = co2['ln_co2']

ch4Xt = ch4['ln_ch4_lag3']
ch4Xct = stats.add_constant(ch4Xt)
ch4Y = ch4['ln_ch4']

noxXt = nox['ln_nox_lag']
noxXct = stats.add_constant(noxXt)
noxY = nox['ln_nox']

co2_urt = stats.OLS(co2Y, co2Xt)
co2_urct = stats.OLS(co2Y, co2Xct)

ch4_urt = stats.OLS(ch4Y, ch4Xt)
ch4_urct = stats.OLS(ch4Y, ch4Xct)

nox_urt = stats.OLS(noxY, noxXt)
nox_urct = stats.OLS(noxY, noxXct)

models = [co2_urt, co2_urct, ch4_urt, ch4_urct, nox_urt, nox_urct]

for mod in models:
    
    res = mod.fit(cov_type = 'HC1')
    print(res.conf_int())

