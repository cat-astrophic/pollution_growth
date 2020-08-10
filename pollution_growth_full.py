# This script performs the statistical analysis for the pollution growth paper

# Importing required modules

import pandas as pd
import numpy as np
import statsmodels.api as stats
from ToTeX import restab

# Reading in the data

data = pd.read_csv('C:/Users/User/Documents/Data/Pollution/pollution_data.csv')
W = pd.read_csv('C:/Users/User/Documents/Data/Pollution/pollution_data_W_reference.csv', header = None)

# Creating a reference list of nations

nations = list(data.Country.unique())

# Prepping data for pollution regression

# Data sets for individual pollutants

co2_data = data[['ln_co2', 'ln_co2_lag', 'ln_sk', 'ln_n5', 'ln_co2_intensity_rate', 'Country', 'Year', 'ln_co2_intensity_lag']].dropna()
ch4_data = data[['ln_ch4', 'ln_ch4_lag', 'ln_sk', 'ln_n5', 'ln_ch4_intensity_rate', 'Country', 'Year', 'ln_ch4_intensity_lag']].dropna()
nox_data = data[['ln_nox', 'ln_nox_lag', 'ln_sk', 'ln_n5', 'ln_nox_intensity_rate', 'Country', 'Year', 'ln_nox_intensity_lag']].dropna()

# Creating dummy variables for each pollutant

co2_national_dummies = pd.get_dummies(co2_data['Country'])
co2_year_dummies = pd.get_dummies(co2_data['Year'])
ch4_national_dummies = pd.get_dummies(ch4_data['Country'])
ch4_year_dummies = pd.get_dummies(ch4_data['Year'])
nox_national_dummies = pd.get_dummies(nox_data['Country'])
nox_year_dummies = pd.get_dummies(nox_data['Year'])

# Replacing Country and Year with fixed effects

co2_data = pd.concat([co2_data, co2_national_dummies, co2_year_dummies], axis = 1)
ch4_data = pd.concat([ch4_data, ch4_national_dummies, ch4_year_dummies], axis = 1)
nox_data = pd.concat([nox_data, nox_national_dummies, nox_year_dummies], axis = 1)

co2_data = co2_data.drop(['Country', 'Year', 1971, 'United States'], axis = 1)
ch4_data = ch4_data.drop(['Country', 'Year', 1971, 'United States'], axis = 1)
nox_data = nox_data.drop(['Country', 'Year', 1971, 'United States'], axis = 1)

# Create the Y and X matrices

CO2 = co2_data['ln_co2']
CH4 = ch4_data['ln_ch4']
NOX = nox_data['ln_nox']

X_CO2 = co2_data.drop(['ln_co2'], axis = 1)
X_CH4 = ch4_data.drop(['ln_ch4'], axis = 1)
X_NOX = nox_data.drop(['ln_nox'], axis = 1)

# Running pollution regressions

co2_mod = stats.OLS(CO2, X_CO2)
ch4_mod = stats.OLS(CH4, X_CH4)
nox_mod = stats.OLS(NOX, X_NOX)

models = [co2_mod, ch4_mod, nox_mod]
names = ['CO2', 'CH4', 'NOx']
res_list = []

for mod in models:
    
    res = mod.fit(cov_type = 'HC1')
    res_list.append(res)
    print(res.summary())
    file = open('C:/Users/User/Documents/Data/Pollution/' + names[models.index(mod)] + '.txt', 'w')
    file.write(res.summary().as_text())
    file.close()
    
restab(res_list, 'C:/Users/User/Documents/Data/Pollution/restab.txt')

# After running the conditional convergence models, we set up the network effects models

# Compute technology growth rate

# \widetilde{g} = \left(\frac{1}{T}\right)\sum\limits_{t=1}^{T}\left(\frac{\eta_{t}}{t-\gamma(t-1)}\right)

g_co2 = (1/23) * sum([(co2_mod.fit().params[i] / ((i-1971) - (co2_mod.fit().params['ln_co2_lag'] * (i-1972)))) for i in range(1972,2015)])
g_ch4 = (1/21) * sum([(ch4_mod.fit().params[i] / ((i-1971) - (ch4_mod.fit().params['ln_ch4_lag'] * (i-1972)))) for i in range(1972,2013)])
g_nox = (1/21) * sum([(nox_mod.fit().params[i] / ((i-1971) - (nox_mod.fit().params['ln_nox_lag'] * (i-1972)))) for i in range(1972,2013)])

# Add technology parameters to the dataframe

co2_tech = []
ch4_tech = []
nox_tech = []

for i in range(len(data)):
    
    if data.Year[i] > 1970 and data.Country[i] in co2_mod.fit().params.keys():
        
        co2_tech.append(co2_mod.fit().params[data.Country[i]] + (g_co2 * (data.Year[i] - 1971)))
        
    else:
        
        co2_tech.append('')
        
    if data.Year[i] > 1970 and data.Country[i] in ch4_mod.fit().params.keys():
        
        ch4_tech.append(ch4_mod.fit().params[data.Country[i]] + (g_ch4 * (data.Year[i] - 1971)))
        
    else:
        
        ch4_tech.append('')
        
    if data.Year[i] > 1970 and data.Country[i] in nox_mod.fit().params.keys():
        
        nox_tech.append(nox_mod.fit().params[data.Country[i]] + (g_nox * (data.Year[i] - 1971)))
        
    else:
        
        nox_tech.append('')

# Add technology values to data set

co2_tech = pd.Series(co2_tech, name = 'co2_tech')
ch4_tech = pd.Series(co2_tech, name = 'ch4_tech')
nox_tech = pd.Series(co2_tech, name = 'nox_tech')

data = pd.concat([data, co2_tech, ch4_tech, nox_tech], axis = 1)

# Convert '' to np.nan to use pandas dropna

data[data[['co2_tech', 'ch4_tech', 'nox_tech']] == ''] = np.nan

# Data prep for network effects regressions for intensities

tc_co2_rob = data[['co2_intensity', 'co2_intensity_init', 'co2_intensity_lag', 'co2_tech', 'TC_CO2_ROB', 'Country', 'Year']].dropna()
tc_ch4_rob = data[['ch4_intensity', 'ch4_intensity_init', 'ch4_intensity_lag', 'ch4_tech', 'TC_CH4_ROB', 'Country', 'Year']].dropna()
tc_nox_rob = data[['nox_intensity', 'nox_intensity_init', 'nox_intensity_lag', 'nox_tech', 'TC_NOX_ROB', 'Country', 'Year']].dropna()

co2_national_dummies = pd.get_dummies(tc_co2_rob['Country'])
co2_year_dummies = pd.get_dummies(tc_co2_rob['Year'])
ch4_national_dummies = pd.get_dummies(tc_ch4_rob['Country'])
ch4_year_dummies = pd.get_dummies(tc_ch4_rob['Year'])
nox_national_dummies = pd.get_dummies(tc_nox_rob['Country'])
nox_year_dummies = pd.get_dummies(tc_nox_rob['Year'])

xtc_co2_rob = pd.concat([tc_co2_rob, co2_national_dummies, co2_year_dummies], axis = 1).drop(['co2_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)
xtc_ch4_rob = pd.concat([tc_ch4_rob, ch4_national_dummies, ch4_year_dummies], axis = 1).drop(['ch4_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)
xtc_nox_rob = pd.concat([tc_nox_rob, nox_national_dummies, nox_year_dummies], axis = 1).drop(['nox_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)

exp_co2_rob = data[['co2_intensity', 'co2_intensity_init', 'co2_intensity_lag', 'co2_tech', 'EXP_CO2_ROB', 'Country', 'Year']].dropna()
exp_ch4_rob = data[['ch4_intensity', 'ch4_intensity_init', 'ch4_intensity_lag', 'ch4_tech', 'EXP_CH4_ROB', 'Country', 'Year']].dropna()
exp_nox_rob = data[['nox_intensity', 'nox_intensity_init', 'nox_intensity_lag', 'nox_tech', 'EXP_NOX_ROB', 'Country', 'Year']].dropna()

co2_national_dummies = pd.get_dummies(exp_co2_rob['Country'])
co2_year_dummies = pd.get_dummies(exp_co2_rob['Year'])
ch4_national_dummies = pd.get_dummies(exp_ch4_rob['Country'])
ch4_year_dummies = pd.get_dummies(exp_ch4_rob['Year'])
nox_national_dummies = pd.get_dummies(exp_nox_rob['Country'])
nox_year_dummies = pd.get_dummies(exp_nox_rob['Year'])

xexp_co2_rob = pd.concat([exp_co2_rob, co2_national_dummies, co2_year_dummies], axis = 1).drop(['co2_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)
xexp_ch4_rob = pd.concat([exp_ch4_rob, ch4_national_dummies, ch4_year_dummies], axis = 1).drop(['ch4_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)
xexp_nox_rob = pd.concat([exp_nox_rob, nox_national_dummies, nox_year_dummies], axis = 1).drop(['nox_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)

imp_co2_rob = data[['co2_intensity', 'co2_intensity_init', 'co2_intensity_lag', 'co2_tech', 'IMP_CO2_ROB', 'Country', 'Year']].dropna()
imp_ch4_rob = data[['ch4_intensity', 'ch4_intensity_init', 'ch4_intensity_lag', 'ch4_tech', 'IMP_CH4_ROB', 'Country', 'Year']].dropna()
imp_nox_rob = data[['nox_intensity', 'nox_intensity_init', 'nox_intensity_lag', 'nox_tech', 'IMP_NOX_ROB', 'Country', 'Year']].dropna()

co2_national_dummies = pd.get_dummies(imp_co2_rob['Country'])
co2_year_dummies = pd.get_dummies(imp_co2_rob['Year'])
ch4_national_dummies = pd.get_dummies(imp_ch4_rob['Country'])
ch4_year_dummies = pd.get_dummies(imp_ch4_rob['Year'])
nox_national_dummies = pd.get_dummies(imp_nox_rob['Country'])
nox_year_dummies = pd.get_dummies(imp_nox_rob['Year'])

ximp_co2_rob = pd.concat([imp_co2_rob, co2_national_dummies, co2_year_dummies], axis = 1).drop(['co2_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)
ximp_ch4_rob = pd.concat([imp_ch4_rob, ch4_national_dummies, ch4_year_dummies], axis = 1).drop(['ch4_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)
ximp_nox_rob = pd.concat([imp_nox_rob, nox_national_dummies, nox_year_dummies], axis = 1).drop(['nox_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)

# Run network effects regressions for intensities

tc_co2_rob_mod = stats.OLS(tc_co2_rob['co2_intensity'].astype(float), stats.add_constant(xtc_co2_rob).astype(float))
tc_ch4_rob_mod = stats.OLS(tc_ch4_rob['ch4_intensity'].astype(float), stats.add_constant(xtc_ch4_rob).astype(float))
tc_nox_rob_mod = stats.OLS(tc_nox_rob['nox_intensity'].astype(float), stats.add_constant(xtc_nox_rob).astype(float))

exp_co2_rob_mod = stats.OLS(exp_co2_rob['co2_intensity'].astype(float), stats.add_constant(xexp_co2_rob).astype(float))
exp_ch4_rob_mod = stats.OLS(exp_ch4_rob['ch4_intensity'].astype(float), stats.add_constant(xexp_ch4_rob).astype(float))
exp_nox_rob_mod = stats.OLS(exp_nox_rob['nox_intensity'].astype(float), stats.add_constant(xexp_nox_rob).astype(float))

imp_co2_rob_mod = stats.OLS(imp_co2_rob['co2_intensity'].astype(float), stats.add_constant(ximp_co2_rob).astype(float))
imp_ch4_rob_mod = stats.OLS(imp_ch4_rob['ch4_intensity'].astype(float), stats.add_constant(ximp_ch4_rob).astype(float))
imp_nox_rob_mod = stats.OLS(imp_nox_rob['nox_intensity'].astype(float), stats.add_constant(ximp_nox_rob).astype(float))

# Write results of regressions to file

co2_models = [tc_co2_rob_mod, exp_co2_rob_mod, imp_co2_rob_mod]
names = ['Competition', 'Exports', 'Imports']
res_list = []

for mod in co2_models:
    
    res = mod.fit(cov_type = 'HC1')
    res_list.append(res)
    print(res.summary())
    file = open('C:/Users/User/Documents/Data/Pollution/Main_CO2_' + names[co2_models.index(mod)] + '.txt', 'w')
    file.write(res.summary().as_text())
    file.close()
    
restab(res_list, 'C:/Users/User/Documents/Data/Pollution/restab_networks_CO2.txt')

ch4_models = [tc_ch4_rob_mod, exp_ch4_rob_mod, imp_ch4_rob_mod]
res_list = []

for mod in ch4_models:
    
    res = mod.fit(cov_type = 'HC1')
    res_list.append(res)
    print(res.summary())
    file = open('C:/Users/User/Documents/Data/Pollution/Main_CH4_' + names[ch4_models.index(mod)] + '.txt', 'w')
    file.write(res.summary().as_text())
    file.close()
    
restab(res_list, 'C:/Users/User/Documents/Data/Pollution/restab_networks_CH4.txt')

nox_models = [tc_nox_rob_mod, exp_nox_rob_mod, imp_nox_rob_mod]
res_list = []

for mod in nox_models:
    
    res = mod.fit(cov_type = 'HC1')
    res_list.append(res)
    print(res.summary())
    file = open('C:/Users/User/Documents/Data/Pollution/Main_NOX_' + names[nox_models.index(mod)] + '.txt', 'w')
    file.write(res.summary().as_text())
    file.close()
    
restab(res_list, 'C:/Users/User/Documents/Data/Pollution/restab_networks_NOX.txt')

# Geographical regression

geo_co2_rob = data[['co2_intensity', 'co2_intensity_init', 'co2_intensity_lag', 'co2_tech', 'GEO_CO2', 'Country', 'Year']].dropna()
co2_national_dummies = pd.get_dummies(geo_co2_rob['Country'])
co2_year_dummies = pd.get_dummies(geo_co2_rob['Year'])
geo_co2_robx = pd.concat([geo_co2_rob, co2_national_dummies, co2_year_dummies], axis = 1).drop(['co2_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)
geo_co2_rob_mod = stats.OLS(geo_co2_rob['co2_intensity'].astype(float), stats.add_constant(geo_co2_robx).astype(float))
res = geo_co2_rob_mod.fit(cov_type = 'HC1')
res_list.append(res)
print(res.summary())
file = open('C:/Users/User/Documents/Data/Pollution/GEO_CO2.txt', 'w')
file.write(res.summary().as_text())
file.close()
   
# Regressions with geographic spillovers and network effects

# GEO + TC

geo_tc_co2_rob = data[['co2_intensity', 'co2_intensity_init', 'co2_intensity_lag', 'co2_tech', 'GEO_CO2', 'TC_CO2_ROB', 'Country', 'Year']].dropna()
co2_national_dummies = pd.get_dummies(geo_tc_co2_rob['Country'])
co2_year_dummies = pd.get_dummies(geo_tc_co2_rob['Year'])
geo_tc_co2_robx = pd.concat([geo_tc_co2_rob, co2_national_dummies, co2_year_dummies], axis = 1).drop(['co2_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)
geo_tc_co2_rob_mod = stats.OLS(geo_tc_co2_rob['co2_intensity'].astype(float), stats.add_constant(geo_tc_co2_robx).astype(float))
res = geo_tc_co2_rob_mod.fit(cov_type = 'HC1')
res_list.append(res)
print(res.summary())
file = open('C:/Users/User/Documents/Data/Pollution/GEO_CO2_TC.txt', 'w')
file.write(res.summary().as_text())
file.close()

# GEO + EXPORTS

geo_exp_co2_rob = data[['co2_intensity', 'co2_intensity_init', 'co2_intensity_lag', 'co2_tech', 'GEO_CO2', 'EXP_CO2_ROB', 'Country', 'Year']].dropna()
co2_national_dummies = pd.get_dummies(geo_exp_co2_rob['Country'])
co2_year_dummies = pd.get_dummies(geo_exp_co2_rob['Year'])
geo_exp_co2_robx = pd.concat([geo_exp_co2_rob, co2_national_dummies, co2_year_dummies], axis = 1).drop(['co2_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)
geo_exp_co2_rob_mod = stats.OLS(geo_exp_co2_rob['co2_intensity'].astype(float), stats.add_constant(geo_exp_co2_robx).astype(float))
res = geo_exp_co2_rob_mod.fit(cov_type = 'HC1')
res_list.append(res)
print(res.summary())
file = open('C:/Users/User/Documents/Data/Pollution/GEO_CO2_EXP.txt', 'w')
file.write(res.summary().as_text())
file.close()

# GEO + IMPORTS

geo_imp_co2_rob = data[['co2_intensity', 'co2_intensity_init', 'co2_intensity_lag', 'co2_tech', 'GEO_CO2', 'IMP_CO2_ROB', 'Country', 'Year']].dropna()
co2_national_dummies = pd.get_dummies(geo_imp_co2_rob['Country'])
co2_year_dummies = pd.get_dummies(geo_imp_co2_rob['Year'])
geo_imp_co2_robx = pd.concat([geo_imp_co2_rob, co2_national_dummies, co2_year_dummies], axis = 1).drop(['co2_intensity', 'Country', 'Year', 'Zimbabwe', 1971], axis = 1)
geo_imp_co2_rob_mod = stats.OLS(geo_imp_co2_rob['co2_intensity'].astype(float), stats.add_constant(geo_imp_co2_robx).astype(float))
res = geo_imp_co2_rob_mod.fit(cov_type = 'HC1')
res_list.append(res)
print(res.summary())
file = open('C:/Users/User/Documents/Data/Pollution/GEO_CO2_IMP.txt', 'w')
file.write(res.summary().as_text())
file.close()

