# This script creates the competition intensity values for the weighted total competition matrices

# Importing required modules

import numpy as np
import pandas as pd

# Reading in the data

main_data = pd.read_csv('C:/Users/User/Documents/Data/Pollution/pollution_data.csv')

# Creating a list of all nations

nations = sorted(main_data.Country.unique().tolist())

# Initializing some dataframes

CO2_df = pd.DataFrame()
CH4_df = pd.DataFrame()
NOX_df = pd.DataFrame()
GHG_df = pd.DataFrame()

# Defining two helper functions for subsetting nations to only those with viable data

# This fucntion restricts nations to those with trade network data

def emissions_lists(xxx_nations, ccc_nations, nations):
    
    for c in nations:
        
        if c not in ccc_nations: # this will be comp_nations in our case
            
            xxx_nations.remove(c)
            
    return xxx_nations

# This function further restricts nations to those with intensity data

def extant_intensity(ydat, listy, emission):
    
    listy2 = [l for l in listy]
    
    for n in listy2:
        
        if (ydat[emission][ydat.Country.tolist().index(n)] > 0) == False:
            
            listy.remove(n)
            
    return listy
    
# A list of years to iterate through

yrs = [i for i in range(1970,2015)]

# The main loop

for y in yrs:
    
    # Cute message
    
    print('Creating data for year ' + str(y) + '.......')
    
    # Refresh lists of nations to pare down
    
    comp_nations = sorted(main_data.Country.unique().tolist())
    co2_nations = sorted(main_data.Country.unique().tolist())
    ch4_nations = sorted(main_data.Country.unique().tolist())
    nox_nations = sorted(main_data.Country.unique().tolist())
    ghg_nations = sorted(main_data.Country.unique().tolist())
    
    # Load TC matrix for year y
    
    TC_co2 = pd.read_csv('C:/Users/User/Documents/Data/Pollution/Networks/TC_' + str(y) + '.csv')
    TC_ch4 = pd.read_csv('C:/Users/User/Documents/Data/Pollution/Networks/TC_' + str(y) + '.csv')
    TC_nox = pd.read_csv('C:/Users/User/Documents/Data/Pollution/Networks/TC_' + str(y) + '.csv')
    TC_ghg = pd.read_csv('C:/Users/User/Documents/Data/Pollution/Networks/TC_' + str(y) + '.csv')
    
    # Determining which countries have all data for current year
    
    # Subset to current year
    
    ydata = main_data[main_data['Year'] == y].reset_index().drop('index', axis = 1)
    
    # Check that each country engaged in competitive behavior
    
    for n in nations:
        
        if (n in TC_co2.columns.tolist()) == False:
            
            comp_nations.remove(n)
        
        elif sum(TC_co2[n]) == 0:
            
            comp_nations.remove(n)
            
    # Creating a beginning for emissions lists
    
    co2_nations = emissions_lists(co2_nations, comp_nations, nations)
    ch4_nations = emissions_lists(ch4_nations, comp_nations, nations)
    nox_nations = emissions_lists(nox_nations, comp_nations, nations)
    ghg_nations = emissions_lists(ghg_nations, comp_nations, nations)
    
    # Further paring down emissions lists based on the existence of intensities data
    
    co2_nations = extant_intensity(ydata, co2_nations, 'co2_intensity')
    ch4_nations = extant_intensity(ydata, ch4_nations, 'ch4_intensity')
    nox_nations = extant_intensity(ydata, nox_nations, 'nox_intensity')
    ghg_nations = extant_intensity(ydata, ghg_nations, 'ghg_intensity')
    
    # Remove extra rows and columns from TC - for each intensity
    
    co2_indices = TC_co2.columns.tolist()
    ch4_indices = TC_ch4.columns.tolist()
    nox_indices = TC_nox.columns.tolist()
    ghg_indices = TC_ghg.columns.tolist()
    
    co2_indices.reverse()
    ch4_indices.reverse()
    nox_indices.reverse()
    ghg_indices.reverse()
    
    for col in co2_indices:
        
        if col not in co2_nations:
            
            TC_co2 = TC_co2.drop(TC_co2.columns.tolist().index(col), axis = 0)
            TC_co2 = TC_co2.drop(col, axis = 1)
    
    for col in ch4_indices:
        
        if col not in ch4_nations:
            
            TC_ch4 = TC_ch4.drop(TC_ch4.columns.tolist().index(col), axis = 0)
            TC_ch4 = TC_ch4.drop(col, axis = 1)
    
    for col in nox_indices:
        
        if col not in nox_nations:
            
            TC_nox = TC_nox.drop(TC_nox.columns.tolist().index(col), axis = 0)
            TC_nox = TC_nox.drop(col, axis = 1)
    
    for col in ghg_indices:
        
        if col not in ghg_nations:
            
            TC_ghg = TC_ghg.drop(TC_ghg.columns.tolist().index(col), axis = 0)
            TC_ghg = TC_ghg.drop(col, axis = 1)
    
    # Normalize TC - for each intensity
    
    co2_sums = [sum(TC_co2[col]) for col in TC_co2.columns]
    ch4_sums = [sum(TC_ch4[col]) for col in TC_ch4.columns]
    nox_sums = [sum(TC_nox[col]) for col in TC_nox.columns]
    ghg_sums = [sum(TC_ghg[col]) for col in TC_ghg.columns]
    
    M_co2 = np.matrix(TC_co2)
    M_ch4 = np.matrix(TC_ch4)
    M_nox = np.matrix(TC_nox)
    M_ghg = np.matrix(TC_ghg)
    
    for col in range(len(M_co2)):
        
        M_co2[col,:] = M_co2[col,:] / co2_sums[col]
    
    for col in range(len(M_ch4)):
        
        M_ch4[col,:] = M_ch4[col,:] / ch4_sums[col]
    
    for col in range(len(M_nox)):
        
        M_nox[col,:] = M_nox[col,:] / nox_sums[col]
    
    for col in range(len(M_ghg)):
        
        M_ghg[col,:] = M_ghg[col,:] / ghg_sums[col]
    
    # Create vector of actual emissions intensities
    
    co2_ints = np.matrix([ydata.co2_intensity[ydata.Country.tolist().index(n)] for n in co2_nations]).T
    ch4_ints = np.matrix([ydata.ch4_intensity[ydata.Country.tolist().index(n)] for n in ch4_nations]).T
    nox_ints = np.matrix([ydata.nox_intensity[ydata.Country.tolist().index(n)] for n in nox_nations]).T
    ghg_ints = np.matrix([ydata.ghg_intensity[ydata.Country.tolist().index(n)] for n in ghg_nations]).T
    
    # Multpliy matrix X vector - for each intensity
    
    co2_data = np.matmul(M_co2,co2_ints)
    ch4_data = np.matmul(M_ch4,ch4_ints)
    nox_data = np.matmul(M_nox,nox_ints)
    ghg_data = np.matmul(M_ghg,ghg_ints)
    
    # Append to DataFrame
    
    current_year = [y for c in co2_nations]
    next_year = [y+1 for c in co2_nations]
    co2_d = [x[0] for x in co2_data.tolist()]
    temp_co2 = pd.DataFrame({'Current Year':current_year, 'Next Year':next_year,
                             'Nation':co2_nations, 'CO2 Data':co2_d})
    CO2_df = pd.concat([CO2_df, temp_co2], axis = 0)
    
    current_year = [y for c in ch4_nations]
    next_year = [y+1 for c in ch4_nations]
    ch4_d = [x[0] for x in ch4_data.tolist()]
    temp_ch4 = pd.DataFrame({'Current Year':current_year, 'Next Year':next_year,
                             'Nation':ch4_nations, 'CH4 Data':ch4_d})
    CH4_df = pd.concat([CH4_df, temp_ch4], axis = 0)
    
    current_year = [y for c in nox_nations]
    next_year = [y+1 for c in nox_nations]
    nox_d = [x[0] for x in nox_data.tolist()]
    temp_nox = pd.DataFrame({'Current Year':current_year, 'Next Year':next_year,
                             'Nation':nox_nations, 'NOX Data':nox_d})
    NOX_df = pd.concat([NOX_df, temp_nox], axis = 0)
    
    current_year = [y for c in ghg_nations]
    next_year = [y+1 for c in ghg_nations]
    ghg_d = [x[0] for x in ghg_data.tolist()]
    temp_ghg = pd.DataFrame({'Current Year':current_year, 'Next Year':next_year,
                             'Nation':ghg_nations, 'GHG Data':ghg_d})
    GHG_df = pd.concat([GHG_df, temp_ghg], axis = 0)
    
# Write dataframe to file

CO2_df.to_csv('C:/Users/User/Documents/Data/Pollution/TC_DATA_CO2.csv', index = False)
CH4_df.to_csv('C:/Users/User/Documents/Data/Pollution/TC_DATA_CH4.csv', index = False)
NOX_df.to_csv('C:/Users/User/Documents/Data/Pollution/TC_DATA_NOX.csv', index = False)
GHG_df.to_csv('C:/Users/User/Documents/Data/Pollution/TC_DATA_GHG.csv', index = False)
    
