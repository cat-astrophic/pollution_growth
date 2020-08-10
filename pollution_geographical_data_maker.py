# This script creates the geographic intensity values for the weighted total trade networks

# Importing required modules

import numpy as np
import pandas as pd

# Reading in the data

main_data = pd.read_csv('C:/Users/User/Documents/Data/Pollution/pollution_data.csv')
W = pd.read_csv('C:/Users/User/Documents/Data/Pollution/pollution_data_W_reference.csv', header = None)

# Creating a list of all nations

nations = sorted(main_data.Country.unique().tolist())

# Initializing a dataframe for results

CO2_df = pd.DataFrame()

# Defining a helper function for subsetting nations to only those with viable data

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
    
    co2_nations = sorted(main_data.Country.unique().tolist())
    
    # Determining which countries have all data for current year
    
    # Subset to current year
    
    ydata = main_data[main_data['Year'] == y].reset_index().drop('index', axis = 1)    
    
    # Paring down emissions lists based on the existence of intensities data
    
    co2_nations = extant_intensity(ydata, co2_nations, 'co2_intensity')
    
    # List of nations to remove by indices
    
    bye = [i for i in range(len(nations)) if nations[i] not in co2_nations]
    
    # Remove extra rows and columns from W and normalizing
    
    X = W.drop(W.index[[bye]])
    X = X.drop(bye, axis = 1)
    X = X.values
    X_sums = [max(1,sum(row)) for row in X]
    X = (X / X_sums).T
    
    # Create vector of actual emissions intensities
    
    co2_ints = [ydata.co2_intensity[ydata.Country.tolist().index(n)] for n in co2_nations]
    I_co2 = np.zeros((len(co2_ints),len(co2_ints)))
    
    for i in range(len(I_co2)):
        
        for j in range(len(I_co2)):
            
            I_co2[i,j] = np.log(co2_ints[j] / co2_ints[i])
            
    # Create network effect data
    
    co2_data = [sum([X[col,row]*I_co2[row,col] for col in range(len(I_co2))]) for row in range(len(I_co2))]
    
    # Append to DataFrame
    
    current_year = [y for c in co2_nations]
    next_year = [y+1 for c in co2_nations]
    temp_co2 = pd.DataFrame({'Current Year':current_year, 'Next Year':next_year,
                             'Nation':co2_nations, 'CO2 Data':co2_data})
    CO2_df = pd.concat([CO2_df, temp_co2], axis = 0)
    
# Write dataframe to file

CO2_df.to_csv('C:/Users/User/Documents/Data/Pollution/GEOGRAPHICAL_DATA_CO2.csv', index = False)

