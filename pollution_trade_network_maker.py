# This script converts trade data into weighted annual trade networks

# Importing required modules

import pandas as pd
import numpy as np

# Reading in the trade data

data = pd.read_csv('C:/Users/User/Documents/Data/Pollution/Networks/Dyadic_COW_4.0.csv')

# Converting -9 to 0 (-9 was the code for no data available)

data.loc[data.flow1 == -9, 'flow1'] = 0
data.loc[data.flow2 == -9, 'flow2'] = 0

# Building the trade network matrices for each year

# List of all years we are interested in (this data only goes through 2014)

yrs = [i for i in range(1969,2015)]

# List of all countries in the full data set

countries = sorted(pd.Series(sorted(data.importer1.unique())+sorted(data.importer2.unique())).unique())

# Creating trade networks for each year

for y in yrs:
    
    # Create dataframe for year y
    
    ydata = data[data['year'] == y].reset_index()
    
    # Create an empty matrix to fill in
    
    A = np.zeros((len(countries),len(countries)))
    
    # Filling in the matrix
    
    for row in range(len(ydata)):
        
        i = countries.index(ydata.importer1[row])
        j = countries.index(ydata.importer2[row])
        A[i,j] = ydata.flow2[row]
        A[j,i] = ydata.flow1[row]

    # Saving the trade network as a dataframe
    
    A = pd.DataFrame(A, columns = countries)
    print('Writing ' + str(y) + ' trade network to file.......')
    A.to_csv('C:/Users/User/Documents/Data/Pollution/Networks/A_' + str(y) + '.csv', index = False)
    
