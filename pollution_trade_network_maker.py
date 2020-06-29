# This script converts trade data into weighted annual trade networks

# Importing required modules

import pandas as pd
import numpy as np

# Reading in the trade data

print('Reading in the raw data.......')
data = pd.read_csv('C:/Users/User/Documents/Data/Pollution/Networks/Dyadic_COW_4.0.csv')

# Converting -9 to 0 (-9 was the code for no data available)

data.loc[data.flow1 == -9, 'flow1'] = 0
data.loc[data.flow2 == -9, 'flow2'] = 0

# Building the trade network matrices for each year

# List of all years we are interested in (this data only goes through 2014)

yrs = [i for i in range(1969,2015)]

# List of all countries in the full data set

countries = sorted(pd.Series(sorted(data.importer1.unique())+sorted(data.importer2.unique())).unique())

# A duplicate which will be updated to match the primary data set

countries_matched = sorted(pd.Series(sorted(data.importer1.unique())+sorted(data.importer2.unique())).unique())

# A dictionary of country names which differ across data sets
# The structure of this dictionary is as follows: {main data set: network data}

dic = {'Antigua & Barbuda':'Antigua and Barbuda', 'Bahamas':'Bahamas, The',
       'Brunei':'Brunei Darussalam', 'Cape Verde':'Cabo Verde',
       'Democratic Republic of the Congo':'Congo, Dem. Rep.',
       'Congo':'Congo, Rep.', 'Ivory Coast':"Cote d'Ivoire",
       'Egypt':'Egypt, Arab Rep.', 'Swaziland':'Eswatini',
       'Gambia':'Gambia, The', 'Iran':'Iran, Islamic Rep.',
       'North Korea':'Korea, Dem. People’s Rep.', 'South Korea':'Korea, Rep.',
       'Kyrgyzstan':'Kyrgyz Republic', 'Laos':'Lao PDR',
       'Federated States of Micronesia':'Micronesia, Fed. Sts.',
       'Macedonia':'North Macedonia', 'Russia':'Russian Federation',
       'Slovakia':'Slovak Republic', 'Syria':'Syrian Arab Republic',
       'East Timor':'Timor-Leste', 'United States of America':'United States',
       'Venezuela':'Venezuela, RB', 'Yemen':'Yemen, Rep.'}

# Rename the countries as appropriate

for c in countries_matched:
    
    if c in dic.keys():
        
        countries_matched[countries_matched.index(c)] = dic[c]
        
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
    
    A = pd.DataFrame(A, columns = countries_matched)
    print('Writing ' + str(y) + ' weighted trade network to file.......')
    A.to_csv('C:/Users/User/Documents/Data/Pollution/Networks/W_' + str(y) + '.csv', index = False)
    
    # Making and saving a binary version
    
    A[A > 0] = 1
    print('Writing ' + str(y) + ' binary trade network to file.......')
    A.to_csv('C:/Users/User/Documents/Data/Pollution/Networks/A_' + str(y) + '.csv', index = False)

    # Making and saving two versions of the competition graphs: total competing markets and binary
    
    # Create an empty matrix to fill in for total competing markets
    
    C = np.zeros((len(countries),len(countries)))
    
    # Create the adjacency matrix of the total competition graph
    
    for col in A.columns:
        
        # For each column, the non-zero entries is the set of competitors for exporting to A[col] 
        
        l = [i for i in range(len(A)) if A[col][i] > 0]
        
        for i in range(len(l)-1):
            
            for j in range(i+1,len(l)):
                
                C[l[i],l[j]] += 1
                C[l[j],l[i]] += 1
                
    # Saving the total competition graph to file as a dataframe
    
    C = pd.DataFrame(C, columns = countries_matched)
    print('Writing ' + str(y) + ' total competition graph to file.......')
    C.to_csv('C:/Users/User/Documents/Data/Pollution/Networks/TC_' + str(y) + '.csv', index = False)
    
    # Making and saving a binary version
    
    C[C > 0] = 1
    print('Writing ' + str(y) + ' competition graph to file.......')
    C.to_csv('C:/Users/User/Documents/Data/Pollution/Networks/C_' + str(y) + '.csv', index = False)

