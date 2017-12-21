'''
    Enviornment Requirement:
    Install package
    pip install reverse_geocoder
'''

import pandas as pd
import numpy as np
import reverse_geocoder as rg

df = pd.read_csv("NYPD_Motor_Vehicle_Collisions.csv")


processedDf = pd.DataFrame()
processedDf['Borough'] = df['BOROUGH']
processedDf['Zip'] = df['ZIP CODE']
processedDf['latitude'] = df['LATITUDE']
processedDf['longitude'] = df['LONGITUDE']
processedDf['on street name'] = df['ON STREET NAME']
processedDf['cross street name'] = df['CROSS STREET NAME']
processedDf['off street name'] = df['OFF STREET NAME']

trainset = processedDf.dropna()
testset = processedDf[(processedDf['Borough'].isnull()) & (processedDf['latitude'].notnull())]
#testset = testset[testset['latitude'].notnull()]
len(testset.index)
'''
 I split the dataset into 1410 subsets, 
 1000 items left each,
 it takes approximately 20s to process one subset
'''
testsub = np.split(testset, [375], axis=0)
testsubset = np.split(testsub[1], 1410, axis=0)
testsubset[1409].index


def getAddress(coordinate):
    location = rg.search(coordinate)
    if location[0]['admin1'] is "New Jersey":
        print("return, admin1 is New Jersey")
        return
    if location[0]['admin2'] is "":
        print("return, admin2 is null")
        return
    borough = None
    if "New York County" in location[0]['admin2']:
        borough = "MANHATTAN"
    elif "Bronx" in location[0]['admin2']:
        borough = "BRONX"
    elif "Kings County" in location[0]['admin2']:
        borough = "BROOKLYN"
    elif "Queens County" or "Nassau County" in location[0]['admin2']:
        borough = "QUEENS"
    return borough

'''
 Change the range, can test one first to see the output, 
 I have finished 0 to 400, 
 and I am planning to train 400 to 800.
 Maybe you can carry on from 800 to 1000? We can keep each other updated with current situations.
'''
for sutnum in range(300, 400):
    for addresses in testsubset[sutnum].iterrows():
        lat = addresses[1][2]
        log = addresses[1][3]
        coordinate = [lat, log]
        coordinate = tuple(coordinate)
        borough = getAddress(coordinate)
        processedDf.loc[addresses[0], ['Borough']] = borough
        print(addresses[0])
    print("finish dataset", sutnum)

'''
 Save the processed processedDf to a csv
 This method is not efficient, because we have to add them together at the end
 But It's faster to run a .py instead of jupyter notebook,
 and I have to save the processedDf as an output for each run
'''
# Remember to change the file name, or new will overwrite the old ones
csv = processedDf.to_csv('subset_300_400.csv')
