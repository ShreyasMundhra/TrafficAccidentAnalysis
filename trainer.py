from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
import numpy as np

def oneHotEncoder(df,filename,cols=None):
    oheDf = pd.get_dummies(data=df,columns=cols)
    # oheDf = pd.DataFrame(data=out,columns=le.classes_)
    # print("OHE columns: ", oheDf.columns)
    # print('OHE size: ', len(oheDf.index))
    # df = pd.concat([df,oheDf])
    # df.drop('BOROUGH')
    oheDf.to_csv(filename, index=False)
    return oheDf

def createBoroughDf(oheDf, filename):
    boroughs = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
    boroughDf = oheDf[["BOROUGH_" + b for b in boroughs]]
    boroughDf.columns = boroughs
    # boroughDf["ACCIDENT"] = pd.Series(data=[1 for i in range(0, len(collisionBoroughDf.index))])
    # print(boroughDf.columns)
    boroughDf.to_csv(filename, index=False)
    return boroughDf

def findProbOfEventInBorough(df):
    n = len(df.index)
    colProbs = {}
    for column in df:
        colProbs[column] = (float)(df[column].sum())/n

    print(colProbs)
    return colProbs

def getBoroughFromCourt(row):
    map = {}
    map["BRONX TVB"] = "BRONX"
    map["BROOKLYN NORTH TVB"] = "BROOKLYN"
    map["BROOKLYN SOUTH TVB"] = "BROOKLYN"
    map["MANHATTAN NORTH TVB"] = "MANHATTAN"
    map["MANHATTAN SOUTH TVB"] = 'MANHATTAN'
    map["RICHMOND TVB"] = "STATEN ISLAND"
    map["QUEENS NORTH TVB"] = "QUEENS"
    map["QUEENS SOUTH TVB"] = "QUEENS"

    return map[row['Court']]


if __name__ == "__main__":
    collisionDf = pd.read_csv("ProcessedData/NYPD_Motor_Vehicle_Collisions.csv", low_memory=False)

    collisionColsList = ['BOROUGH']
    reasons = ['CONTRIBUTING FACTOR VEHICLE ' + str(i) for i in range(1,6)]
    for reason in reasons:
        collisionColsList.append(reason)

    collisionDf = collisionDf[collisionColsList]
    collisionOheDf = oneHotEncoder(collisionDf,'IntermediateData/collision_ohe.csv')
    collisionBoroughDf = createBoroughDf(collisionOheDf,"IntermediateData/collisionBorough.csv")


    # collisionOheDf = pd.read_csv('IntermediateData/collision_ohe.csv')
    # collisionBoroughDf = pd.read_csv("IntermediateData/collisionBorough.csv")

    findProbOfEventInBorough(collisionBoroughDf)

    ticketDf = pd.read_csv("ProcessedData/Traffic_Tickets_Issued__Four_Year_Window - Copy.csv", low_memory=False)
    courts = ["BRONX TVB","BROOKLYN NORTH TVB","BROOKLYN SOUTH TVB","MANHATTAN NORTH TVB","MANHATTAN SOUTH TVB","RICHMOND TVB","QUEENS NORTH TVB","QUEENS SOUTH TVB"]
    # print("Ticket columns: " + str(ticketDf.columns))
    # print("Unique courts: " + str(len(ticketDf['Court'].unique())))

    nycticketDf = ticketDf.loc[ticketDf['Court'].isin(courts)]
    # print("NYC courts: " + str(nycticketDf['Court'].unique()))

    ticketColsList = ['Violation Charged Code','Violation Description','Court']
    nycticketDf = nycticketDf[ticketColsList]
    nycticketDf["BOROUGH"] = nycticketDf.apply(lambda row: getBoroughFromCourt(row), axis=1)
    # print("NYC boroughs: " + str(nycticketDf['BOROUGH'].unique()))
    # print("NYC ticket reasons: " + str(len(nycticketDf['Violation Description'].unique())))
    # nycticketDf.to_csv('ticketdf.csv',index=False)

    ticketOheDf = oneHotEncoder(nycticketDf,'IntermediateData/ticket_ohe.csv',['BOROUGH'])
    ticketBoroughDf = createBoroughDf(ticketOheDf, "IntermediateData/ticketBorough.csv")
    findProbOfEventInBorough(ticketBoroughDf)
