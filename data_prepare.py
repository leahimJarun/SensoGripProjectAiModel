"""
Prepare data for further process.

Read data from "Schulkinder" Data and save each dataset ina python dict. Also add a "SEMS" Column for each dataset corresponding to the SEMS score from "SEMS_Werte.xlsx"

"""

import csv
import json
import os
import random
import argparse
from pandas import *


def generateDict(dataSetDict,semsDataframe):
    dictList = []

    for dictEntry in list(dataSetDict):
    
        datasetDF = read_csv('Daten_Schulkinder/Datasets/'+dictEntry)
        datasetDF = datasetDF.drop(['timestamp','tipUpperRange','tipLowerRange','fingerUpperRange','fingerLowerRange'],axis=1)
        datasetDF = datasetDF[datasetDF.tipPressure!=0]# Delete all rows with 0 in tippressure
        datasetDF = datasetDF.transpose()
        if dataSetDict[dictEntry] in semsDataframe['Unnamed: 0'].array:
            SEMS = str(semsDataframe[semsDataframe['Unnamed: 0']==dataSetDict[dictEntry]]['SEMS'].iloc[0])
        else:
            SEMS = 'NEGATIVE'
        dic = datasetDF.to_dict()
        transdic = {"SEMS":SEMS,"rows":[]}
        for idx, row in dic.items():
            itemslist = []
            for colum, value in row.items():
                itemslist.append(value)
            transdic["rows"].append(itemslist)        
        dictList.append(transdic)

    return dictList

def makeDataSetDict(dataSetList):
    dataSetDict = {}

    for dataset in dataSetList:
        semsKeyName = dataset.split("-",1)
        semsKeyName = semsKeyName[0]

        semsNumber = semsKeyName.split("_")[1]

        if(semsNumber=="SEMS2"): 
            semsNumber="_2"
        else:
            semsNumber=""
        
        semsKeyName = semsKeyName.split("_")[0]+semsNumber

        dataSetDict[dataset] = semsKeyName
    
    return dataSetDict

def splitData(dictList,trainProp,validProp):
    datasetNumber = len(dictList)
    numberTrainSets = int(datasetNumber*trainProp)
    numberValidSets = int(datasetNumber*validProp)
    numberTestSets = int(datasetNumber*(1-(trainProp+validProp)))

    numberTrainSets = numberTrainSets + datasetNumber-(numberTestSets+numberTrainSets+numberValidSets)

    trainList = []
    validList= []
    testList = []

    for indx, dict in enumerate(dictList):
        if(indx<=numberTrainSets-1):
            trainList.append(dict)
        if(indx>numberTrainSets-1 and indx<=(numberTrainSets+numberValidSets-1)):
            validList.append(dict)
        if(indx>(numberTrainSets+numberValidSets-1)):
            testList.append(dict)

    return trainList,validList,testList

def prepareData(neg=False,trainProp=0.4,validProp=0.4):
    
    """Import the SEMS values workbook into a dataframe"""
    xls = ExcelFile('Daten_Schulkinder/SEMS_Werte.xlsx')
    semsDataframe = xls.parse(xls.sheet_names[0])

    """Fetch all of the dataset file names from directory"""

    datasetList = sorted(os.listdir('Daten_Schulkinder/Datasets'))
 
    """make dict with dataset name + its key in SEMS value dataframe"""

    dataSetDict = makeDataSetDict(datasetList)    
    

    """delet any entries in the dict where the key is not found in the SEMS value dataframe"""

    for dictEntry in list(dataSetDict):
        if dataSetDict[dictEntry] in semsDataframe['Unnamed: 0'].values:
            print(dataSetDict[dictEntry]+" dataset is present in SEMS Values")
        else:
            print(dataSetDict[dictEntry]+" dataset is not present in SEMS Values")
            dataSetDict.pop(dictEntry)

    """Convert each dataset into a dict and append to list of dictionaries"""
    
    dictList = generateDict(dataSetDict,semsDataframe)

    """Split Data into train,validation and test data Approx train=50%,validation=25%,test=25%"""
    trainList,validList,testList = splitData(dictList,trainProp,validProp)

    """export dicts into json"""

    writeData(trainList,'Data/train/train.json')
    writeData(validList,'Data/valid/valid.json')
    writeData(testList,'Data/test/test.json')


    if(neg):

        """Do the same for data including NEGATIVE datasets"""
        dataSetDictNeg = makeDataSetDict(datasetList)

        dictListNeg = generateDict(dataSetDictNeg,semsDataframe)

        trainList,validList,testList = splitData(dictListNeg,trainProp,validProp)

        writeData(trainList,'DataNeg/train/train.json')
        writeData(validList,'DataNeg/valid/valid.json')
        writeData(testList,'DataNeg/test/test.json')

def writeData(dictList,Path):
    with open(Path, "w") as f:
        for idx, item in enumerate(dictList):
            dic = json.dumps(item, ensure_ascii=False)
            f.write(dic)
            f.write("\n")

def read_data(path):
  data = [] 
  with open(path, "r") as f:
    lines = f.readlines()
    for idx, line in enumerate(lines): 
      dic = json.loads(line)
      data.append(dic)
  print("data_length:" + str(len(data)))
  return data



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--neg", "-n")
    parser.add_argument("--train", "-t")
    parser.add_argument("--valid", "-v")
    args = parser.parse_args()    

    prepareData(bool(args.neg),float(args.train),float(args.valid))
    #TODO Arguments einbauen
    # z.B mit NEG und nicht, Anteil an Train, Valid, Test data
