
from os import listdir
from os.path import isfile, join
import os
import sys
import pandas as pd

if __name__ == '__main__':

	dataPathIdx = 1
	firstNPercentIdx = 2
	trainDestinationIdx = 3
	testDestinationIdx = 4
	dataPath = sys.argv[dataPathIdx]
	firstNpercent = int(sys.argv[firstNPercentIdx])
	trainDest = sys.argv[trainDestinationIdx]
	testDest = sys.argv[testDestinationIdx]

	onlyfiles = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
	cutIdx = int(firstNpercent/100*len(onlyfiles))
	print(cutIdx)
	train = sorted(onlyfiles)[:cutIdx]
	test = sorted(onlyfiles)[cutIdx:]

	# print(onlyfiles)
	# print(train)
	# print(test)

	pd.DataFrame(train).to_csv(trainDest,sep=os.linesep,header=False,index=False)
	pd.DataFrame(test).to_csv(testDest,sep=os.linesep,header=False,index=False)

