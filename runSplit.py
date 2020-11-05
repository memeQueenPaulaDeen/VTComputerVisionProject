
from os import listdir
from os.path import isfile, join
import os
import sys
import pandas as pd

if __name__ == '__main__':

	pwd = os.path.dirname(os.path.abspath(sys.argv[0]))

	dataPath = os.path.join(pwd,'labels')
	splitPath = os.path.join(pwd,'testTrainSplits')
	print(dataPath)
	firstNpercent = 80
	trainDest = join(splitPath,'train80.csv')
	testDest = join(splitPath,'test80.csv')

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
