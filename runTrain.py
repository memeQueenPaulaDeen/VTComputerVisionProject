from os import listdir
from os.path import isfile, join
import os
import sys
import pandas as pd


def getLabelDF(labelPath):
	labelValIdxDict = {'label': 0, 'truncLvl': 1, 'occLvl': 2,
	                   'obvAngleRad': 3, 'bbl': 4, 'bbt': 5, 'bbr': 6, 'bbb': 7, 'height': 8, 'width': 9,
	                   'length': 10, 'x': 11, 'y': 12, 'z': 13, 'rotYRad': 14}
	df = pd.DataFrame()
	for filename in os.listdir(labelPath):
		# print(filename)
		tdf = pd.DataFrame(columns=labelValIdxDict)
		tdf = pd.read_csv(os.path.join(labelPath , filename), delimiter=' ', names=list(labelValIdxDict.keys()), header=None)
		tdf['img'] = filename[:-4]
		df = df.append(tdf)



def int2ImgPath(path,i):
	return os.path.join(path, str(i).rjust(6,'0')+".png")



def generateYoloTrainFile(labeldf,imagePath,yoloInputFile):

	def writeTrainingFile(file, img, imgpath, encoding, left, right, top, bot, label):
		string = int2ImgPath(imgpath, img) + ' ' + str(int(left)) + ',' + str(int(top)) + ',' + str(
			int(right)) + ',' + str(int(bot)) + ',' + str(encoding[label])
		buildLine(img, imgpath, string, file)

	# abusing scope
	def buildLine(img, imgpath, string, file):
		if last[0] == img or firstTime[0]:
			firstTime[0] = False

			# check if the image part should be added
			if lastString[0].__contains__(int2ImgPath(imgpath, img)):
				string = ' ' + string.split(' ')[1]

			string = lastString[0] + string
			last[0] = img
			lastString[0] = string
			return string
		else:
			file.write(lastString[0] + '\n')
			last[0] = img
			lastString[0] = string


	last = ['']
	lastString = ['']
	firstTime = [True]
	file = open(yoloInputFile,'w')
	labeldf.apply(lambda x: writeTrainingFile(file,x.img,imagePath,encoding,x.bbl,x.bbr,x.bbt,x.bbb,x.label),axis=1)
	file.close()

if __name__ == '__main__':

	pwd = os.path.dirname(os.path.abspath(sys.argv[0]))

	weights_file = "yolov3.weights"
	h5_file = "yolo.h5"
	cfg_file = "yolov3.cfg"

	labelPath = os.path.join(pwd, 'labels')
	trainLabelPath = os.path.join(pwd,'testTrainSplits')
	imagePath = join(pwd,'image_2')
	logPath = join(pwd,'modelWeights')
	imageClassFile = join(pwd,'TrainYourOwnYOLO-kitti','data_classes.txt')

	listOfLablesToTrainOnCSV = join(trainLabelPath,'train80.csv')
	yoloInputFile = join(trainLabelPath,'yoloInputFile.txt')

	trainScriptPath = join(pwd,'TrainYourOwnYOLO','2_Training')

	train = pd.read_csv(listOfLablesToTrainOnCSV).iloc[:,0].to_list()


	#labeldf = getLabelDF(labelPath)
	labeldf = pd.read_csv(join(pwd,'TrainYourOwnYOLO-kitti','cache.csv'))
	labeldf = labeldf.sort_values('img')

	encoding = {
		"Car": 0,
		"Van": 1,
		"Truck": 2,
		"Pedestrian": 3,
		"Person_sitting": 4,
		"Cyclist": 5,
		"Tram": 6,
		"Misc": 7,
		"DontCare": 8,
	}

	generateYoloTrainFile(labeldf,imagePath,yoloInputFile)


	command = 'python3 '+ join(trainScriptPath,'Train_YOLO.py') + ' --annotation_file ' + yoloInputFile + \
	          ' --classes_file ' +imageClassFile + ' --log_dir ' + logPath

	os.system(command)

