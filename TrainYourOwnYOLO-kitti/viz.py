from bounding_box import bounding_box as bb
import pandas as pd
import os
import cv2
import copy
import swifter
import numpy as np

def int2ImgPath(path,i):
	return path + str(i).rjust(6,'0')+".png"

def addBox(row,image,color,labelName='label'):
	bb.add(image,row.bbl,row.bbt,row.bbr,row.bbb,row[labelName],color)

def move(deltax,deltay,df,asNewRow=False,bblist =None):
	df['move'] = False
	if bblist != None:
		df.loc[df['bbid'].isin(bblist),'move'] =True
	if asNewRow:
		tdf = copy.deepcopy(df)
	else:
		tdf = df
	tdf.bbl = tdf.apply(lambda x: mv(deltax,x.bbl,x.move),axis=1)
	tdf.bbr = tdf.apply(lambda x: mv(deltax,x.bbr,x.move),axis=1)

	tdf.bbt = tdf.apply(lambda x: mv(deltay,x.bbt,x.move),axis=1)
	tdf.bbb = tdf.apply(lambda x: mv(deltay,x.bbb,x.move),axis=1)

	if asNewRow:
		df = df.append(tdf)
		df['bbid'] = df.groupby((df['img'] != df['img'].shift(1)).cumsum()).cumcount() + 1
		return df
	else:
		return df



def mv(delta,bound,moveBool):
	if moveBool:
		return bound+delta
	else:
		return bound

def findOverLap(yrow, hdf, ioudf,ydf,progress=None):
	#given a ground truth bounding box and label in y [a row in y]
	#iterate over all hypothisisezed bounding boxes [all rows in h that match image num in y]
	#create a mapping between the IOU of all hypothisised bounding boxes and a single ground truth bb
	#The larest value in the keyset will map to the best bbid of all hypothisised bb
	#Save the iou and coresponding IOU bb as well as the mapping between h and y
	#The bbid for y is saved as bbid its predicted coresponding bbid in the set of predictions is mid
	#SO that IOUDF can be joined with ground truth on img and bbid
	#so that IOUDF can be joined with predictionos on img and mid
	#so that predictions and ground truth could be joined on img and bbid to mid [not directly]

	if progress != None:
		print(progress)
		progress[0] = progress[0] + 1
	#not inveraent to order!!!
	hdf = hdf.loc[hdf.img == yrow.img]
	if type(yrow.bbl) == pd.Series:
		bb1 = {'x1': yrow.bbl.values[0], 'x2': yrow.bbr.values[0], 'y1': yrow.bbt.values[0], 'y2': hrow.bbb.values[0]}
	else:
		bb1 = {'x1': yrow.bbl, 'x2': yrow.bbr, 'y1': yrow.bbt, 'y2': yrow.bbb}
	mapping = {}
	#build a mapping of overlaps
	iouList = []
	for id in hdf.bbid:
		hrow = hdf.loc[hdf.bbid == id]
		if type(hrow.bbl) == pd.Series:
			bb2 = {'x1': hrow.bbl.values[0], 'x2': hrow.bbr.values[0], 'y1': hrow.bbt.values[0], 'y2': hrow.bbb.values[0]}
		else:
			bb2 = {'x1': hrow.bbl, 'x2': hrow.bbr, 'y1': hrow.bbt, 'y2': hrow.bbb}
		iou ,bbl, bbr, bbt, bbb = get_iou(bb1,bb2)
		iouList.append((id,iou)) #just for debug
		if 'mid' in ioudf.columns.to_list() and id in ioudf.mid.to_list():
			if all(ioudf.loc[(ioudf.img == yrow.img)&(ioudf.mid == id)].iou < iou):
				mapping[iou] = id
			#else dont add
		else:
			if iou != 0:
				mapping[iou] = id
	if len(mapping) == 0:
		#no valid mapping exists none for row will drop all nans in next step
		ioudf.loc[(ioudf.img == yrow.img) & (ioudf.bbid == yrow.bbid)] = np.nan
		return
	matchid  = max(mapping.keys())
	matchid = mapping[matchid]
	hrow = hdf.loc[hdf.bbid == matchid]
	hclass = hrow.label
	if type(hrow.bbl) == pd.Series:
		bb2 = {'x1': hrow.bbl.values[0], 'x2': hrow.bbr.values[0], 'y1': hrow.bbt.values[0], 'y2': hrow.bbb.values[0]}
		hclass = hrow.label.values[0]
	else:
		bb2 = {'x1': hrow.bbl, 'x2': hrow.bbr, 'y1': hrow.bbt, 'y2': hrow.bbb}
		hclass = hrow.label
	iou, bbl, bbr, bbt, bbb = get_iou(bb1, bb2)

	ioudf.loc[(ioudf.img == yrow.img) & (ioudf.bbid == yrow.bbid), 'iou'] = iou
	ioudf.loc[(ioudf.img == yrow.img) & (ioudf.bbid == yrow.bbid), 'bbl'] = bbl
	ioudf.loc[(ioudf.img == yrow.img) & (ioudf.bbid == yrow.bbid), 'bbr'] = bbr
	ioudf.loc[(ioudf.img == yrow.img) & (ioudf.bbid == yrow.bbid), 'bbt'] = bbt
	ioudf.loc[(ioudf.img == yrow.img) & (ioudf.bbid == yrow.bbid), 'bbb'] = bbb
	ioudf.loc[(ioudf.img == yrow.img) & (ioudf.bbid == yrow.bbid), 'mid'] = matchid
	ioudf.loc[(ioudf.img == yrow.img) & (ioudf.bbid == yrow.bbid), 'hlabel'] = hclass

	#match the hypothosis bb to the ground truth with which it has the most overlap
	if len(ioudf.loc[(ioudf.img == yrow.img)&(ioudf['mid'] == matchid)]) > 1 and matchid != None:
		#conflict must be resolved
		cdf = ioudf.loc[(ioudf.img == yrow.img) & (ioudf['mid']==matchid)]
		winner = cdf.loc[cdf.iou == cdf.iou.max()].mid.to_list()[0]
		if all(winner == x for x in cdf.loc[cdf.iou == cdf.iou.max()].mid.to_list()):
			conf = ioudf.loc[(ioudf.img == yrow.img) & (ioudf.mid == winner)]
			l = conf.loc[(conf.iou == conf.iou.min())]
			l = l.reset_index(drop=True)
			# need to remove hypotisis and recalc for y
			ydf = ydf.loc[ydf.img == yrow.img].reset_index(drop=True)
			ydf = ydf.reset_index(drop=True)
			yrow = ydf.loc[(ydf.img.isin(l.img)) & (ydf.bbid.isin( l.bbid))]
			yrow = yrow.iloc[0]
			findOverLap(yrow, hdf.loc[hdf.bbid != winner], ioudf, ydf.loc[ydf.img == yrow.img])
		else:
			findOverLap(yrow, hdf.loc[hdf.bbid != winner], ioudf,ydf.loc[ydf.img == yrow.img])


	# if len(ioudf.loc[(ioudf.img == yrow.img)&(ioudf['mid'] == matchid)]) > 1 and matchid != None:
	# 	cdf = ioudf.loc[(ioudf.img == yrow.img) & (ioudf['mid'] == matchid)]
	# 	winner = cdf.loc[cdf.iou == cdf.iou.max()].mid.to_list()[0]
	# 	findOverLap(yrow, hdf.loc[hdf.bbid != winner], ioudf, ydf.loc[ydf.img == yrow.img])
		#findOverLap(yrow, hdf.loc[hdf.bbid != winner], ioudf.loc[ioudf.img == yrow.img], ydf.loc[ydf.img == yrow.img])
	#print("IOU " +str(iou))
	#print(yrow.img)



def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]

    taken from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0, None, None, None, None

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0

    bbl = x_left
    bbr = x_right
    bbt = y_top
    bbb = y_bottom

    return iou ,bbl, bbr, bbt, bbb


def plotIOUOverlapForImageNum(imageNum,df,hdf,ioudf):
	y = df.loc[df['img'] == imageNum]
	# y = move(70,-55,y,asNewRow=True)
	yimage = cv2.imread(int2ImgPath(imagePath, imageNum), cv2.IMREAD_COLOR)
	y.apply(lambda x: addBox(x, yimage, 'red'), axis=1)
	h = hdf.loc[hdf['img'] == imageNum]
	# h =df.loc[df['img']==imageNum]
	# h = move(0,40,h,bblist=[1])
	# h = move(-90,70,h,asNewRow=True)
	# himage = cv2.imread(int2ImgPath(imagePath,imageNum), cv2.IMREAD_COLOR)
	h.apply(lambda x: addBox(x, yimage, 'green'), axis=1)
	# ioudf = copy.deepcopy(y)
	# y.apply(lambda x: findOverLap(x, h, ioudf),axis=1)
	# remove any invalid matches
	# ioudf.dropna(how='any',inplace = True)
	i = ioudf.loc[ioudf['img'] == imageNum]
	print(i)
	print(h.confidence)
	i.apply(lambda x: addBox(x, yimage, 'yellow', 'hlabel'), axis=1)
	# agg = ioudf[['label','iou']].groupby('label')
	# import matplotlib.pyplot as plt
	cv2.imshow('Y', yimage)
	# cv2.imshow('H, IOY', himage)
	cv2.waitKey(0)

def matchIOUrowwithYdf(iouRow,mdf):
	mdf.loc[(iouRow.img == mdf.img) & (iouRow.mid == mdf.bbid)].bbb = iouRow.bbb
	mdf.loc[(iouRow.img == mdf.img) & (iouRow.mid == mdf.bbid)].bbt = iouRow.bbt
	mdf.loc[(iouRow.img == mdf.img) & (iouRow.mid == mdf.bbid)].bbr = iouRow.bbr
	mdf.loc[(iouRow.img == mdf.img) & (iouRow.mid == mdf.bbid)].bbl = iouRow.bbl
	mdf.loc[(iouRow.img == mdf.img) & (iouRow.mid == mdf.bbid)]['mid'] = iouRow.mid
	mdf.loc[(iouRow.img == mdf.img) & (iouRow.mid == mdf.bbid)]['hlabel'] = iouRow.hlabel
	mdf.loc[(iouRow.img == mdf.img) & (iouRow.mid == mdf.bbid)]['iou'] = iouRow.iou

def addFalsePositive(hdf,mdf,hrow):
	bblist = mdf.loc[mdf.img == hrow.img]['bbid']
	himg = hdf.loc[hdf.img == hrow.img]
	falsePositves = himg.loc[~himg.bbid.isin(bblist)]
	if len(falsePositves) > 0:
		print(falsePositves)
		mdf = mdf.append(hrow)


def constructIOUdf(df):
	ioudf = copy.deepcopy(df[['bbl', 'bbt', 'bbr', 'bbb', 'img', 'bbid']])
	p = [0]
	df.apply(lambda x: findOverLap(x, hdf, ioudf, df, progress=p), axis=1)
	# remove any invalid matches
	ioudf.dropna(how='any', inplace=True)
	return ioudf


def convertLablesToDFCSV(labelPath,labelValIdxDict):
	global df
	df = pd.DataFrame()
	for filename in os.listdir(labelPath):
		# print(filename)
		tdf = pd.DataFrame(columns=labelValIdxDict)
		tdf = pd.read_csv(labelPath + filename, delimiter=' ', names=list(labelValIdxDict.keys()), header=None)
		tdf['img'] = filename[:-4]
		df = df.append(tdf)
	df.to_csv('cache.csv')


if __name__ == '__main__':

	labelValIdxDict = {'label':0, 'truncLvl':1, 'occLvl':2,
	                   'obvAngleRad':3 ,'bbl':4, 'bbt':5, 'bbr':6, 'bbb':7,'height':8,'width':9,
	                   'length':10, 'x':11, 'y':12, 'z':13,'rotYRad':14}
	classMapping = { 0:"Car", 1:"Van", 2:"Truck", 3:"Pedestrian", 4:"Person_sitting",
                    5:"Cyclist", 6:"Tram", 7:"Misc", 8:"DontCare" }

	labelPath = '../labels/'
	detectionsPath = '../'
	#imagePath = '../images/data_object_image_2/testing/image_2/'
	imagePath = '../image_2/'

	detectionFile = 'Detection_Results.csv'
	hdf = pd.read_csv(detectionsPath+detectionFile)
	hdf['label'] = hdf.apply(lambda x: classMapping[x.label],axis=1)
	#y axis is fliped so re flipping here for consistancy bbt is the highest in visual image
	hdf = hdf.rename(columns={'image':'img','xmin':'bbl', 'ymin':'bbt', 'xmax':'bbr', 'ymax':'bbb'})
	hdf['img'] = hdf.apply(lambda x: int(x.img[:-4]),axis=1)
	hdf = hdf.sort_values(['img'])
	hdf['bbid'] = hdf.groupby((hdf['img'] != hdf['img'].shift(1)).cumsum()).cumcount() + 1

	print(hdf)

	convertLablesToDFCSV(labelPath,labelValIdxDict)

	df = pd.read_csv('cache.csv')
	df.rename(columns={'Unnamed: 0': 'bbid'}, inplace=True)

	print(df)

	#drop all dont cares
	hdf = hdf[hdf['label']!="DontCare"]
	df = df[df['label']!="DontCare"]

	ioudf = constructIOUdf(df)
	ioudf.to_csv('cacheIOU.csv')


	ioudf = pd.read_csv('cacheIOU.csv',index_col='Unnamed: 0')
	ioudf.mid = ioudf.mid.astype(int)


	df['mid'] = df['bbid']


	mdf = df.set_index(['img','mid'],drop=False)
	ioudf = ioudf.set_index(['img','mid'],drop=False)

	ioudf= ioudf.sort_values(by='iou')

	print(len(df))
	print(len(mdf))
	print(len(ioudf))

	mdf = mdf.merge(ioudf,suffixes=('','h'),how="left",left_index=True,right_index=True)

	print(len(df))
	print(len(mdf))
	print(len(ioudf))

	print()


	# ioudf.loc[ioudf.img ==1].apply(lambda row: matchIOUrowwithYdf(row, mdf), axis=1)
	#ioudf.swifter.allow_dask_on_strings(enable=True).apply(lambda row: matchIOUrowwithYdf(row, mdf), axis=1)
	# mdf.to_csv('cacheMerged.csv')
	# mdf = pd.read_csv('cacheMerged.csv', index_col='Unnamed: 0')

	# hdf.apply(lambda row: addFalsePositive(hdf, mdf, row), axis=1)
	#hdf.swifter.allow_dask_on_strings(enable=True).apply(lambda row: addFalsePositive(hdf, mdf, row), axis=1)
	# mdf.to_csv('cacheMerged.csv')
	# mdf = pd.read_csv('cacheMerged.csv',index_col='Unnamed: 0')



	####ex#

	imageNum = 944
	plotIOUOverlapForImageNum(imageNum,df,hdf,ioudf)
	print()

	#create a df with iou and predicted/ ground truth lables in places where there
	#was no detection bb is ground truth in places of flase detection with no ovelap
	#bb is predicted else IOU



