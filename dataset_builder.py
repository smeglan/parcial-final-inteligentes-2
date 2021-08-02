import cv2 as cv
import numpy as np
import os
from pathlib import Path

def rescaleImage(image, scale=0.2):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)


def createCrop(image, approx):
    points = np.squeeze(approx)
    x = points[:, 0]
    y = points[:, 1]
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = image[topy:bottomy+1, topx:bottomx+1]
    return out


def shapeDetector(image):
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            return approx
    return None


def prepareToSaveCrop(image, width=128, height=128):
    dim = (width, height)
    image = rescaleImage(image)
    approx = shapeDetector(image)
    if approx is not None:
        image = createCrop(image, approx)
    # Change color to gray scale.
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)

def prepareToSaveRaw(image, width=128, height=128):
    dim = (width, height)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)

def createDataSet(directory, datasetFolder="dataset"):
    Path(datasetFolder).mkdir(parents=True, exist_ok=True)
    for folder in os.walk(directory):
        if(folder[0] != directory):
            folderName = folder[0].split("\\")[1]
            classPath = datasetFolder+"/"+folderName
            Path(classPath).mkdir(parents=True, exist_ok=True)
            for source in folder[2]:
                img = prepareToSaveCrop(cv.imread(directory+"/"+folderName+"/"+source))
                cv.imwrite(classPath+"/"+source, img)  
                img = prepareToSaveRaw(cv.imread(directory+"/"+folderName+"/"+source))
                cv.imwrite(classPath+"/raw"+source, img)         

datasetRoot = "dataset"
if not os.path.exists(datasetRoot):
    os.makedirs(datasetRoot)

createDataSet("raw_set", datasetRoot+"/train")

