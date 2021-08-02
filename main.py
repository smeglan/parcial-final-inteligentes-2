import sys
from tensorflow import keras
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
# Recrea exactamente el mismo modelo solo desde el archivo


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
    out = cv.resize(out, (180, 180))
    return out


def shapeDetector(image):
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    listAprox = []
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            listAprox.append(approx)
            print("append", listAprox)
        if len(listAprox) == 2:
            return listAprox
    return None


def splitImage(image):
    bounds = shapeDetector(image)
    images = []
    print("bounds", bounds)
    if bounds is not None:
        images.append(createCrop(image, bounds[0]))
        images.append(createCrop(image, bounds[1]))
    return images


def addClasses(class1, class2):
    return class1+class2


run = True
model = keras.models.load_model('best_model.h5', compile=True)
while run:
    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename()
    test_image = cv.imread(filename)
    test_image = cv.resize(test_image, (180, 180))
    images = splitImage(test_image)
    cv.imshow("Imagen 1", images[0])
    cv.imshow("Imagen 2", images[1])
    cv.waitKey(0)
    #images[0] = np.reshape(images[0], (180, 180, 3))
    #images[1] = np.reshape(images[0], (180, 180, 3))
    #images[0] = np.expand_dims(images[0], axis=0)
    #images[1] = np.expand_dims(images[1], axis=0)
    images[0] = images[0].flatten()
    images[0] = images[0]/255
    images[1] = images[1].flatten()
    images[1] = images[1]/255
    images = np.array(images)
    cv.destroyAllWindows()
    # predict the result
    class1 = model.predict(x=images)
    print(class1)
    #print("Resultado: "+class1 + " + " + class2 +
    #      " = " + addClasses(class1+class2))
    entry = ""
    while(entry.lower() != "y" and entry.lower() != "yes" and entry.lower() != "n" and entry.lower() != "no"):
        entry = input("¿Desea ingresar otra imagen? [Y/N]")
        print(entry)
        if entry.lower() == "y" and entry.lower() == "yes":
            run = True
        else:
            run = False
