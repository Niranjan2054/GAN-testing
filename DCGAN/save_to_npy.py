#!/use/bin/env python3 
from PIL import Image 
import numpy as np 
import os 

def grabListOfFiles(startingDirectory,extension=".jpg"):
    listOfFiles = []
    for file in os.listdir(startingDirectory):
        if file.endswith(extension):
            listOfFiles.append(os.path.join(startingDirectory,file))
    return listOfFiles 

def grabArrayOfImages(listOfFiles,resizeW=64,resizeH=64,gray=False):
    imageArr = [] 
    for f in listOfFiles:
        if gray: 
            im = Image.open(f).convert("L")
        else: 
            im = Image.open(f).convert("RGB")
        im = im.resize((resizeW,resizeH))
        imData = np.asarray(im)
        imageArr.append(imData)
    return imageArr 

direc = "/content/GAN-testing/DCGAN/datasets/facades/train/"
listOfFiles = grabListOfFiles(direc)
imageArrGray = grabArrayOfImages(listOfFiles,resizeW=64,resizeH=64,gray=True)
imageArrRGB  = grabArrayOfImages(listOfFiles,resizeW=64,resizeH=64,gray=False)
print("Shape of ImageArr Gray: ",np.shape(imageArrGray))
print("Shape of ImageArr Color: ",np.shape(imageArrRGB))

np.save('data/facadeGray.npy', imageArrGray)
np.save('data/facadeRGB.npy', imageArrRGB)