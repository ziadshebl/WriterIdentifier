import cv2
import numpy as np
import math
import glob
import skimage
import os
from shutil import copyfile
import random
from tqdm import tqdm

#Number of writers to generate the test dataset from
numberOfWriters = 672

#Number of test cases needed
numberOfTestCases = 600

#Path for Writers data
writersPath = "Data\Writers\\"

#Path to generate the dataset at
testDatasetDirectory = "Data\data\\"

#Path to all of the images in the dataset
allDatasetDirectory = "Data\AllDataSet\\"

#Path to the writers form.txt
writersFormDirectory = "Data\\asciiData\\forms.txt"


def get_writers_images_names(file_path):
    try:
        file = open(file_path, "r")
        file_lines = file.readlines()
        for i in range(len(file_lines)):
            file_lines[i] = file_lines[i].rstrip("\n")
        file.close()
        return file_lines
    except IOError:
        return []


#This cell is responsible for grouping the ids of a specific writer and discarding writers who has less than 3 images
writerImagesID = []
for i in range(numberOfWriters):
    writerID = '%03d' % i
    writerImagesIDItem = get_writers_images_names(writersPath+writerID+".txt")
    if len(writerImagesIDItem) >= 3:
        writerImagesID.append(writerImagesIDItem)

#This function is responsible for getting the writed id of a specific image name


def getWriterId(path, imageId):
    imageId = imageId.split(".")[0]
    file = open(path, "r")
    file_lines = file.readlines()
    line = file_lines[18]
    #imageId = "a01-007"
    imageId = imageId+" "
    for line in file_lines[16:]:
        if imageId in line:
            if(line[7] == " "):
                return line[8:11]
            else:
                return line[9:12]
    file.close()


#Generating the folders if not available and create them
for i in range(numberOfTestCases):
    z = '%02d' % i
    directory = testDatasetDirectory + z+'\\'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory+'1\\'):
        os.makedirs(directory+'1\\')
    if not os.path.exists(directory+'2\\'):
        os.makedirs(directory+'2\\')
    if not os.path.exists(directory+'3\\'):
        os.makedirs(directory+'3\\')

for i in tqdm(range(numberOfTestCases)):
    testCaseNumber = '%02d' % i
    directory = testDatasetDirectory+testCaseNumber+'\\'
    randomWriterIndex1 = random.randrange(0, len(writerImagesID), 1)

    writersID = []

    #Generating images of writer 1 in the test case
    image1 = ''
    image2 = ''
    while image1 == image2:
        image1 = writerImagesID[randomWriterIndex1][random.randrange(
            0, len(writerImagesID[randomWriterIndex1]), 1)]
        image2 = writerImagesID[randomWriterIndex1][random.randrange(
            0, len(writerImagesID[randomWriterIndex1]), 1)]
    dest1 = testDatasetDirectory+testCaseNumber+'\\1\\'+image1
    dest2 = testDatasetDirectory+testCaseNumber+'\\1\\'+image2
    copyfile('Data\AllDataSet\\'+image1, dest1)
    copyfile('Data\AllDataSet\\'+image2, dest2)
    writersID.append(getWriterId(writersFormDirectory, image1))

    #Generating images of writer 2 in the test case
    randomWriterIndex2 = random.randrange(0, len(writerImagesID), 1)
    while randomWriterIndex2 == randomWriterIndex1:
        randomWriterIndex2 = random.randrange(0, len(writerImagesID), 1)

    image3 = ''
    image4 = ''
    while image3 == image4:
        image3 = writerImagesID[randomWriterIndex2][random.randrange(
            0, len(writerImagesID[randomWriterIndex2]), 1)]
        image4 = writerImagesID[randomWriterIndex2][random.randrange(
            0, len(writerImagesID[randomWriterIndex2]), 1)]
    dest1 = testDatasetDirectory+testCaseNumber+'\\2\\'+image3
    dest2 = testDatasetDirectory+testCaseNumber+'\\2\\'+image4
    copyfile(allDatasetDirectory+image3, dest1)
    copyfile(allDatasetDirectory+image4, dest2)
    writersID.append(getWriterId(writersFormDirectory, image3))

    #Generating images of writer 3 in the test case
    randomWriterIndex3 = random.randrange(0, len(writerImagesID), 1)
    while randomWriterIndex3 == randomWriterIndex1 or randomWriterIndex3 == randomWriterIndex2:
        randomWriterIndex3 = random.randrange(0, len(writerImagesID), 1)

    image5 = ''
    image6 = ''
    while image5 == image6:
        image5 = writerImagesID[randomWriterIndex3][random.randrange(
            0, len(writerImagesID[randomWriterIndex3]), 1)]
        image6 = writerImagesID[randomWriterIndex3][random.randrange(
            0, len(writerImagesID[randomWriterIndex3]), 1)]
    dest1 = testDatasetDirectory+testCaseNumber+'\\3\\'+image5
    dest2 = testDatasetDirectory+testCaseNumber+'\\3\\'+image6
    copyfile(allDatasetDirectory+image5, dest1)
    copyfile(allDatasetDirectory+image6, dest2)
    writersID.append(getWriterId(writersFormDirectory, image5))

    #Generating the test image randomly from the dataset of the 3 writers
    #Making sure the test image is different than those from the given training images
    randomWriterIndexTest = random.choice(
        [randomWriterIndex1, randomWriterIndex2, randomWriterIndex3])
    imageTest = writerImagesID[randomWriterIndexTest][random.randrange(
        0, len(writerImagesID[randomWriterIndexTest]), 1)]
    while imageTest == image1 or imageTest == image2 or imageTest == image3 or imageTest == image4 or imageTest == image5 or imageTest == image6:
        imageTest = writerImagesID[randomWriterIndexTest][random.randrange(
            0, len(writerImagesID[randomWriterIndexTest]), 1)]
    copyfile(allDatasetDirectory+imageTest,
             testDatasetDirectory+testCaseNumber+'\\'+imageTest)
    writersID.append(getWriterId(writersFormDirectory, imageTest))

    f = open(testDatasetDirectory+testCaseNumber+"\ids.txt", "a")
    for i in range(len(writersID)):
        f.write(writersID[i]+"\n")
    f.close()
