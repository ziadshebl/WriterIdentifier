import cv2
import numpy as np
from Preprocessor import Preprocessor
from FeatureExtractor import LBP_Feature_Extractor
from Utilities import Utilitites
from tqdm import tqdm
from LineSegmentor import LineSegmentor
from sklearn import svm
from sklearn import tree
from statistics import mode
from Classifiers import Classifiers
from FeatureExtractor2 import *
# Constants

# Path to generate the dataset at
testDatasetDirectory = "Data\\data\\"

prep = Preprocessor()
extractor = LBP_Feature_Extractor()

# fig = plt.figure(figsize=(15,15))
# plt.imshow(x_test)
# plt.show

SVMcounter = 0
KNNcounter = 0
RandomForest8counter = 0
NNcounter = 0
counter = 0
SlantCounter = 0
for i in tqdm(range(0, 600)):

    # #################################TRAINING############################################################
    # #####Reading the inputs and labelling the training dataset######

    z = '%02d' % i
    # z = '32'

#     if not os.path.exists(testDatasetDirectory+z):
#         continue
    x_train, y_train, x_test = Utilitites.readTestCase(testDatasetDirectory+z)
    x_train_gray = []
    x_train_binary = []
    x_train_original = []

    # ####Preprocessing every image in the input dataset######
    for img in x_train:
        gray, binary, original = prep.preprocessing_pipeline_image(img)
        x_train_gray.append(gray)
        x_train_binary.append(binary)
        x_train_original.append(original)

    # ####Segmenting the dataset into lines#####
    x_train_segments = np.empty(256)
    y_train_segments = []

    for i in range(len(x_train_gray)):
        gray_lines, binary_lines, orig_lines = LineSegmentor(
            x_train_gray[i], x_train_binary[i], x_train_original[i]).segment()

        for j in range(len(binary_lines)):
            #             if j>4:
            #                 break
            lbpHist = computeLBPHist(binary_lines[j])
            x_train_segments = np.vstack((x_train_segments, lbpHist))

            # #Labelling the segmented dataset##
            if (i < 2):
                y_train_segments.append(1)
            elif (i < 4):
                y_train_segments.append(2)
            else:
                y_train_segments.append(3)

    y_train_segments = np.asarray(y_train_segments)

    # #####################################TEST######################################################

    # ####Preprocessing every image in the testing dataset######
    x_test_gray, x_test_binary, x_test_original = prep.preprocessing_pipeline_image(x_test)

    # ####Segmenting the dataset into lines#####
    # ###And Calculating the features vector####
    gray_lines, binary_lines, orig_lines = LineSegmentor(x_test_gray, x_test_binary, x_test_original).segment()
    x_test_segments = np.empty([len(binary_lines), 256])

    for i in range(len(binary_lines)):
        lbpHist = computeLBPHist(binary_lines[i])
        x_test_segments[i] = lbpHist

    # ###Classifying the test case#####
    SVMResults = Classifiers.SVMClassifier(x_train_segments[1:len(x_train_segments)],
        y_train_segments, x_test_segments)
    KNNResults = Classifiers.KNNClassifier(x_train_segments[1:len(
    x_train_segments)], y_train_segments, x_test_segments)
    RandomForest8Results = Classifiers.RandForestClassifier(x_train_segments[1:len(
    x_train_segments)], y_train_segments,
     x_test_segments, max_depth=8, random_state=0)

    results = SVMResults
    results = np.vstack((results, KNNResults))
    results = np.vstack((results, RandomForest8Results))
    results = np.sum(results, axis=0)

    if results[np.argmax(results)]/(sum(results)*100) < 40:
        NNResults = Classifiers.NNClassifier(x_train_segments[1:len(
        x_train_segments)], y_train_segments, x_test_segments, 
        hidden_layer_sizes=[100, 50], max_iter=400)
        results = np.vstack((results, NNResults))

        ##############################################
        # print(z)
#             x_train_slant =  np.empty(9)
#             y_train_slant = []
#             for i in range(len(x_train_gray)):
#                 gray_lines, binary_lines, orig_lines = LineSegmentor(x_train_gray[i], x_train_binary[i], x_train_original[i]).segment()

#                 for j in range(len(binary_lines)):
#                     if j>2:
#                         break
#                     slantHist = computeSlantHistogram(binary_lines[j])
#                     x_train_slant = np.vstack((x_train_slant, slantHist))

#                     ##Labelling the segmented dataset##
#                     if (i<2):
#                         y_train_slant.append(1)
#                     elif (i<4):
#                         y_train_slant.append(2)
#                     else:
#                         y_train_slant.append(3)

#             ###TEST####
#             gray_lines, binary_lines, orig_lines = LineSegmentor(x_test_gray, x_test_binary, x_test_original).segment()
#             x_test_slant = np.empty([len(binary_lines), 9])
#             for i in range(len(binary_lines)):
#                 slantHist = computeSlantHistogram(binary_lines[i])
#                 x_test_slant[i] = slantHist


#             RandSlantResults = RandForestClassifier(x_train_slant[1:len(x_train_segments)],
#  y_train_slant, x_test_slant, max_depth=8, random_state=0)
#             RandSlantResults = np.sum(RandSlantResults, axis = 0)
#             #print(KNNSlantResults)
#             predictionSlant = np.zeros(1)
#             predictionSlant[0] = np.argmax(RandSlantResults) + 1
#             if isCorrect(testDatasetDirectory, z, predictionSlant):
#                 SlantCounter += 1

#             results = np.vstack((results,RandSlantResults))
#             ##############################################

    results = np.sum(results, axis=0)

    prediction = np.zeros(1)
    prediction[0] = np.argmax(results) + 1

    if Utilitites.isCorrect(testDatasetDirectory, z, prediction):
        counter += 1
    if(i % 100 == 0):
        print(counter)
#     else:
#         print(results)
#         print(z)

# print(SVMcounter)
# print(KNNcounter)
# print(KNN5counter)
# print(KMeanscounter)
# print(RandomForest2counter)
# print(RandomForest4counter)
# print(RandomForest6counter)
# print(RandomForest8counter)
# print(RandomForest10counter)
# print(SlantCounter)
print(counter)
# print(NNcounter)
