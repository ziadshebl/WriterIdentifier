import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from sklearn import svm
from sklearn import tree
from statistics import mode
from skimage import feature
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier



%run Utilities.ipynb
%run Preprocessor.ipynb
%run LineSegmentor.ipynb
%run FeatureExtractor.ipynb
%run FeatureExtractor2.ipynb
%run Classifiers.ipynb


#Constants

#Path to generate the dataset at
testDatasetDirectory = "Data\data\\"
prep = Preprocessor()



def isCorrect(testDatasetDirectory, z, results):
   
       reader = open(testDatasetDirectory+z+'\ids.txt')
   
       inputs = list(reader)
   
       true = -1
       if int(inputs[-1].lstrip()) == int(inputs[0].lstrip()):
           true = 1
       elif int(inputs[-1].lstrip()) == int(inputs[1].lstrip()):
           true = 2
       elif int(inputs[-1].lstrip()) == int(inputs[2].lstrip()):
           true = 3
   
       if true == mode(results):
           return True
       return False
   
def KNNClassifier(x_train, y_train, x_test, k=3):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results


def SVMClassifier(x_train, y_train, x_test):
    clf = svm.SVC(probability=True)  # , gamma='auto', C=5.0)
    clf.fit(x_train, y_train)
    results = clf.predict_proba(x_test)
    return results


def RandForestClassifier(x_train, y_train, x_test, max_depth=2, random_state=0):
    clf = RandomForestClassifier(
        max_depth=max_depth, random_state=random_state)
    clf.fit(x_train, y_train)
    results = clf.predict_proba(x_test)
    return results


def KMeansClassifier(x_train, y_train, x_test, n_clusters=3, random_state=0):
    clf = KMeans(n_clusters=n_clusters, random_state=random_state)
    clf.fit(x_train)
    results = clf.predict(x_test)
    results + 1
    return results


def AdaboostClassifier(x_train, y_train, x_test,n_estimators=100, random_state=0):
    clf = AdaBoostClassifier(n_estimators=n_estimators,
                            random_state=random_state)
    clf.fit(x_train, y_train)
    results = clf.predict_proba(x_test)
    return results




def NNClassifier(x_train, y_train, x_test, max_iter=300, random_state=0, hidden_layer_sizes=[100, 50]):
    clf = MLPClassifier(random_state=random_state,
                        max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)
    clf.fit(x_train, y_train)
    results = clf.predict_proba(x_test)
    return results   
   

def buildImage(width,height,avgHeight,images):
    
    Image=np.ones((height,width))
    #concatenate horizonatl
    x=0 #must be updated after attaching every word in a line
    y=0 #must be updated after finishing a line
    #i=0
    maxy=-1;
    maxX=-1;
    for img in images:    
        if(x+img.shape[1]<=width):
            Image[y:y+img.shape[0],x:x+img.shape[1]]*=img #np.logical_or(Image[y:y+img.shape[0],x:x+img.shape[1]],img)
            x+=img.shape[1]
            maxX=maxX if maxX>=x else x
            maxy=maxy if maxy>=y+img.shape[0] else y+img.shape[0]
        else: #start a new line
            y=y+int(1.25*avgHeight)
            x=0
            Image[y:y+img.shape[0],x:x+img.shape[1]]*=img
            x+=img.shape[1]
            maxy=y+img.shape[0] if maxy>=y+img.shape[0] else y+img.shape[0]
      
    
    return Image[0:maxy,0:maxX]

def compressedImageFeature(img): #takes a gray image of segmented paragraphs, desired width of image 

        #img = cv2.imread('./Data/AllDatasetPreprocessedImages/006/gray/a01-011x.png',cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread('./Data/AllDatasetPreprocessedImages/001/binary/a01-000x.png')
        #print(img.shape)
        widthF=img.shape[1]
        heightF=img.shape[0]

        connectivity =16 
        otsu_thr, otsu_mask = cv2.threshold(img,0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
        output = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)
        num_labels, labelmap, stats, centers = output

        x=stats[:,0]
        y=stats[:,1]
        width=stats[:,2]
        height=stats[:,3]

        avgh=np.average(height[height!=img.shape[0]])
        #print("average height of components ", avgh)
        area=stats[:,4]
        segmented_images=[]
        for i in range(stats.shape[0]):
            if  width[i]>5: #area[i]>200  #trying to discard rectanges of area smaller than 200 and noise 
                if height[i]!=img.shape[0] : ## we do not want a component with a height equls to image height because it will probably be noise not a letter
                    image1=img[y[i]:(y[i]+height[i]),x[i]:(x[i]+width[i])]
                    segmented_images.append(image1)
                    #the next line shows boundes rectangles on the image
                    #img = cv2.rectangle(img,(x[i],y[i]),(x[i]+width[i],y[i]+height[i]),(0,0,0), 2)#must be removed when concatenation is made 

        
        

        #compressed= buildImage(widthF,heightF,avgh,segmented_images)
        compressed= buildImage(widthF,heightF,avgh,segmented_images)
        #otsu_thr, otsu_mask = cv2.threshold(compressed,127, 255, cv2.THRESH_BINARY)
    
        #arr=[]
        #arr.append(img)
        #show_images(arr)
        #cv2.imshow("cropped",img )
        #cv2.waitKey(0)         
    
    
        return compressed 

def get_LBP_Image(image,numPoints=8,radius=3):
        lbp = feature.local_binary_pattern(image, numPoints,radius, method="default")     
        (hist, _) = np.histogram(lbp, bins=256, range=(0,255))
        hist = hist.astype("float")
        hist /= (hist.sum())
        # return the histogram of Local Binary Patterns
        return hist[0:256].tolist()
    

SVMcounter = 0
KNNcounter = 0
RandomForest8counter = 0
NNcounter = 0
counter = 0
SlantCounter = 0
for i in tqdm(range(0, 10)):
    ##################################TRAINING############################################################
    ######Reading the inputs and labelling the training dataset######
    z = '%02d' % i
    #z = '01'
    x_train, y_train, x_test = readTestCase(testDatasetDirectory+z)
    x_train_gray = []
    x_train_binary = []
    x_train_original = []

    
    #####Preprocessing every image in the input dataset######
    for img in x_train:    
        gray, binary, original = prep.preprocessing_pipeline_image(img)
        x_train_gray.append(gray)
        x_train_binary.append(binary)
        x_train_original.append(original)
    
    
    
   
        
    
    #####Segmenting the dataset into lines#####
    x_train_segments =  np.empty(256)
    y_train_segments = []


    #looping on gray extracted paragraphs
    for i in range(len(x_train_gray)):  
      
        compressed=compressedImageFeature(x_train_gray[i])
        lbp=get_LBP_Image(compressed)
        x_train_segments = np.vstack((x_train_segments, lbp))

        if (i<2):
             y_train_segments.append(1)
        elif (i<4):
             y_train_segments.append(2)
        else:
             y_train_segments.append(3) 
    
    
    y_train_segments = np.asarray(y_train_segments)
    
    
    print(x_train_segments.shape)
    print(y_train_segments)
  
    

    

#   ######################################TEST###################################################### 
    
    x_test_gray, x_test_binary, x_test_original = prep.preprocessing_pipeline_image(x_test)
    compressed_test=compressedImageFeature(x_test_gray)
    lbp_test=get_LBP_Image(compressed_test)
    x_test_segments= lbp_test
    
    x_test_segments=np.asarray(x_test_segments).reshape(1,-1)

    
    
    
    SVMResults =SVMClassifier(x_train_segments[1:len(x_train_segments)], y_train_segments, x_test_segments)
    KNNResults =KNNClassifier(x_train_segments[1:len(x_train_segments)], y_train_segments, x_test_segments)
    RandomForest8Results =RandForestClassifier(x_train_segments[1:len( x_train_segments)], y_train_segments,x_test_segments, max_depth=8, random_state=0)
    #results = RandomForest8Results
    results = SVMResults
    #results = KNNResults
    results = np.vstack((results, KNNResults))
    results = np.vstack((results, RandomForest8Results))
    results = np.sum(results, axis=0)

    if results[np.argmax(results)]/(sum(results)*100) < 40:
        NNResults = NNClassifier(x_train_segments[1:len(
        x_train_segments)], y_train_segments, x_test_segments, 
        hidden_layer_sizes=[100, 50], max_iter=2000)
        results = np.vstack((results, NNResults))

      
    results = np.sum(results, axis=0)

    prediction = np.zeros(1)
    prediction[0] = np.argmax(results) + 1

    if isCorrect(testDatasetDirectory, z, prediction):
        counter += 1
    if(i % 100 == 0):
        print(counter)    
            