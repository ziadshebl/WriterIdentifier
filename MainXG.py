from Preprocessor import Preprocessor
from Old import LBP_Feature_Extractor
from Utilities import Utilities
from tqdm import tqdm
from LineSegmentor import LineSegmentor
from Classifiers import Classifiers
from Old2 import *


# Constants

# Path to generate the dataset at
testDatasetDirectory = "Data\\data2\\"

prep = Preprocessor()
extractor = LBP_Feature_Extractor()

counter = 0
predictions = []
for i in tqdm(range(100, 300)):

    # #################################TRAINING############################################################
    # #####Reading the inputs and labelling the training dataset######

    z = '%02d' % i

    x_train, y_train, x_test = Utilities.read_test_case(testDatasetDirectory + z)
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
            lbpHist = compute_lbp_hist(binary_lines[j])
            x_train_segments = np.vstack((x_train_segments, lbpHist))

            # #Labelling the segmented dataset##
            if i < 2:
                y_train_segments.append(1)
            elif i < 4:
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
        lbpHist = compute_lbp_hist(binary_lines[i])
        x_test_segments[i] = lbpHist

    # ------Classifying the test case----- #
    SVMResults = Classifiers.svm_classifier(x_train_segments[1:len(x_train_segments)], y_train_segments, x_test_segments)
    KNNResults = Classifiers.knn_classifier(x_train_segments[1:len(x_train_segments)], y_train_segments, x_test_segments)

    results = SVMResults
    results = np.vstack((results, KNNResults))
    results = np.sum(results, axis=0)

    if results[np.argmax(results)] / (sum(results) * 100) < 40:
        # NNResults = Classifiers.NNClassifier(x_train_segments[1:len(x_train_segments)], y_train_segments, x_test_segments,
        #                          random_state=random.seed(120), hidden_layer_sizes=[75, 35], max_iter=500, )
        # results = np.vstack((results, NNResults))

        XGBoostResults = Classifiers.gradient_classifier(x_train_segments[1:len(x_train_segments)], y_train_segments, x_test_segments,
                                                         max_depth=8, random_state=random.seed(120), learning_rate=1, n_estimators=200)
        results = np.vstack((results, XGBoostResults))
        results = np.sum(results, axis=0)

    prediction = np.argmax(results) + 1
    predictions.append(prediction)

# ----------- Calculating Accuracy -------- #
for prediction,i in tqdm(zip(predictions, range(100, 300))):
    z = '%02d' % i
    x = np.zeros(1)
    x[0] = prediction
    if Utilities.is_correct(testDatasetDirectory, z, x):
        counter += 1
    else:
        print(z)


print(counter)
