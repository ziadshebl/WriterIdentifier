from Preprocessor import Preprocessor
from Utilities import Utilities
from LineSegmentor import LineSegmentor
from Classifiers import Classifiers
from FeatureExtractor import LBPFeatureExtractor
from tqdm import tqdm
import timeit
import numpy as np
import os
# Constants

# Path to generate the dataset at
test_dataset_directory = "data\\"
final_directory = "data\\"
predictions_filename = "results.txt"
timers_filename = "timers.txt"
folder_count = 0
for folders in os.listdir(test_dataset_directory):
    if".txt" in folders:
        None
    else:
        folder_count += 1  # increment counter
test_cases_range = range(0, folder_count)

# Variables
counter = 0
predictions = []
timers = []
for i in tqdm(test_cases_range):

    # Starting Timer
    start = timeit.default_timer()

    # Test Case Folder Name
    folder_name = '%02d' % i

    # ------------------------------TRAINING----------------------------------------- #

    # Reading the inputs and labelling the training dataset
    x_train, y_train, x_test = Utilities.read_test_case(test_dataset_directory + folder_name)
    x_train_binary = []

    # Preprocessing every image in the input dataset
    for img in x_train:
        binary = Preprocessor.preprocessing_pipeline_image(img)
        x_train_binary.append(binary)

    # Segmenting the dataset into lines
    # Then calculate the LBP Histogram for each segment
    x_train_segments = np.empty(256)
    y_train_segments = []

    for w in range(len(x_train_binary)):

        binary_lines = LineSegmentor.segmentation_pipeline(x_train_binary[w])
        for j in range(len(binary_lines)):

            lbp_hist = LBPFeatureExtractor.compute_lbp_hist(binary_lines[j])
            x_train_segments = np.vstack((x_train_segments, lbp_hist))

            # labelling the segmented dataset
            if w < 2:
                y_train_segments.append(1)
            elif w < 4:
                y_train_segments.append(2)
            else:
                y_train_segments.append(3)

    y_train_segments = np.asarray(y_train_segments)

    # --------------------------------TEST---------------------------------#

    # Preprocessing the image in the testing dataset
    x_test_binary = Preprocessor.preprocessing_pipeline_image(x_test)

    # Segmenting the dataset into lines
    # And Calculating the features vector
    binary_lines = LineSegmentor.segmentation_pipeline(x_test_binary)
    x_test_segments = np.empty([len(binary_lines), 256])
    for w in range(len(binary_lines)):
        lbp_hist = LBPFeatureExtractor.compute_lbp_hist(binary_lines[w])
        x_test_segments[w] = lbp_hist

    # Classifying the test case
    # SVM Classifier
    svm_results = Classifiers.svm_classifier(x_train_segments[1:len(
        x_train_segments)], y_train_segments, x_test_segments)

    # KNN Classifier
    knn_results = Classifiers.knn_classifier(x_train_segments[1:len(
        x_train_segments)], y_train_segments, x_test_segments)

    # Random Forest Classifier
    random_forest_results = Classifiers.rand_forest_classifier(x_train_segments[1:len(
        x_train_segments)], y_train_segments, x_test_segments, max_depth=8, random_state=0)

    # Stacking all the results together
    # Then sum all of them vertically
    results = svm_results
    results = np.vstack((results, knn_results))
    results = np.vstack((results, random_forest_results))
    results = np.sum(results, axis=0)

    # Checking how much is the classifying system sure
    # If less than 40%, Neural Network is used to enhance the results
    if results[np.argmax(results)]/(sum(results)*100) < 40:
        NNResults = Classifiers.nn_classifier(x_train_segments[1:len(
            x_train_segments)], y_train_segments, x_test_segments,
                                              hidden_layer_sizes=[100, 50], max_iter=400)
        results = np.vstack((results, NNResults))
        results = np.sum(results, axis=0)

    # Calculating the predicted writer
    # and append it to the predictions array
    prediction = np.argmax(results) + 1
    predictions.append(prediction)

    # Closing Timer
    final = timeit.default_timer()
    timers.append(final-start)


Utilities.write_answers(final_directory, predictions_filename, predictions)
Utilities.write_answers(final_directory, timers_filename, timers)
# Utilities.generate_answers(final_directory, test_cases_range)
# print(Utilities.calculate_accuracy(final_directory+"results.txt", final_directory+"true.txt"))
print(Utilities.calculate_average_time(final_directory+"timers.txt"))
