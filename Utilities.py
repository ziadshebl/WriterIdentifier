import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from statistics import mode


class Utilities:
    @staticmethod
    def show_images(images, titles=None):
        # This function is used to show image(s) with titles by sending an array of images.
        # images[0] will be drawn with the title titles[0] if exists
        # You aren't required to understand this function, use it as-is.
        n_ims = len(images)
        if titles is None:
            titles = ['(%d)' % i for i in range(1, n_ims + 1)]
        fig = plt.figure(figsize=(15, 15))
        n = 1
        for image, title in zip(images, titles):
            a = fig.add_subplot(1, n_ims, n)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
            n += 1
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
        plt.show()

    @staticmethod
    def read_test_case(test_case_directory):
        x_train = []
        y_train = []

        images_glob = glob.glob(test_case_directory + "\\1\\*.png")
        x_train.append(cv2.imread(images_glob[0]))
        x_train.append(cv2.imread(images_glob[1]))
        y_train.append(1)
        y_train.append(1)

        images_glob = glob.glob(test_case_directory + "\\2\\*.png")
        x_train.append(cv2.imread(images_glob[0]))
        x_train.append(cv2.imread(images_glob[1]))
        y_train.append(2)
        y_train.append(2)

        images_glob = glob.glob(test_case_directory + "\\3\\*.png")
        x_train.append(cv2.imread(images_glob[0]))
        x_train.append(cv2.imread(images_glob[1]))
        y_train.append(3)
        y_train.append(3)

        x_test = cv2.imread(glob.glob(test_case_directory + "\\*.png")[0])

        return x_train, y_train, x_test

    @staticmethod
    def is_correct(test_dataset_directory, z, results):

        reader = open(test_dataset_directory + z + "\\ids.txt")

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

    @staticmethod
    def write_answers(final_directory, filename, results):

        file = open(final_directory + '\\'+filename, 'w+')
        for result in results:
            file.write(str(result)+'\n')
        file.close()

    @staticmethod
    def generate_answers(final_directory, range):
        answers = []
        for i in range:
            folder_name = '%02d' % i
            reader = open(final_directory + folder_name + "\\ids.txt")
            inputs = list(reader)

            true = -1
            if int(inputs[-1].lstrip()) == int(inputs[0].lstrip()):
                true = 1
            elif int(inputs[-1].lstrip()) == int(inputs[1].lstrip()):
                true = 2
            elif int(inputs[-1].lstrip()) == int(inputs[2].lstrip()):
                true = 3

            answers.append(true)
        Utilities.write_answers(final_directory, 'true.txt', answers)

    @staticmethod
    def calculate_accuracy(true_file, results_file):

        true_reader = open(true_file)
        results_reader = open(results_file)

        true_inputs = list(true_reader)
        results_inputs = list(results_reader)
        counter = 0
        for result, true in zip(results_inputs, true_inputs):
            if true == result:
                counter += 1

        return counter

    @staticmethod
    def calculate_average_time(file):

        reader = open(file)

        inputs = list(reader)
        counter = 0
        for timer in inputs:
            counter += float(timer.lstrip('\\'))

        return counter/len(inputs)
