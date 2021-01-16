import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from statistics import mode

class Utilitites:
    @staticmethod
    def show_images(images, titles=None):
        #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
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
    def readTestCase(testCaseDirectory):
        x_train = []
        y_train = []

        imagesGlob = glob.glob(testCaseDirectory+"\\1\\*.png")
        x_train.append(cv2.imread(imagesGlob[0]))
        x_train.append(cv2.imread(imagesGlob[1]))
        y_train.append(1)
        y_train.append(1)

        imagesGlob = glob.glob(testCaseDirectory+"\\2\\*.png")
        x_train.append(cv2.imread(imagesGlob[0]))
        x_train.append(cv2.imread(imagesGlob[1]))
        y_train.append(2)
        y_train.append(2)

        imagesGlob = glob.glob(testCaseDirectory+"\\3\\*.png")
        x_train.append(cv2.imread(imagesGlob[0]))
        x_train.append(cv2.imread(imagesGlob[1]))
        y_train.append(3)
        y_train.append(3)

        x_test = cv2.imread(glob.glob(testCaseDirectory+"\*.png")[0])

        return x_train, y_train, x_test

    @staticmethod
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
