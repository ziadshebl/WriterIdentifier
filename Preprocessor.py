import cv2
import numpy as np
import os


class Preprocessor:
    @staticmethod
    def read_images(filename, images_ids):
        x_train = []
        x_images_names = []
        for imageID in images_ids:
            img = cv2.imread(filename + '/' + imageID)
            x_images_names.append(imageID)
            x_train.append(img)
        x_train = np.asarray(x_train)
        return x_train, x_images_names

    @staticmethod
    def preprocess(img):
        # Reduce image noise.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Binarize the image.
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Return pre processed images.
        return binary

    @staticmethod
    def crop(img):

        # Finding all contours in the image
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Minimum contour width to be considered as the black separator line.
        threshold_width = 1000
        line_offset = 0

        # Page paragraph boundaries.
        height, width = img.shape
        up, down, left, right = 0, height - 1, 0, width - 1

        # Detect the main horizontal black
        # separator lines of the IAM handwriting forms.
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < threshold_width:
                continue
            if y < height // 2:
                up = max(up, y + h + line_offset)
            else:
                down = min(down, y - line_offset)

        # Applying filters to enhance the image
        kernel = np.ones((3, 3), np.uint8)
        eroded_img = cv2.erode(img, kernel, iterations=2)
        # Get horizontal and vertical histograms.
        hor_hist = np.sum(eroded_img, axis=1) / 255
        ver_hist = np.sum(eroded_img, axis=0) / 255

        # Detect paragraph white padding.
        while left < right and ver_hist[left] == 0:
            left += 1
        while right > left and ver_hist[right] == 0:
            right -= 1
        while up < down and hor_hist[up] == 0:
            up += 1
        while down > up and hor_hist[down] == 0:
            down -= 1

        binary = img[up:down + 1, left:right + 1]

        return binary

    @staticmethod
    def save_preprocessed(filename, x_images_names, x_train_binary, writer_id):

        binary_directory = 'PreprocessedImages/'+writer_id+'/binary/'
        if not os.path.exists(filename + binary_directory):
            os.makedirs(filename + binary_directory)
        for i in range(len(x_train_binary)):
            print(filename + 'binary/' + str(x_images_names[i]))
            cv2.imwrite(filename + binary_directory + str(x_images_names[i]), x_train_binary[i])

    @staticmethod
    def preprocessing_pipeline(file_name, images_ids, writer_id, save=False):
        x_train, x_images_names = Preprocessor.read_images(file_name, images_ids)
        x_train_binary = []
        for origImg in x_train:
            preprocessed_image = Preprocessor.preprocess(origImg)
            cropped_image_binary = Preprocessor.crop(preprocessed_image)
            x_train_binary.append(cropped_image_binary)

        if save:
            Preprocessor.save_preprocessed(
                file_name, x_images_names, x_train_binary, writer_id)
        return x_train_binary

    @staticmethod
    def preprocessing_pipeline_image(image):

        preprocessed_image = Preprocessor.preprocess(image)
        cropped_image_binary = Preprocessor.crop(preprocessed_image)
        return cropped_image_binary
