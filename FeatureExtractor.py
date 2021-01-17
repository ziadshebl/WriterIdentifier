import numpy as np
import cv2


class LBPFeatureExtractor:
    @staticmethod
    def lbp_calculated_pixel(window, row_index, col_index):
        values = list()
        values.append(1 if (window[row_index, col_index] <
                            window[row_index - 1, col_index + 1]) else 0)  # Top Right
        values.append(1 if (window[row_index, col_index] <
                            window[row_index, col_index + 1]) else 0)  # Right
        values.append(1 if (window[row_index, col_index] <
                            window[row_index + 1, col_index + 1]) else 0)  # Bottom Right
        values.append(1 if (window[row_index, col_index] <
                            window[row_index + 1, col_index]) else 0)  # Bottom

        values.append(1 if (window[row_index, col_index] <
                            window[row_index + 1, col_index - 1]) else 0)  # Bottom Left
        values.append(1 if (window[row_index, col_index] <
                            window[row_index, col_index - 1]) else 0)  # Left
        values.append(1 if (window[row_index, col_index] <
                            window[row_index - 1, col_index - 1]) else 0)  # Top Left
        values.append(1 if (window[row_index, col_index] <
                            window[row_index - 1, col_index]) else 0)  # Top

        power_values = [128, 64, 32, 16, 8, 4, 2, 1]
        final_value = 0
        for i in range(len(values)):
            final_value += values[i] * power_values[i]
        return final_value

    @staticmethod
    def lbp_calculated_window(window):
        window = np.pad(window, (1, 1), mode='constant')
        lbp_window = np.zeros((5, 5))
        lbp_histogram = np.zeros(256)
        for row_index in range(1, 6):
            for col_index in range(1, 6):
                lbp_value = LBPFeatureExtractor.lbp_calculated_pixel(window, row_index, col_index)
                lbp_window[row_index - 1, col_index - 1] = lbp_value
                lbp_histogram[lbp_value] = lbp_histogram[lbp_value] + 1

        return lbp_histogram

    @staticmethod
    def compute_lbp_hist(line_input):
        line = np.array(line_input)
        scale_percent = 25
        width = int(line.shape[1] * scale_percent / 100)
        height = int(line.shape[0] * scale_percent / 100)
        dim = (width, height)
        line_resize = cv2.resize(line, dim, interpolation=cv2.INTER_AREA)
        lbp_hist = np.zeros(256)
        h, w = line_resize.shape
        line_resize[line_resize == 255] = 1
        for i in range(2, h - 2, 5):
            for j in range(2, w - 2, 5):
                window = line_resize[i - 2:i + 3, j - 2:j + 3]
                if not np.all((window == 0)):
                    lbp_histogram = LBPFeatureExtractor.lbp_calculated_window(window)
                    lbp_hist = np.add(lbp_hist, lbp_histogram)
        return lbp_hist
