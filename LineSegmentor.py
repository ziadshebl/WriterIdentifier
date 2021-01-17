import numpy as np


class LineSegmentor:

    @staticmethod
    def segment(lines_boundaries, binary_image):

        # Initialize lines lists.
        binary_lines = []

        # Loop on every line boundary.
        for left, up, right, down in lines_boundaries:

            # Crop binary line.
            b_line = binary_image[up:down + 1, left:right + 1]
            binary_lines.append(b_line)

        # Return list of separated lines.
        return binary_lines

    @staticmethod
    def detect_peaks(horizontal_hist, threshold_high):

        peaks = []

        i = 0
        while i < len(horizontal_hist):
            # If the black pixels density of the row is below than threshold
            # then continue to the next row.
            if horizontal_hist[i] < threshold_high:
                i += 1
                continue

            # Get the row with the maximum density from the following
            # probable row lines.
            peak_idx = i
            # The code will enter this while loop only if the row index(i) is greater than the threshold
            # and its neighbours is still greater than the threshold too
            while i < len(horizontal_hist) and LineSegmentor.is_probable_peak(i, horizontal_hist, threshold_high):
                if horizontal_hist[i] > horizontal_hist[peak_idx]:
                    peak_idx = i
                i += 1

            # Add peak row index to the list.
            peaks.append(peak_idx)
        return peaks

    @staticmethod
    def detect_valleys(peaks, avg_peaks_dist, horizontal_hist):

        valleys = [0]

        i = 1
        while i < len(peaks):
            up = peaks[i - 1]
            down = peaks[i]
            i += 1

            expected_valley = down - avg_peaks_dist // 2
            valley_idx = up

            while up < down:
                distance1 = np.abs(up - expected_valley)
                distance2 = np.abs(valley_idx - expected_valley)

                condition1 = horizontal_hist[up] < horizontal_hist[valley_idx]
                condition2 = horizontal_hist[up] == horizontal_hist[valley_idx] and distance1 < distance2

                if condition1 or condition2:
                    valley_idx = up

                up += 1

            valleys.append(valley_idx)

        valleys.append(len(horizontal_hist) - 1)
        return valleys

    @staticmethod
    def detect_line_boundaries(valleys, binary_image, horizontal_hist):

        # Get image dimensions.
        height, width = binary_image.shape

        lines_boundaries = []

        i = 1
        while i < len(valleys):
            up = valleys[i - 1]
            down = valleys[i]
            left = 0
            right = width - 1
            i += 1

            while up < down and horizontal_hist[up] == 0:
                up += 1
            while down > up and horizontal_hist[down] == 0:
                down -= 1

            vertical_hist = np.sum(binary_image[up:down + 1, :], axis=0) // 255

            while left < right and vertical_hist[left] == 0:
                left += 1
            while right > left and vertical_hist[right] == 0:
                right -= 1

            lines_boundaries.append((left, up, right, down))

        return lines_boundaries

    @staticmethod
    def is_probable_valley(row, horizontal_hist, threshold_low):

        width = 30
        count = 0

        for i in range(-width, width):
            if row + i < 0 or row + i >= len(horizontal_hist):
                return True
            if horizontal_hist[row + i] <= threshold_low:
                count += 1

        if count * 2 >= width:
            return True

        return False

    @staticmethod
    def detect_missing_peaks_valleys(valleys, avg_peak_distance, horizontal_hist, threshold_low, peaks):

        i = 1
        found = False

        while i < len(valleys):
            # Calculate distance between two consecutive valleys.
            up, down = valleys[i - 1], valleys[i]
            dis = down - up

            i += 1

            # If the distance is about twice the average distance between
            # two consecutive peaks, then it is most probable that we are missing
            # a line in between these two valleys.
            if dis < 1.5 * avg_peak_distance:
                continue

            u = up + avg_peak_distance
            d = min(down, u + avg_peak_distance)

            while (d - u) * 2 > avg_peak_distance:
                if LineSegmentor.is_probable_valley(u, horizontal_hist, threshold_low) and \
                        LineSegmentor.is_probable_valley(d, horizontal_hist, threshold_low):
                    peak = LineSegmentor.get_peak_in_range(u, d, horizontal_hist)
                    if horizontal_hist[peak] > threshold_low:
                        peaks.append(LineSegmentor.get_peak_in_range(u, d, horizontal_hist))
                        found = True

                u = u + avg_peak_distance
                d = min(down, u + avg_peak_distance)

        # Re-distribute peaks and valleys if new ones are found.
        if found:
            peaks.sort()
            valleys = LineSegmentor.detect_valleys(peaks, avg_peak_distance, horizontal_hist)
        return valleys

    @staticmethod
    def get_peak_in_range(up, down, horizontal_hist):
        peak_idx = up

        while up < down:
            if horizontal_hist[up] > horizontal_hist[peak_idx]:
                peak_idx = up
            up += 1

        return peak_idx

    @staticmethod
    def is_probable_peak(row, horizontal_hist, threshold_high):

        width = 15

        for i in range(-width, width):
            # Checking if row is at the beginning of the image or at the end of the image
            if row + i < 0 or row + i >= len(horizontal_hist):
                continue
            if horizontal_hist[row + i] >= threshold_high:
                return True

        return False

    @staticmethod
    def segmentation_pipeline(binary_image):

        # Get horizontal histogram.
        horizontal_hist = np.sum(binary_image, axis=1, dtype=int) // 255

        # Get line density thresholds.
        threshold_high = int(np.max(horizontal_hist) // 3)
        threshold_low = 25

        # Calculate peaks and valleys of the page.
        peaks = LineSegmentor.detect_peaks(horizontal_hist, threshold_high)
        avg_peak_distance = int((peaks[-1] - peaks[0]) // len(peaks))
        valleys = LineSegmentor.detect_valleys(peaks, avg_peak_distance, horizontal_hist)
        valleys = LineSegmentor.detect_missing_peaks_valleys(valleys, avg_peak_distance,
                                                             horizontal_hist, threshold_low, peaks)
        lines_boundaries = LineSegmentor.detect_line_boundaries(valleys, binary_image, horizontal_hist)
        binary_lines = LineSegmentor.segment(lines_boundaries, binary_image)

        return binary_lines
