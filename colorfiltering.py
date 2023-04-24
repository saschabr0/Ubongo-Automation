import cv2 as cv
import numpy as np
import os
from PIL import Image
# cyan ranges: (240-255; 240-255; 0-15)
# yellow ranges: (0-30; 225-255; 225-255)
# centroid of the playing card in solution: (80, 115)
card_centroid = [80, 115]
bgr_color = [0, 255, 255]


class ColorFilter(object):

    def __init__(self, visualize=False):
        self.visualize = visualize

    def get_cyan_tile(self, image):
        input_image = image

        lower_range_cyan = np.array([240, 240, 0])  # Set the Lower range value of color in BGR
        upper_range_cyan = np.array([255, 255, 15])  # Set the Upper range value of color in BGR

        mask_cyan = cv.inRange(input_image, lower_range_cyan, upper_range_cyan)  # Create a mask with range
        result_cyan = cv.bitwise_and(input_image, input_image, mask=mask_cyan)  # Performing bitwise and operation with mask in img variable

        return result_cyan

    def get_yellow_tile(self, image):
        input_image = image

        lower_range_yellow = np.array([0, 225, 225])  # Set the Lower range value of color in BGR
        upper_range_yellow = np.array([30, 255, 255])  # Set the Upper range value of color in BGR

        mask_yellow = cv.inRange(input_image, lower_range_yellow, upper_range_yellow)  # Create a mask with range
        result_yellow = cv.bitwise_and(input_image, input_image, mask=mask_yellow)  # Performing bitwise and operation with mask in img variable

        return result_yellow

    def get_cyan_centroid(self, image):
        output_cyan_tile = image

        gray = cv.cvtColor(output_cyan_tile, cv.COLOR_BGR2GRAY)

        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        # Find contours in the image
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Loop through the contours and draw them on the input image
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(output_cyan_tile, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Calculate the centroid of the object
            M = cv.moments(c)
            cx_cyan = int(M["m10"] / M["m00"])
            cy_cyan = int(M["m01"] / M["m00"])

        """Checking if centroid is inside/outside contour"""
        img_check_cyan_centroid = Image.fromarray(output_cyan_tile)
        r, g, b = img_check_cyan_centroid.getpixel((cx_cyan, cy_cyan))
        print("the rgb values of the cyan tile are: ")
        print((r, g, b))

        # Draw the centroid on the image
        cv.circle(output_cyan_tile, (cx_cyan, cy_cyan), 3, (0, 0, 255), -1)

        return output_cyan_tile, cx_cyan, cy_cyan

    def get_yellow_centroid(self, image):
        output_yellow_tile = image

        gray = cv.cvtColor(output_yellow_tile, cv.COLOR_BGR2GRAY)

        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        # Find contours in the image
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Loop through the contours and draw them on the input image
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(output_yellow_tile, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Calculate the centroid of the object
            M = cv.moments(c)
            cx_yellow = int(M["m10"] / M["m00"])
            cy_yellow = int(M["m01"] / M["m00"])

        """Checking if centroid is inside/outside contour"""
        img_check_yellow_centroid = Image.fromarray(output_yellow_tile)
        r, g, b = img_check_yellow_centroid.getpixel((cx_yellow, cy_yellow))
        print("the rgb values of the yellow tile are: ")
        print((r, g, b))

        # Draw the centroid on the image
        cv.circle(output_yellow_tile, (cx_yellow, cy_yellow), 3, (0, 0, 255), -1)

        return output_yellow_tile, cx_yellow, cy_yellow

    def dist_cyan_centroid(self, c_c_x, c_c_y):
        c_x_dist = c_c_x-card_centroid[0]
        c_y_dist = c_c_y-card_centroid[1]

        return c_x_dist, c_y_dist

    def dist_yellow_centroid(self, c_y_x, c_y_y):
        y_x_dist = c_y_x-card_centroid[0]
        y_y_dist = c_y_y-card_centroid[1]

        return y_x_dist, y_y_dist

    def __str__(self):
        desc = "This is a color filter"
        return desc
