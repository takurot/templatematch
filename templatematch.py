import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from skimage import data
from skimage.feature import match_template

from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift


def main():
    TEMPLATE_SIZE = 32
    capture = cv2.VideoCapture(0)

    while True:
        ret, org_img = capture.read()
        img = cv2.resize(org_img, dsize=None, fx=0.5, fy=0.5)
        height, width, channnel = img.shape[:3]
        
        y1 = int(height/2-TEMPLATE_SIZE)
        y2 = int(height/2+TEMPLATE_SIZE)
        x1 = int(width/2-TEMPLATE_SIZE)
        x2 = int(width/2+TEMPLATE_SIZE)

        # print(width, height, x1, x2, y1, y2)
        if ret != True:
            print("Error1")
            return
        disp = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.imshow("Select Template(Size 64x64) Press ESC", disp)
        key = cv2.waitKey(10)
        if key == 27: # ESC 
            break

    image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    template = image[y1:y2, x1:x2]

    cv2.imshow("Template-2", template)

    while True:
        time_start = time.time()
        ret, org_img2 = capture.read()
        if ret != True:
            print("Error2")
            return
        img2 = cv2.resize(org_img2, dsize=None, fx=0.5, fy=0.5)
        offset_image = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        time_cap = int((time.time() - time_start) * 1000)

        time_start = time.time()
        result = match_template(offset_image, template)
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]
        meas_image = offset_image[y:(y+TEMPLATE_SIZE*2), x:(x+TEMPLATE_SIZE*2)]
        # print (template.shape[0], template.shape[1], meas_image.shape[0], meas_image.shape[1])
        shift, error, diffphase = register_translation(template, meas_image, 100)
        time_meas = int((time.time() - time_start) * 1000)

        cv2.rectangle(img2, (x, y), (x+TEMPLATE_SIZE*2, y+TEMPLATE_SIZE*2), (0, 255, 0), 3)

        cv2.imshow("Real Time Measurement 640x480", img2)

        print ("Capture[ms]:", time_cap, "Meas[ms]:", time_meas, "X[pix]:", x+TEMPLATE_SIZE+shift[0], "Y[pix]:", y+TEMPLATE_SIZE+shift[1])

        key = cv2.waitKey(10)
        if key == 27: # ESC 
            break

if __name__ == "__main__":
    main()
