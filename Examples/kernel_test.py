import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

if __name__=='__main__':
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help="Path to function")
    args = vars(ap.parse_args())

    img = cv2.imread(args['image'])
    plt.imshow(img, cmap='gray')
    plt.show()
