import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

def gaussian(size, std_dev=1):
    ax = np.linspace(-(size//2), size//2, size) #
    x, y = np.meshgrid(ax, ax)
    kernel = 1/(2*np.pi*std_dev*std_dev)*np.exp(-((x*x+y*y)/(2*std_dev*std_dev)))
    return kernel


def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output

def filter(image, filter, name, write_img=False):
    out = convolution(image, filter)
    if write_img:
        cv2.imwrite(f'./images/{name}.jpg', out)
    plt.imshow(out, cmap='gray')
    plt.title(name)
    plt.show()
    return out



if __name__=='__main__':
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help="Path to function")
    args = vars(ap.parse_args())


    img = cv2.imread(args['image'], cv2.IMREAD_GRAYSCALE)
    i_height, i_width = img.shape
    gaussian_kernel = gaussian(size=9, std_dev=np.sqrt(9))
    #inverse_kernel = np.linalg.inv(gaussian_kernel)

    """blurred_img = filter(img, gaussian_kernel, "gaussian_blured", write_img=True)
    horiz=filter(blurred_img, sobel_kernel, "edge_detect", write_img=True)
    vertic=filter(blurred_img, np.flip(sobel_kernel.T, axis=0), "vertical_edge_detect", write_img=True)
    total = np.sqrt(horiz*horiz+vertic*vertic)
    total *= 255.0 / total.max()
    plt.imshow(total, cmap='gray')
    plt.title("Horizontal Edge")
    plt.show()"""
    filter(img, np.array([1/9, 0,0,0,0,0,0,0, 1/9]), "motion_blur", True)
    filter(img, np.array([[-2, -1, 0],[-1, 1, 1],[0, 1, 2]]), "emboss", True)
    filter(img, np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]]), "edge glow", True)


