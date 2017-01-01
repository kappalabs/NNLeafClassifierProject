# coding=utf-8
import numpy as np
import matplotlib.image as mpimg
from skimage import measure
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import data, util
from skimage.measure import label


def cart2pol(x, y):
    """
    Cartesian to polar coordinates.

    :param x: Cartesian X axis value.
    :param y: Cartesian Y axis value.
    :return: Coordinates in the polar system.
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return [rho, phi]


def pol2cart(rho, phi):
    """
    Polar to cartesian coordinates.

    :param rho: Distance in polar coordinates.
    :param phi: Angle in polar coordinates.
    :return: Coordinates in the cartesian system.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return [x, y]


def extract_fourier_descriptors(in_file, num_descriptors=64):
    """
    Extracts required number of fourier descriptors from image in given file.

    :param in_file: File containing image to be processed.
    :param num_descriptors: Number of required fourier descriptors.
    """
    # Read an image file using matplotlib into a numpy array
    img = mpimg.imread(in_file)

    # Use image processing module of scipy to find the center of the leaf
    cy, cx = ndi.center_of_mass(img)

    # Blur the image a little before processing
    img = ndi.gaussian_filter(img, 1)

    # scikit-learn imaging contour finding, returns a list of found edges, we select the longest one
    contour = max(measure.find_contours(img, .8), key=len)
    #
    # plt.plot(contour[::, 1], contour[::, 0], linewidth=0.5)
    # plt.imshow(img, cmap='Set3')
    # plt.show()

    # Move contour centroid to zero
    contour[::, 1] -= cx
    contour[::, 0] -= cy

    # Transformation on all pairs in the set
    polar_contour = np.array([cart2pol(rho, phi) for rho, phi in contour])

    # Fourier transform of the polar representation
    f = np.fft.fft(polar_contour[::, 0])

    # Select only few first values of the transform without the first one (bias),
    # scale by the bias to get scale invariance -> fourier descriptors
    fds = f[1:num_descriptors + 1] / f[0]

    # Take the magnitude
    fds = np.log(np.abs(fds))

    return fds

    # # Fourier transform of the polar representation
    # f = np.fft.fft2(polar_contour)
    #
    # # Select only few first values of the transform without the first one (bias),
    # # scale by the bias to get scale invariance -> fourier descriptors
    # fds1 = f[1:num_descriptors / 2 + 1, 0] / f[0, 0]
    # fds2 = f[1:num_descriptors / 2 + 1, 1] / f[0, 1]
    #
    # # Take the magnitude
    # fds1 = np.log(np.abs(fds1))
    # fds2 = np.log(np.abs(fds2))
    #
    # return [fds1, fds2]


def extract_volume_matrix(in_file, rows, columns):
    # Read an image file using matplotlib into a numpy array
    img = mpimg.imread(in_file)

    # Use image processing module of scipy to find the center of the leaf
    cy, cx = ndi.center_of_mass(img)

    #TODO: polární výseče do vektorů

    return []


def extract_image_descriptors(in_file, num_descriptors):
    fds = extract_fourier_descriptors(in_file, num_descriptors)

    # Read an image file using matplotlib into a numpy array
    img = mpimg.imread(in_file)
    cy, cx = ndi.center_of_mass(img)

    img = util.img_as_ubyte(img) > 110
    label_img = label(img)
    props = measure.regionprops(label_img, intensity_image=img)
    min_row, min_col, max_row, max_col = props[0].bbox
    wid = max_col - min_col
    hei = max_row - min_row

    # plt.imshow(img, cmap='Set3')  # show me the leaf
    # plt.scatter([cx, cx1], [cy, cy1])  # show me its center
    # plt.show()

    return [fds, wid / float(hei), cy / float(hei), cx / float(wid)]
