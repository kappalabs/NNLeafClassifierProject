import numpy as np
import matplotlib.image as mpimg
from skimage import measure
import scipy.ndimage as ndi


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

    # scikit-learn imaging contour finding, returns a list of found edges, we select the longest one
    contour = max(measure.find_contours(img, .8), key=len)

    # Move contour centroid to zero
    contour[::, 1] -= cx
    contour[::, 0] -= cy

    # Transformation on all pairs in the set
    polar_contour = np.array([cart2pol(rho, phi) for rho, phi in contour])

    # Fourier transform of the polar representation
    f = np.fft.fft(polar_contour[::, 0])

    # Select only few first values of the transform without the first one (bias),
    # scale by the bias to get scale invariance -> fourier descriptors
    f = f[1:num_descriptors + 1] / f[0]

    return f.real
