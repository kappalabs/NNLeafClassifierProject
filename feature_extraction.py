# coding=utf-8
import numpy as np
import matplotlib.image as mpimg
from skimage import measure
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import util
from skimage.measure import label
from random import uniform


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
    # img = ndi.gaussian_filter(img, 1)

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


def extract_volume_matrix(in_file, rows, columns):
    """
    Extracts information about volume and shape of the object on given image.
    The resulting matrix with specified shape will contain information about how
    much of the object is present in the according window.
    For further details uncomment visual debug section.

    :param in_file: File containing image to be processed.
    :param rows: Number of desired rows - windows in single ange view.
    :param columns: Number of desired columns - angles separating the image.
    :return: Volume-shape matrix in a vector form with values in [0; 1].
    """
    # Read an image file using matplotlib into a numpy array
    img = mpimg.imread(in_file)

    # Use image processing module of scipy to find the center of the leaf
    cy, cx = ndi.center_of_mass(img)

    out = np.zeros((rows, columns))

    # Number of taken samples to determine 'usage' of a window - Monte-carlo estimation
    samples = 10
    xs = []
    ys = []

    width, height = img.shape
    max_dist = max(width, height) / 2
    phi_step = 360 / float(columns) / 2.
    dist_step = max_dist / float(rows) / 2.
    for phi in range(columns):
        for dist in range(rows):
            # Monte-carlo method to find out used part of this window
            for _ in range(samples):
                dist_tmp = dist_step + dist * dist_step * 2. + uniform(-dist_step, dist_step)
                phi_range = np.tan(phi_step / 180 * np.pi) * dist_tmp
                phi_tmp = phi * phi_step * 2. + uniform(-phi_range, phi_range)
                x, y = pol2cart(dist_tmp, phi_tmp / 180 * np.pi)
                x_pos = int(x + cx)
                y_pos = int(y + cy)
                miss = 0
                if x_pos < 0 or x_pos >= height - 1:
                    miss = 1
                if y_pos < 0 or y_pos >= width - 1:
                    miss = 1
                xs.append(x_pos)
                ys.append(y_pos)
                if miss == 0 and img[y_pos, x_pos]:
                    out[dist][phi] += 1
    out /= samples

    # Shift to preserve rotation invariance
    out = np.roll(out, -int(np.argmax(out.sum(axis=0))), axis=1)

    # ## For visual debug

    # plt.subplot(121)
    # plt.imshow(img, cmap='Set3')
    #
    # plt.subplot(121)
    # plt.scatter(xs, ys)
    #
    # plt.subplot(122)
    # plt.imshow(out, cmap='Greys',  interpolation='nearest')
    #
    # plt.show()

    return np.asarray(out).reshape(-1)


def extract_image_descriptors(in_file, num_descriptors, shape_rows=8, shape_columns=20):
    """
    Generates a vector of characterizations from given file
    :param in_file: File containing image to be processed.
    :param num_descriptors: Number of required fourier descriptors.
    :param shape_rows: Number of parts to use for a shape-volume retrieval in one direction.
    :param shape_columns: Number of angles to use a shape-volume retrieval.
    :return: Vector with all desired characterizations of the input image.
    """
    # Read an image file using matplotlib into a numpy array
    img = mpimg.imread(in_file)
    cy, cx = ndi.center_of_mass(img)

    # Find minimal bounding box
    img = util.img_as_ubyte(img) > 110
    label_img = label(img)
    props = measure.regionprops(label_img, intensity_image=img)
    min_row, min_col, max_row, max_col = props[0].bbox
    wid = max_col - min_col
    hei = max_row - min_row

    # Stack all characterizations into one vector
    all_ds = np.hstack((
        extract_fourier_descriptors(in_file, num_descriptors),  # Fourier descriptors
        extract_volume_matrix(in_file, rows=shape_rows, columns=shape_columns),  # Vectorized shape-volume matrix
        min(wid, hei) / float(max(wid, hei)),  # Similarity to rectangle
        cy / float(hei),  # Normalized centroid y position
        cx / float(wid),  # Normalized centroid x position
        ))

    return all_ds
