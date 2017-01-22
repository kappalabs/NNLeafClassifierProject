# coding=utf-8
import numpy as np
import matplotlib.image as mpimg
from skimage import measure
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import util
from skimage.measure import label
from random import uniform
from symmetry_detection import dist_line_point, normalize


def cart2pol(x, y):
    """
    Cartesian to polar coordinates.

    :param x: Cartesian X axis value.
    :param y: Cartesian Y axis value.
    :return: Coordinates in the polar system (distance, angle).
    """
    distance = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(y, x)
    return [distance, angle]


def pol2cart(distance, angle):
    """
    Polar to cartesian coordinates.

    :param distance: Distance in polar coordinates.
    :param angle: Angle in polar coordinates.
    :return: Coordinates in the cartesian system.
    """
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    return [x, y]


def extract_distance_line(image, line_length=64, normalization=True):
    """
    Extracts distance line from given image.

    :param normalization: If normalization of the curve (into [0;1]) should be performed.
    :param image: Image to be processed.
    :param line_length: Desired length of the distance curve vector.
    :return: Distance curve of given image measured from center of mass.
    """
    shape = max(measure.find_contours(image, .8), key=len)

    # Blur the shape
    shape = ndi.gaussian_filter(shape, 20)

    # Find center of mass of the leaf
    cy, cx = ndi.center_of_mass(image)

    # Calculate distance curve
    curve = dist_line_point(shape, [cx, cy])

    # plt.subplot(121)
    # plt.plot(curve, linewidth=0.5)

    # Take only given number of points on this curve
    curve = curve[range(0, len(curve), len(curve) / line_length)]
    if normalization:
        curve = normalize(curve)[0:line_length]

    # plt.subplot(122)
    # plt.plot(curve, linewidth=0.5)
    # plt.show()
    # print(curve.shape)

    return np.asarray(curve)


def extract_fourier_descriptors(img_orig, num_descriptors=64):
    """
    Extracts required number of fourier descriptors from image in given file.

    :param img_orig: Image to be processed.
    :param num_descriptors: Number of required fourier descriptors.
    """
    # Transformation on all pairs in the set
    polar_contour = extract_distance_line(img_orig, line_length=num_descriptors*2, normalization=False)

    # Fourier transform of the polar representation
    f = np.fft.fft(polar_contour)

    # Select only few first values of the transform without the first one (bias),
    # scale by the bias to get scale invariance -> fourier descriptors
    fds = f[1:num_descriptors + 1] / f[0]

    # Take the magnitude
    fds = np.log(1 + np.abs(fds))

    # Normalize them
    # fds = normalize(fds)

    return fds


def extract_volume_matrix(image, rows, columns):
    """
    Extracts information about volume and shape of the object on given image.
    The resulting matrix with specified shape will contain information about how
    much of the object is present in the according window.
    For further details uncomment visual debug section.

    :param image: Image to be processed.
    :param rows: Number of desired rows - windows in single ange view.
    :param columns: Number of desired columns - angles separating the image.
    :return: Volume-shape matrix in a vector form with values in [0; 1].
    """
    # Use image processing module of scipy to find the center of the leaf
    cy, cx = ndi.center_of_mass(image)

    out = np.zeros((rows, columns))

    # Number of taken samples to determine 'usage' of a window - Monte-carlo estimation
    samples = 10
    xs = []
    ys = []

    width, height = image.shape
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
                if miss == 0 and image[y_pos, x_pos]:
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


def find_main_axe(image):
    # TODO
    return image


def extract_image_descriptors(in_file, num_descriptors, shape_rows=8, shape_columns=10):
    """
    Generates a vector of characterizations from given file

    :param in_file: File containing image to be processed.
    :param num_descriptors: Number of required fourier descriptors.
    :param shape_rows: Number of parts to use for a shape-volume retrieval in one direction.
    :param shape_columns: Number of angles to use a shape-volume retrieval.
    :return: Vector with all desired characterizations of the input image.
    """
    # Read an image file using matplotlib into a numpy array
    img_orig = mpimg.imread(in_file)
    cy, cx = ndi.center_of_mass(img_orig)

    # Find minimal bounding box
    img_orig = util.img_as_ubyte(img_orig) > 110
    label_img = label(img_orig)
    props = measure.regionprops(label_img, intensity_image=img_orig)
    min_row, min_col, max_row, max_col = props[0].bbox
    wid = max_col - min_col
    hei = max_row - min_row

    # Stack all characterizations into one vector
    all_ds = np.hstack((
        # extract_distance_line(img_orig),  # Distance curve
        extract_fourier_descriptors(img_orig, num_descriptors),  # Fourier descriptors
        # extract_volume_matrix(img_orig, rows=shape_rows, columns=shape_columns),  # Vectorized shape-volume matrix
        min(wid, hei) / float(max(wid, hei)),  # Similarity to rectangle
        cy / float(hei),  # Normalized centroid y position
        cx / float(wid),  # Normalized centroid x position
    ))

    return all_ds


def visualize_leaf(curve, number):
    # Read an image file using matplotlib into a numpy array
    img = mpimg.imread("images/"+str(number)+".jpg")

    # Use image processing module of scipy to find the center of the leaf
    cy, cx = ndi.center_of_mass(img)

    # scikit-learn imaging contour finding, returns a list of found edges, we select the longest one
    contour = max(measure.find_contours(img, .8), key=len)

    # Move contour centroid to zero
    contour[::, 1] -= cx
    contour[::, 0] -= cy

    polar_contour_dist = np.array([cart2pol(rho, phi) for rho, phi in contour])
    # polar_contour_angle = polar_contour_dist[::, 1]
    polar_contour_dist = polar_contour_dist[::, 0]

    contour = extract_distance_line(img, False)

    plt.subplot(131)
    plt.title("picture "+str(number))
    plt.plot(contour[::, 1], contour[::, 0], linewidth=0.5)

    # Calculate reconstruction based on original data
    leaf = curve[64:127]

    max_polar_countour_dist = max(polar_contour_dist)
    max_ipolar_countour_dist = max(leaf)

    k = 0
    ipolar_contourx = np.zeros((len(leaf), 1))
    ipolar_contoury = np.zeros((len(leaf), 1))
    for i in range(len(leaf)):
        ipolar_contourx[k], ipolar_contoury[k] = pol2cart(leaf[k] / max_ipolar_countour_dist * max_polar_countour_dist,
                                                          float(i) / len(leaf) * 2 * np.pi)
        k += 1

    plt.subplot(132)
    plt.title("original reconstructed")
    plt.plot(np.vstack((ipolar_contourx, ipolar_contourx[0])), np.vstack((ipolar_contoury, ipolar_contoury[0])))

    # Calculate best possible reconstruction with given data
    leaf = polar_contour_dist

    k = 0
    ipolar_contourx = np.zeros((len(leaf), 1))
    ipolar_contoury = np.zeros((len(leaf), 1))
    for i in range(len(leaf)):
        ipolar_contourx[k], ipolar_contoury[k] = pol2cart(leaf[k] / max_ipolar_countour_dist * max_polar_countour_dist,
                                                          float(i) / len(leaf) * 2 * np.pi)
                                                          # polar_contour_angle[k])
        k += 1

    plt.subplot(133)
    plt.title("best possible reconstruction")
    plt.plot(np.vstack((ipolar_contourx, ipolar_contourx[0])), np.vstack((ipolar_contoury, ipolar_contoury[0])))

    plt.show()


def visualize_data(data, ids):
    """
    Visualization of all given data. Corresponding IDs of images must be provided.

    :param data: Original matrix with 192 features for every leaf sample.
    :param ids: IDs of corresponding images for given samples.
    """
    for i in range(data.shape[0]):
        visualize_leaf(data[i], number=ids[i])
