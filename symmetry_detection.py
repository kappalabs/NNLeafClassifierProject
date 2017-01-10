import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage as ndi
import scipy.signal as signal
import scipy.fftpack as fftpack
from matplotlib.pyplot import fill
from pylab import rcParams
from scipy.signal import argrelextrema
from skimage import measure
from sklearn import metrics
from collections import deque


# ----------------------------------------------------- I/O ---

def read_img(img_no):
    """reads image from disk"""
    return mpimg.imread('images/' + str(img_no) + '.jpg')


def get_imgs(num):
    """convenience function, yields random sample from leaves"""
    if type(num) == int:
        imgs = range(1, 1584)
        num = np.random.choice(imgs, size=num, replace=False)
        
    for img_no in num:
        yield img_no, preprocess(read_img(img_no))

def get_img(img_no):
    # return preprocess(read_img(img_no))
    return read_img(img_no)

# ----------------------------------------------------- preprocessing ---

def threshold(img, threshold=250):
    """splits img to 0 and 255 values at threshold"""
    return ((img > threshold) * 255).astype(img.dtype)


def portrait(img):
    """makes all leaves stand straight"""
    y, x = np.shape(img)
    return img.transpose() if x > y else img
    

def resample(img, size):
    """resamples img to size without distorsion"""
    ratio = size / max(np.shape(img))
    return sp.misc.imresize(img, ratio, mode='L', interp='nearest')



def normalize_mean_std(arr1d):
    """move mean to zero, 1st SD to -1/+1"""
    return (arr1d - arr1d.mean()) / arr1d.std()

def normalize(arr1d):
    return arr1d/np.max(arr1d)

def coords_to_cols(coords):
    """from x,y pairs to feature columns"""
    return coords[::,1], coords[::,0]
    # return np.hstack((coords[::, 1], coords[0, 1])), np.hstack((coords[::, 0], coords[0, 0]))


def get_contour(img):
    """returns the coords of the longest contour"""
    return max(measure.find_contours(img, .8), key=len)


def downsample_contour(coords, bins=1024):
    """splits the array to ~equal bins, and returns one point per bin"""
    edges = np.linspace(0, coords.shape[0], 
                       num=bins).astype(int)
    for b in range(bins-1):
        yield [coords[edges[b]:edges[b+1],0].mean(), 
               coords[edges[b]:edges[b+1],1].mean()]



def near0_ix(timeseries_1d, radius=5):
    """finds near-zero values in time-series"""
    return np.where(timeseries_1d < radius)[0]


def dist_line_line(src_arr, tgt_arr):
    """
    returns 2 tgt_arr length arrays, 
    1st is distances, 2nd is src_arr indices
    """
    return np.array(sp.spatial.cKDTree(src_arr).query(tgt_arr))


def dist_line_point(src_arr, point):
    src = np.copy(src_arr)
    """returns 1d array with distances from point"""
    point1d = [[point[0], point[1]]] * len(src_arr)
    return metrics.pairwise.paired_distances(src_arr, point1d)



# wrapper function for all preprocessing tasks    
def preprocess(img, do_portrait=False, do_resample=300, do_threshold=250):
    """ prepares image for processing"""
    if do_portrait:
        img = portrait(img)
    if do_resample:
        img = resample(img, size=do_resample)
    if do_threshold:
        img = threshold(img, threshold=do_threshold)
        
    return img


def best_symmetry(extrema, shape):
    size = extrema.size
    errors = []
    for i in range(size):
        err = 0
        for n in range(1, size // 2):
            diff1 = np.abs(shape[extrema[i]] - shape[extrema[(i - n) % size]])
            # print(diff1)
            diff2 = np.abs(shape[extrema[(i + n) % size]] - shape[extrema[i]])
            err += np.linalg.norm(diff1 - diff2)
        # print(err)
        errors.append(err)
    arr = np.array(errors)
    sorted = arr.argsort()
    return extrema[sorted[0]], extrema[sorted[1]]
    # return (max_index, max_index1)

def arc_error(start,end,shape):
    size = len(shape)
    err=0

    p1=start
    p2=end
    if (start>end):
        p2=start
        p1=end


    for x in range(p1,p2):
        diff1= np.abs(shape[(x+p1)%size]-shape[p1])
        diff2= np.abs(shape[(p1-x)%size]-shape[p1])     
        err+=np.linalg.norm(diff1-diff2)
    return err


def shape_distance(start, end, shape):
    size = len(shape)
    length = 0

    if start > end:
        end += size
        # tmp = start
        # start = end
        # end = tmp

    for pos in range(start + 1, end):
        length += np.linalg.norm(shape[pos % size] - shape[[(pos - 1) % size]], ord=2)
        # print("pos1=", shape[pos % size], ", pos2=", shape[[(pos - 1) % size]])

    # print("from " + str(start) + " to " + str(end))
    # plt.plot(shape_x, shape_y, c='b')
    # plt.plot(orig_shape_x, orig_shape_y, c='g')
    # plt.scatter(shape_cy, shape_cx, marker='x')
    # plt.scatter(shape_x[range(start + 1, end) % size], shape_y[range(start + 1, end) % size], linewidth=0, s=80, c='r')
    # plt.show()

    return length


def best_symmetry2(extrema,shape,arclength):
    size = extrema.size

    errors = []
    errors_ind = []
    for i in range(size//2+1):
        err=0
        loc_errors = []
        for n in range(1,size):
            err = np.abs(np.abs(extrema[i]-extrema[(i+n)%size]) - arclength/2)
            loc_errors.append(err)
            #print(err)
        #arr = np.array(errors)
        mini = np.argmin(loc_errors)
        #print(min)
        errors.append(loc_errors[mini])
        errors_ind.append([i,(i+mini+1)%size])
    arr = np.array(errors)
    sorted_err= arr.argsort()
    arr2 = np.array(errors_ind)
    sorted = arr2[sorted_err]

    #print(sorted)
    mini=1000000000
    min_index = 0
    for i in range(0,min(8,len(sorted))):
        err = arc_error(extrema[sorted[i][0]], extrema[sorted[i][1]],shape)
        #print(err)
        if (err<mini):
            min_index=i
            mini = err

    return extrema[sorted[min_index][0]], extrema[sorted[min_index][1]]


def rotate(l, n):
    return l[-n:] + l[:-n]


def best_symmetry3(inflexes, curve, shape):
    errors = []
    errors_indexes = []
    num_inflexes = len(inflexes)

    offset = 0
    for i in range(0, len(curve) / 2, 50):
        while i >= inflexes[offset % num_inflexes]:
            offset += 1
        err = 0
        for j in range(int(np.floor(num_inflexes / 2))):
            right_inflex_index = inflexes[(j + offset) % num_inflexes]
            left_inflex_index = inflexes[(num_inflexes - j - 1 + offset) % num_inflexes]
            err += np.abs(shape_distance(i, right_inflex_index, shape) - shape_distance(left_inflex_index, i, shape))

            # plt.plot(shape_x, shape_y, c='b')
            # plt.plot(orig_shape_x, orig_shape_y, c='g')
            # plt.scatter(shape_cy, shape_cx, marker='x')
            # plt.scatter(shape_x[i], shape_y[i], linewidth=0, s=80, c='r')
            # plt.scatter(shape_x[inflexes], shape_y[inflexes], linewidth=0, s=80, c='y')
            # plt.scatter(shape_x[right_inflex_index], shape_y[right_inflex_index], linewidth=0, s=50, c='g')
            # plt.scatter(shape_x[left_inflex_index], shape_y[left_inflex_index], linewidth=0, s=50, c='b')
            # plt.show()
        errors = np.hstack((errors, err))
        errors_indexes.append(i)
        print("i = " + str(i) + ", err = " + str(err))

    error_arr = np.array(errors)
    error_indexes_arr = np.array(errors_indexes)
    sorted_indexes = error_arr.argsort()
    sorted_indexes = error_indexes_arr[sorted_indexes]

    return [sorted_indexes[0], sorted_indexes[1]]

#title, img = list(get_imgs([58]))[0]  #48 #188 #709 #53


def maxima_space(curve):
    space_x = []
    for filter_size in range(1, 200):
        curve_smooth = ndi.gaussian_filter(curve, filter_size / 2, mode='wrap')

        max_indexes = argrelextrema(curve_smooth, np.greater)[0]

        plt.scatter(max_indexes, np.ones((1, max_indexes.size)) * filter_size, linewidth=0, s=1, c='r')

    print(space_x)
    plt.show()


# First, design the Buterworth filter
N  = 2    # Filter order
Wn = 0.05 # Cutoff frequency
b, a = signal.butter(N, Wn, analog=False)

for img_no in range(1, 10):

    img = get_img(img_no)
     
    shape = get_contour(img)
    orig_shape = np.copy(shape)

    shape_cx, shape_cy = ndi.center_of_mass(img)

    # apply filter
    shape_x, shape_y = coords_to_cols(shape)
    orig_shape_x, orig_shape_y = coords_to_cols(orig_shape)

    # shape_y = signal.filtfilt(b, a, shape_y, padtype="odd")
    # shape_x = signal.filtfilt(b, a, shape_x, padtype="odd")

    shape_y = ndi.gaussian_filter(shape_y, 150, mode='wrap')
    shape_x = ndi.gaussian_filter(shape_x, 150, mode='wrap')

    # shape_x = sp.signal.savgol_filter(shape_x, 51, 3)
    # shape_y = sp.signal.savgol_filter(shape_y, 51, 3)

    rx = np.reshape(shape_x, (-1, 1))
    ry = np.reshape(shape_y, (-1, 1))
    shape = np.concatenate((rx, ry), axis=1)

    curve = dist_line_point(shape, [shape_cx, shape_cy])
    curve = normalize(curve)

    # maxima_space(curve)

    derivation = sp.fftpack.diff(curve)
    second_derivation = sp.fftpack.diff(derivation)

    # derivation_x = derivation_x[np.isclose(derivation_x, 0)]
    der_indexes = []
    last_val = second_derivation[0]
    for i in range(1, len(second_derivation)):
        if last_val < 0 <= second_derivation[i] or last_val > 0 >= second_derivation[i]:
            # der_indexes = der_indexes.append(i)
            if len(der_indexes) > 0:
                der_indexes = np.hstack((der_indexes, i))
            else:
                der_indexes = [i]
        last_val = second_derivation[i]

    func = np.zeros(curve.shape)
    print(func.size)
    func[der_indexes] = curve[der_indexes]

    conv = np.convolve(np.hstack((func, func)), func[::-1], 'valid')

    # max_indices_conv = (argrelextrema(conv, np.greater, order=50)[0] - len(curve)/2) % len(curve)
    max_indices_conv = (argrelextrema(conv, np.greater, order=50)[0]) % len(curve)

    print("conv indexes: ", max_indices_conv)

    max_conv = np.argmax(conv) % len(curve)


    # local extrema
    max_indices = argrelextrema(curve, np.greater, order=30)[0]
    min_indices = argrelextrema(curve, np.less, order=30)[0]
    max_indices = np.insert(max_indices, 0, 0)
    print (min_indices.shape, max_indices.shape)
    extrema = np.insert(min_indices, np.arange(len(max_indices)), max_indices)
    extrema = np.sort(extrema)
    print ("extrema:", extrema)

    # conv = np.convolve(extrema, extrema[::-1], 'same')
    # max_conv = np.argmax(conv)

    best, second_best = best_symmetry3(der_indexes, curve, shape)

    rcParams['figure.figsize'] = (16,10)

    ax1 = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
    ax1.set_title('Image #' + str(img_no))

    ax1.plot(shape_x, shape_y, c='b')
    ax1.plot(orig_shape_x, orig_shape_y, c='g')
    ax1.scatter(shape_cy, shape_cx, marker='x')
    # ax1.scatter(shape_x[max_indices], shape_y[max_indices], linewidth=0, s=30, c='r')
    # ax1.scatter(shape_x[min_indices], shape_y[min_indices], linewidth=0, s=30, c='b')
    ax1.scatter(orig_shape_x[best], orig_shape_y[best], linewidth=0, s=70, c='g')
    ax1.scatter(orig_shape_x[second_best], orig_shape_y[second_best], linewidth=0, s=70, c='y')
    ax1.scatter(shape_x[der_indexes], shape_y[der_indexes], linewidth=0, s=50, c='r')

    # ax1.scatter(shape_x[max_indices_conv], shape_y[max_indices_conv], linewidth=0, s=80, c='y')
    # ax1.scatter(shape_x[max_conv], shape_y[max_conv], linewidth=0, s=100, c='g')

    ax2 = plt.subplot2grid((2,3), (0,2))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Distance curve ('+ 
                  str(len(curve))+'features)')
    ax2.plot(range(len(curve)), curve, c='g')
    # ax2.scatter(max_indices, curve[max_indices],linewidth=0, s=30, c='r')
    # ax2.scatter(min_indices, curve[min_indices],linewidth=0, s=30, c='b')
    ax2.scatter(der_indexes, curve[der_indexes], linewidth=0, s=30, c='r')

    ax3 = plt.subplot2grid((2,3), (1,2))
    ax3.set_title('convolution')
    #ax3.text(2460, 30, 'correlation', rotation=270)
    #ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.plot(range(len(conv)), conv, c='r')


    plt.show()
    # input("press Enter")

