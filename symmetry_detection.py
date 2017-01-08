import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage as ndi
import scipy.signal as signal
from matplotlib.pyplot import fill
from pylab import rcParams
from scipy.signal import argrelextrema
from skimage import measure
from sklearn import metrics


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
def preprocess(img, do_portrait=False, do_resample=300,do_threshold=250):
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


#title, img = list(get_imgs([58]))[0]  #48 #188 #709 #53


# First, design the Buterworth filter
N  = 2    # Filter order
Wn = 0.04 # Cutoff frequency
b, a = signal.butter(N, Wn, analog=False)

for img_no in range(1,10):

    img = get_img(img_no)
     
    shape = get_contour(img)
    orig_shape = np.copy(shape)

    shape_cx, shape_cy = ndi.center_of_mass(img)

    # apply filter
    shape_x, shape_y = coords_to_cols(shape)
    orig_shape_x, orig_shape_y = coords_to_cols(orig_shape)

    shape_y = signal.filtfilt(b,a, shape_y, padtype="odd")
    shape_x = signal.filtfilt(b,a, shape_x, padtype="odd")

    rx = np.reshape(shape_x,(-1,1))
    ry = np.reshape(shape_y,(-1,1))
    shape = np.concatenate((ry,rx), axis = 1)


    curve = dist_line_point(shape, [shape_cx, shape_cy])


    curve = normalize(curve)


    conv = np.convolve(np.hstack((curve, curve)), curve, 'same')

    max_indices_conv = argrelextrema(curve, np.greater, order=50)[0]

    max_conv = np.argmax(conv) % len(curve)


    # local extrema
    max_indices = argrelextrema(curve, np.greater, order=20)[0]
    min_indices = argrelextrema(curve, np.less, order=20)[0]
    max_indices = np.insert(max_indices, 0, 0)
    print (min_indices.shape, max_indices.shape)
    extrema = np.insert(min_indices, np.arange(len(max_indices)), max_indices)
    extrema = np.sort(extrema)
    print ("extrema:",extrema)

    # conv = np.convolve(extrema, extrema[::-1], 'same')
    # max_conv = np.argmax(conv)

    best,second_best = best_symmetry(extrema,shape)

    rcParams['figure.figsize'] = (16,10)

    ax1 = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
    ax1.set_title('Image #' + str(img_no))

    ax1.plot(shape_x, shape_y, c='b')
    #ax1.plot(orig_shape_x, orig_shape_y, c='g')
    ax1.scatter(shape_cy, shape_cx, marker='x')
    ax1.scatter(shape_x[max_indices], shape_y[max_indices],linewidth=0, s=30, c='r')
    ax1.scatter(shape_x[min_indices], shape_y[min_indices],linewidth=0, s=30, c='b')
    ax1.scatter(shape_x[best], shape_y[best],linewidth=0, s=50, c='g')
    ax1.scatter(shape_x[second_best], shape_y[second_best],linewidth=0, s=50, c='y')

    ax1.scatter(shape_x[max_conv], shape_y[max_conv], linewidth=0, s=100, c='g')
    ax1.scatter(shape_x[max_indices_conv], shape_y[max_indices_conv],linewidth=0, s=90, c='y')

    ax2 = plt.subplot2grid((2,3), (0,2))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Distance curve ('+ 
                  str(len(curve))+'features)')
    ax2.plot(range(len(curve)), curve, c='g')
    ax2.scatter(max_indices, curve[max_indices],linewidth=0, s=30, c='r')
    ax2.scatter(min_indices, curve[min_indices],linewidth=0, s=30, c='b')

    ax3 = plt.subplot2grid((2,3), (1,2))
    ax3.set_title('convolution')
    #ax3.text(2460, 30, 'correlation', rotation=270)
    #ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.plot(range(len(conv)), conv, c='r')


    plt.show()
    # input("press Enter")

