import numpy as np  # numeric python lib
import matplotlib.image as mpimg  # reading images to numpy arrays
import matplotlib.pyplot as plt  # to plot any graph
from skimage import measure  # to find shape contour
import scipy.ndimage as ndi  # to determine shape centrality
from pylab import rcParams

# TODO: zatím jen jako demo co tohle umí, později vracet pro parametr 'obrázek' vektor fourierových deskriptorů

rcParams['figure.figsize'] = (6, 6)  # setting default size of plots

# reading an image file using matplotlib into a numpy array
# good ones: 11, 19, 23, 27, 48, 53, 78, 218
img = mpimg.imread('images/19.jpg')

# using image processing module of scipy to find the center of the leaf
cy, cx = ndi.center_of_mass(img)

# scikit-learn imaging contour finding, returns a list of found edges
contours = measure.find_contours(img, .8)

# from which we choose the longest one
contour = max(contours, key=len)


# cartesian to polar coordinates
def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return [rho, phi]


# numpy is smart and assumes the same about us
# if we substract a number from an array of numbers,
# it assumes that we wanted to substract from all members
contour[::, 1] -= cx  # demean X
contour[::, 0] -= cy  # demean Y

# just calling the transformation on all pairs in the set
polar_contour = np.array([cart2pol(rho, phi) for rho, phi in contour])

# and plotting the result
rcParams['figure.figsize'] = (12, 6)


#######################
# Fourier descriptors #
#######################

# Specifies how many descriptors will be extracted
# first_few = 2113 # maximum
first_few = 128
# first_few = 64
# first_few = 32
# first_few = 4


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return [x, y]


plt.subplot(231)
plt.title('Original Cartesian'), plt.xticks([]), plt.yticks([])
plt.scatter(contour[::, 1],  # x axis is radians
            contour[::, 0],  # y axis is distance from center
            linewidth=0, s=2,  # small points, w/o borders
            c=range(len(contour)))  # continuous coloring (so that plots match)

polar_y = polar_contour[::, 0]
polar_x = polar_contour[::, 1]

f = np.fft.fft(polar_y)
f = f[0:first_few + 1] / f[0]
fpolar_x = polar_x[range(0, len(polar_x), int(len(polar_x) / first_few))]
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.subplot(234)
plt.title('Original Polar'), plt.xticks([]), plt.yticks([])
plt.scatter(polar_x, polar_y, linewidth=0, s=2, c=polar_x)

plt.subplot(235)
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.scatter(range(len(f)), magnitude_spectrum, linewidth=0, s=2, c=range(len(f)))
plt.plot(range(len(f)), magnitude_spectrum)

plt.subplot(236)
plt.title('Inverse Polar'), plt.xticks([]), plt.yticks([])
ipolar_y = np.fft.ifft(f)
fpolar_x = fpolar_x[0:len(ipolar_y)]
print(fpolar_x)
print(len(fpolar_x))
print(len(polar_y))
print(len(ipolar_y.real))
plt.plot(fpolar_x, ipolar_y.real)
# plt.plot(ipolar_y.real)

k = 0
ipolar_contourx = np.zeros((len(ipolar_y.real), 1))
ipolar_contoury = np.zeros((len(ipolar_contourx), 1))
for i in ipolar_y.real:
    ipolar_contoury[k], ipolar_contourx[k] = pol2cart(i, fpolar_x[k])
    k += 1

plt.subplot(233)
plt.title('Inverse Cartesian'), plt.xticks([]), plt.yticks([])
# plt.scatter(ipolar_contourx, ipolar_contoury, linewidth=0, s=2, c=range(len(ipolar_contourx)))
plt.plot(ipolar_contourx, ipolar_contoury)

plt.show()
