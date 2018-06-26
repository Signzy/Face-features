#local binary patterns
# <script src="https://gist.github.com/absaravanan/56abdeb5aec053b986512066987f11fc.js"></script>

from skimage.feature import local_binary_pattern
from matplotlib import pyplot
import cv2

def lbp(img_filepath):
	im = cv2.imread(img_filepath)
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	radius = 9
	no_points = 8 * radius
	lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')


	pyplot.figure(figsize=(10,5))
	pyplot.subplot(121)
	pyplot.imshow(im, cmap='gray')
	pyplot.title("Original image")

	pyplot.subplot(122)
	pyplot.imshow(lbp, cmap='gray')
	pyplot.title("lbp")
	pyplot.show()	



# gober wabelets
# <script src="https://gist.github.com/absaravanan/a09cc5241398680bcea184d0cb9e0b7d.js"></script>

import cv2
import numpy
import math
import bob.ip.gabor
import bob.sp
from matplotlib import pyplot

def gober_wavelets(img_filepath):

	im = cv2.imread(img_filepath)
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	height, width = im.shape[:2]

	# create Gabor wavelet
	wavelet = bob.ip.gabor.Wavelet(resolution = (height, width), frequency = (math.pi/8., 0))

	# compute wavelet transform in frequency domain
	freq_image = bob.sp.fft(im_gray.astype(numpy.complex128))
	transformed_freq_image = wavelet.transform(freq_image)
	transformed_image = bob.sp.ifft(transformed_freq_image)

	# get layers of the image
	real_image = numpy.real(transformed_image)
	abs_image = numpy.abs(transformed_image)

	# get the wavelet in spatial domain
	spat_wavelet = bob.sp.ifft(wavelet.wavelet.astype(numpy.complex128))
	real_wavelet = numpy.real(spat_wavelet)
	# align wavelet to show it centered
	aligned_wavelet = numpy.roll(numpy.roll(real_wavelet, 64, 0), 64, 1)


	# create figure
	pyplot.figure(figsize=(20,5))
	pyplot.subplot(141)
	pyplot.imshow(im, cmap='gray')
	pyplot.title("Original image")

	pyplot.subplot(142)
	pyplot.imshow(aligned_wavelet, cmap='gray')
	pyplot.title("Gabor wavelet")

	pyplot.subplot(143)
	pyplot.imshow(real_image, cmap='gray')
	pyplot.title("Real part")

	pyplot.subplot(144)
	pyplot.imshow(abs_image, cmap='gray')
	pyplot.title("Abs part")

	pyplot.show()




# local phase quantisation
# <script src="https://gist.github.com/absaravanan/a145f3b1a364d2a499bca79525b2667b.js"></script>

import cv2
import numpy as np
from scipy.signal import convolve2d

def lpq(img_filepath, winSize=3,freqestim=1,mode='im'):

	img = cv2.imread(img_filepath,0)
	rho=0.90

	STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
	sigmaS=(winSize-1)/4 # Sigma for STFT Gaussian window (applied if freqestim==2)
	sigmaA=8/(winSize-1) # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

	convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

	img=np.float64(img) # Convert np.image to double
	r=(winSize-1)/2 # Get radius from window size
	x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

	if freqestim==1:  #  STFT uniform window
	    #  Basic STFT filters
	    w0=np.ones_like(x)
	    w1=np.exp(-2*np.pi*x*STFTalpha*1j)
	    w2=np.conj(w1)

	## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
	# Run first filters
	filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
	filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
	filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
	filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

	# Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
	freqResp=np.dstack([filterResp1.real, filterResp1.imag,
	                    filterResp2.real, filterResp2.imag,
	                    filterResp3.real, filterResp3.imag,
	                    filterResp4.real, filterResp4.imag])

	## Perform quantization and compute LPQ codewords
	inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
	LPQdesc=((freqResp>0)*(2**inds)).sum(2)

	## Switch format to uint8 if LPQ code np.image is required as output
	if mode=='im':
		LPQdesc=np.uint8(LPQdesc)


		pyplot.figure(figsize=(20,10))
		pyplot.subplot(121)
		pyplot.imshow(img, cmap='gray')
		pyplot.title("Original image")

		pyplot.subplot(122)
		pyplot.imshow(LPQdesc, cmap='gray')
		pyplot.title("lpq")
		pyplot.show()


	## Histogram if needed
	if mode=='nh' or mode=='h':
	    LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

	## Normalize histogram if needed
	if mode=='nh':
	    LPQdesc=LPQdesc/LPQdesc.sum()

	print (LPQdesc)
	# return LPQdesc



# gabor jet similarities
# <script src="https://gist.github.com/absaravanan/a8c0f776e32aa3744fd629faad46fe21.js"></script>

import numpy
import bob.io.base
import bob.io.base.test_utils
import bob.ip.gabor
import cv2

def gabor_jet_similarities(img_filepath):

	# load test image
	# image = bob.io.base.load(bob.io.base.test_utils.datafile("testimage.hdf5", 'bob.ip.gabor'))
	im = cv2.imread(img_filepath)
	image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# perform Gabor wavelet transform on image
	gwt = bob.ip.gabor.Transform()
	trafo_image = gwt(image)
	# extract Gabor jet at right eye location
	pos = (380, 220)
	eye_jet = bob.ip.gabor.Jet(trafo_image, pos)

	# compute similarity field over the whole image
	cos_sim = bob.ip.gabor.Similarity("ScalarProduct")
	disp_sim = bob.ip.gabor.Similarity("Disparity", gwt)
	cos_image = numpy.zeros(((image.shape[0]+1)//4, (image.shape[1]+1)//4))
	disp_image = numpy.zeros(((image.shape[0]+1)//4, (image.shape[1]+1)//4))
	# .. re-use the same Gabor jet object to avoid memory allocation
	image_jet = bob.ip.gabor.Jet()
	for y in range(2, image.shape[0], 4):
	  for x in range(2, image.shape[1], 4):
	    image_jet.extract(trafo_image, (y,x))
	    cos_image[y//4,x//4] = cos_sim(image_jet, eye_jet)
	    # disp_image[y//4,x//4] = disp_sim(image_jet, eye_jet)

	# plot the image and the similarity map side-by-side
	from matplotlib import pyplot
	pyplot.subplot(121)
	pyplot.imshow(image, cmap='gray')
	pyplot.plot(pos[1], pos[0], "gx", markersize=20, mew=4)
	pyplot.axis([0, image.shape[1], image.shape[0], 0])
	pyplot.axis("scaled")

	pyplot.subplot(122)
	pyplot.imshow(cos_image, cmap='jet')
	pyplot.title("Similarities (cosine)")

	pyplot.show()




# difference of gaussians
# <script src="https://gist.github.com/absaravanan/d5525494b35eeb3e4e3e63032f18ed07.js"></script>

from skimage import data, feature, color, img_as_float, filters
from matplotlib import pyplot as plt
import cv2

def diff_of_gaussians(img_filepath):

	im = cv2.imread(img_filepath)
	img = color.rgb2gray(im)

	k = 1.6

	plt.subplot(2,3,1)
	plt.imshow(im)
	plt.title('Original Image')

	for idx,sigma in enumerate([4.0,8.0,16.0,32.0]):
		s1 = filters.gaussian(img,k*sigma)
		s2 = filters.gaussian(img,sigma)

		# multiply by sigma to get scale invariance
		dog = s1 - s2
		plt.subplot(2,3,idx+2)
		print (dog.min(),dog.max())
		plt.imshow(dog,cmap='RdBu')
		plt.title('DoG with sigma=' + str(sigma) + ', k=' + str(k))

	ax = plt.subplot(2,3,6)
	blobs_dog = [(x[0],x[1],x[2]) for x in feature.blob_dog(img, min_sigma=4, max_sigma=32,threshold=0.5,overlap=1.0)]
	# skimage has a bug in my version where only maxima were returned by the above
	blobs_dog += [(x[0],x[1],x[2]) for x in feature.blob_dog(-img, min_sigma=4, max_sigma=32,threshold=0.5,overlap=1.0)]

	#remove duplicates
	blobs_dog = set(blobs_dog)

	img_blobs = color.gray2rgb(img)
	for blob in blobs_dog:
		y, x, r = blob
		c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
		ax.add_patch(c)
	plt.imshow(img_blobs)
	plt.title('Detected DoG Maxima')

	plt.show()




# Histogram of oriented gradients
# <script src="https://gist.github.com/absaravanan/2f4182aaf5f17c7fdac8b8db7cf3484c.js"></script>

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from scipy import misc

def HoG(img_filepath):

	image = misc.imread(img_filepath, flatten=True)

	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
	                    cells_per_block=(1, 1), visualise=True)
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

	ax1.axis('off')
	ax1.imshow(image, cmap=plt.cm.gray)
	ax1.set_title('Input image')

	# Rescale histogram for better display
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

	ax2.axis('off')
	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax2.set_title('Histogram of Oriented Gradients')
	plt.show()




# FFT 1D
# <script src="https://gist.github.com/absaravanan/90f24d4bf4b7a340ed3d4bcd4da379cc.js"></script>

import cv2
import numpy as np
from matplotlib import pyplot as plt

def fft(img_filepath):

	img = cv2.imread(img_filepath,0)

	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	fft = 20*np.log(np.abs(fshift))

	plt.subplot(121),plt.imshow(img, cmap = 'gray')
	plt.title('Original image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(fft, cmap = 'gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()





# Blob features
# <script src="https://gist.github.com/absaravanan/c65101543cb61b65566fe7c8c17c72dd.js"></script>

import cv2
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def blob_features(img_filepath):

	image = cv2.imread(img_filepath)
	image_gray = rgb2gray(image)

	blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

	# Compute radii in the 3rd column.
	blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

	blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
	blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

	blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

	blobs_list = [blobs_log, blobs_dog, blobs_doh]
	colors = ['yellow', 'lime', 'red']
	titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
	          'Determinant of Hessian']
	sequence = zip(blobs_list, colors, titles)

	fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
	ax = axes.ravel()

	for idx, (blobs, color, title) in enumerate(sequence):
	    ax[idx].set_title(title)
	    ax[idx].imshow(image, interpolation='nearest')
	    for blob in blobs:
	        y, x, r = blob
	        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
	        ax[idx].add_patch(c)
	    ax[idx].set_axis_off()

	plt.tight_layout()
	plt.show()




# Censure features
# <script src="https://gist.github.com/absaravanan/e91efe6c3ce6ec18a7467f24ebd71038.js"></script>

import cv2
from skimage import data
from skimage import transform as tf
from skimage.feature import CENSURE
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def censure_features(img_filepath):

	img_orig = cv2.imread(img_filepath,0)

	tform = tf.AffineTransform(scale=(1.5, 1.5), rotation=0.5,
	                           translation=(150, -200))
	img_warp = tf.warp(img_orig, tform)
	detector = CENSURE()
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

	ax1.imshow(img_orig, cmap=plt.cm.gray)
	ax1.set_title("Original Image")

	detector.detect(img_orig)


	ax2.imshow(img_orig, cmap=plt.cm.gray)
	ax2.scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
	              2 ** detector.scales, facecolors='none', edgecolors='r')
	ax2.set_title("Censure features")
	plt.show()




# ORB features
# <script src="https://gist.github.com/absaravanan/f3099c864b39edf2a47dafced4ba3319.js"></script>

import numpy as np
import cv2
from matplotlib import pyplot as plt

def ORB(img_filepath):

	img = cv2.imread(img_filepath,0)

	# Initiate STAR detector
	orb = cv2.ORB_create()

	# find the keypoints with ORB
	kp = orb.detect(img,None)

	# compute the descriptors with ORB
	kp, des = orb.compute(img, kp)

	# draw only keypoints location,not size and orientation
	img2 = cv2.drawKeypoints(img,kp,img.copy(),color=(0,255,0), flags=0)

	pyplot.figure(figsize=(10,5))
	pyplot.subplot(121)
	pyplot.imshow(img, cmap='gray')
	pyplot.title("Original image")

	pyplot.subplot(122)
	pyplot.imshow(img2, cmap='gray')
	pyplot.title("ORB")
	pyplot.show()






if __name__  == "__main__":
	src_img = "/home/saravanan/ABS/my_face.jpg"

	# lbp(src_img)
	# gober_wavelets(src_img)
	# lpq(src_img)
	# gabor_jet_similarities(src_img)
	# diff_of_gaussians(src_img)
	# HoG(src_img)
	# blob_features(src_img)
	# censure_features(src_img)
	# ORB(src_img)
	# fft(src_img)
