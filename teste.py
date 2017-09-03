# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('color.jpeg')
# b,g,r = cv2.split(img)
# rgb_img = cv2.merge([r,g,b])
#
# vet = [6,7,10,11,21,31,86,87,88,89]
# for i in vet:
#     try:
#         gray = cv2.cvtColor(img,i)
#         ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
#         plt.subplot(121),plt.imshow(rgb_img)
#         plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#         plt.subplot(122),plt.imshow(thresh, 'gray')
#         plt.title("Otus's binary threshold"), plt.xticks([]), plt.yticks([])
#         plt.show()
#     except ValueError:
#         print("Bora para proxima.")

### bublessss####

# from math import sqrt
# from skimage import data
# from skimage.feature import blob_dog, blob_log, blob_doh
# from skimage.color import rgb2gray
# import cv2
#
# import matplotlib.pyplot as plt
#
#
# image = cv2.imread('color.jpeg')
# b,g,r = cv2.split(image)
# rgb_img = cv2.merge([r,g,b])
# gray = cv2.cvtColor(image,7)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
#
#
# image_gray = rgb2gray(thresh)
#
# blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
#
# # Compute radii in the 3rd column.
# blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
#
# blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
# blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
#
# blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)
#
# blobs_list = [blobs_log, blobs_dog, blobs_doh]
# colors = ['yellow', 'lime', 'red']
# titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
#           'Determinant of Hessian']
# sequence = zip(blobs_list, colors, titles)
#
# fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True,
#                          subplot_kw={'adjustable': 'box-forced'})
# ax = axes.ravel()
#
# for idx, (blobs, color, title) in enumerate(sequence):
#     ax[idx].set_title(title)
#     ax[idx].imshow(image, interpolation='nearest')
#     for blob in blobs:
#         y, x, r = blob
#         c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
#         ax[idx].add_patch(c)
#     ax[idx].set_axis_off()
#
# plt.tight_layout()
# plt.show()



# from skimage.filters import threshold_otsu
# from skimage.filters import threshold_mean
# import matplotlib.pyplot as plt
# import cv2
# import matplotlib.pyplot as plt
# from skimage import data
# from skimage import color
# from skimage import img_as_float
#
#
# image = cv2.imread('saco.jpeg')
#
#
# grayscale_image = img_as_float(image)
# image = color.gray2rgb(grayscale_image)
#
# red_multiplier = [1, 0, 0]
# yellow_multiplier = [1, 1, 0]
#
# fig, (ax1, ax2,ax3, ax4,ax5) = plt.subplots(ncols=5, figsize=(8, 4), sharex=True, sharey=True)
# ax1.imshow(red_multiplier * image)
# ax2.imshow(yellow_multiplier * image)
#
# #image = data.camera()
# thresh = threshold_mean(yellow_multiplier * image)
# # image = data.camera()
# thresh = threshold_mean(yellow_multiplier * image)
# binary = image > thresh
#
# fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
# ax = axes.ravel()
#
# ax[0].imshow(image, cmap=plt.cm.gray)
# ax[0].set_title('Original image')
#
# ax[1].imshow(binary, cmap=plt.cm.gray)
# ax[1].set_title('Result')
#
#
# for a in ax:
#     a.axis('off')
#
# plt.show()




# from skimage import data
# from skimage.color import rgb2hed
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib.pyplot as plt
# import cv2
#
# # Create an artificial color close to the orginal one
# cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
# cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
#                                              'saddlebrown'])
# cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',
#                                                'white'])
#
# ihc_rgb = cv2.imread('soja.jpeg')
# ihc_hed = rgb2hed(ihc_rgb)
#
# fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True,
#                          subplot_kw={'adjustable': 'box-forced'})
# ax = axes.ravel()
#
# ax[0].imshow(ihc_rgb)
# ax[0].set_title("Original image")
#
# ax[1].imshow(ihc_hed[:, :, 0], cmap=cmap_hema)
# ax[1].set_title("Hematoxylin")
#
# ax[2].imshow(ihc_hed[:, :, 1], cmap=cmap_eosin)
# ax[2].set_title("Eosin")
#
# ax[3].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)
# ax[3].set_title("DAB")
#
# for a in ax.ravel():
#     a.axis('off')
#
# fig.tight_layout()
#
# plt.show()



# from scipy import ndimage as ndi
# import matplotlib.pyplot as plt
#
# from skimage.morphology import watershed, disk
# from skimage import data
# from skimage.filters import rank
# from skimage.util import img_as_ubyte
#
# import cv2
#
# import urllib, cStringIO
#
# file = cStringIO.StringIO(urllib.urlopen(URL).read())
# image = Image.open(file)
#
#
#
# # denoise image
# denoised = rank.median(image, disk(2))
#
# # find continuous region (low gradient -
# # where less than 10 for this image) --> markers
# # disk(5) is used here to get a more smooth image
# markers = rank.gradient(denoised, disk(5)) < 10
# markers = ndi.label(markers)[0]
#
# # local gradient (disk(2) is used to keep edges thin)
# gradient = rank.gradient(denoised, disk(2))
#
# # process the watershed
# labels = watershed(gradient, markers)
#
# # display results
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
# ax = axes.ravel()
#
# ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
# ax[0].set_title("Original")
#
# ax[1].imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
# ax[1].set_title("Local Gradient")
#
# ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
# ax[2].set_title("Markers")
#
# ax[3].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
# ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
# ax[3].set_title("Segmented")
#
# for a in ax:
#     a.axis('off')
#
# fig.tight_layout()
# plt.show()


# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# from skimage import color
# from skimage import img_as_float
#
# img = cv2.imread('soja.jpeg')
# b,g,r = cv2.split(img)
# rgb_img = cv2.merge([r,g,b])
#
#
# ihc_hed = rgb2hed(rgb_img)
#
# fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True,
#                          subplot_kw={'adjustable': 'box-forced'})
#
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# plt.subplot(121),plt.imshow(red_multiplier * image)
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(thresh, 'gray')
# plt.title("Otus's binary threshold"), plt.xticks([]), plt.yticks([])
#
# # plt.subplot(123),plt.imshow(red_multiplier * image)
# # plt.title('red Image'), plt.xticks([]), plt.yticks([])
# # plt.subplot(124),plt.imshow(yellow_multiplier * image)
# # plt.title("yellow Image"), plt.xticks([]), plt.yticks([])
#
# plt.show()
#! /usr/bin/env python2
#
# import cv2
# import numpy as np
#
# colors = []
# first = False;
# lower_blue = np.array([33,186,250])
# upper_blue = np.array([28,193,241])
#
# def on_mouse_click (event, x, y, flags, frame):
#     if event == cv2.EVENT_LBUTTONUP:
#         colors.append(frame[y,x].tolist())
#         first = True
#
# def main():
#     capture = cv2.VideoCapture(0)
#
#     while True:
#         _, frame = capture.read()
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         if colors:
#             cv2.putText(frame, str(colors[-1]), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
#         cv2.imshow('frame', frame)
#         cv2.setMouseCallback('frame', on_mouse_click, frame)
#
#         lower_blue = np.array([33,186,250])
#         upper_blue = np.array([28,193,241])
#
#         # Threshold the HSV image to get only blue colors
#         mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
#         # Bitwise-AND mask and original image
#         res = cv2.bitwise_and(frame,frame, mask= mask)
#
#         cv2.imshow('mask',mask)
#         cv2.imshow('res',res)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     capture.release()
#     cv2.destroyAllWindows()
#
#     # avgb = int(sum(c[0] for c in colors) / len(colors))
#     # avgg = int(sum(c[0] for c in colors) / len(colors))
#     # avgr = int(sum(c[0] for c in colors) / len(colors))
#     # print avgb, avgg, avgr
#
#     minb = min(c[0] for c in colors)
#     ming = min(c[1] for c in colors)
#     minr = min(c[2] for c in colors)
#     maxb = max(c[0] for c in colors)
#     maxg = max(c[1] for c in colors)
#     maxr = max(c[2] for c in colors)
#     print minr, ming, minb, maxr, maxg, maxb
#
#     if first:
#         lb = [minb,ming,minr]
#         ub = [maxb,maxg,maxr]
#         lower_blue = np.array(lb)
#         upper_blue = np.array(ub)
#         first = False;
#
#     print lb, ub
#
#
#
#
#
#
# if __name__ == "__main__":
#     main()
#
#


import cv2
import numpy as np
import sys
import numpy as np
import cv2
from threading import Thread


class regular(Thread):
    """docstring for regular."""
    def __init__(self, num):
        Thread.__init__(self)
        self.num = num

    def run(self):
        while (1):
            hsf_select = raw_input("digita hsf: \n")
            min_value = raw_input("digita min: \n")
            min_max = raw_input("digita min: \n")


blue = sys.argv[1]
green = sys.argv[2]
red = sys.argv[3]

color = np.uint8([[[blue, green, red]]])
hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

hue = hsv_color[0][0][0]

print("Lower bound is :"),
print("[" + str(hue-10) + ", 100, 100]\n")

print("Upper bound is :"),
print("[" + str(hue + 10) + ", 255, 255]")
cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([hue-5,50,50])
    upper_blue = np.array([hue+5,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()



#! /usr/bin/env python2

# import cv2
# import numpy as np
#
# colors = []
#
# def on_mouse_click (event, x, y, flags, frame):
#     if event == cv2.EVENT_LBUTTONUP:
#         colors.append(frame[y,x].tolist())
#
# def main():
#     capture = cv2.VideoCapture(0)
#
#     while True:
#         _, frame = capture.read()
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
#         if colors:
#             cv2.putText(frame, str(colors[-1]), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
#         cv2.imshow('frame', frame)
#         cv2.setMouseCallback('frame', on_mouse_click, hsv)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     capture.release()
#     cv2.destroyAllWindows()
#
#     # avgb = int(sum(c[0] for c in colors) / len(colors))
#     # avgg = int(sum(c[0] for c in colors) / len(colors))
#     # avgr = int(sum(c[0] for c in colors) / len(colors))
#     # print avgb, avgg, avgr
#
#     minb = min(c[0] for c in colors)
#     ming = min(c[1] for c in colors)
#     minr = min(c[2] for c in colors)
#     maxb = max(c[0] for c in colors)
#     maxg = max(c[1] for c in colors)
#     maxr = max(c[2] for c in colors)
#     print minr, ming, minb, maxr, maxg, maxb
#
#     lb = [minb,ming,minr]
#     ub = [maxb,maxg,maxr]
#     print lb, ub
#
# if __name__ == "__main__":
#     main()
