import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import ndimage as ndi
import skimage
import matplotlib.patches as patches

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=14)

def set_ROI(im,x1,x2,y1,y2, show_ims=False):
    im_roi = im[x1:x2, y1:y2]
    if show_ims == True:
        plt.subplot(211)
        plt.imshow(im)
        plt.subplot(212)
        plt.imshow(im[x1:x2, y1:y2])
        plt.show()
    return im_roi

def binary_mask(im, threshold=False, show_ims=False):
    if not threshold:
        threshold = np.average(im)
    br_bin = np.zeros(np.shape(im))
    br_bin[im>threshold] = 255
    if show_ims == True:
        plt.subplot(121)
        plt.imshow(im)
        plt.subplot(122)
        plt.imshow(br_bin)
        plt.show()
    return br_bin
    
def edge_image2scatter(edge_im,show_ims=False):
    xy = np.column_stack(np.where(edge_im > 0)) # Get the x and y values of the interface for a scatter plot.
    x = xy[:,1]
    y = xy[:,0]
    unique_x = np.unique(x) # Remove values that are not single values of f(x) חד חד ערכי
    avg_y = np.array([np.mean(y[x == ux]) for ux in unique_x])
    if show_ims==True:
        plt.plot(unique_x,avg_y)
    return unique_x, avg_y
    
def zoom_in(im,x1,x2,y1,y2,linewidth=8,show_ims=False):
    image = np.dstack([im,im,im]) # Convert a grayscale image to rgb
    lw = linewidth # This parameter set the linewidth of the zoomed in rectangle
    color = [255,0,0]
    image[y1-lw:y1,x1:x2] = color
    image[y2:y2+lw,x1:x2] = color
    image[y1:y2,x1-lw:x1] = color
    image[y1:y2,x2:x2+lw] = color
    zi_image = image[y1:y2,x1:x2]
    if show_ims == True:
        plt.subplot(211)
        plt.imshow(image)
        plt.subplot(212)
        plt.imshow(zi_image)
        plt.show()
    return image, zi_image

br_roi_x1, br_roi_x2,br_roi_y1,br_roi_y2 = 0, -1, 300, 430
br_zoom_x1, br_zoom_x2, br_zoom_y1, br_zoom_y2 = 800, 1400, 320, 470
cl_roi_x1, cl_roi_x2,cl_roi_y1,cl_roi_y2 = 0, -1, 700, 880
cl_zoom_x1, cl_zoom_x2, cl_zoom_y1, cl_zoom_y2 = 1800, 2500, 750, 920
files = ['images/br.tif', 'images/90_rt_cl_15-1.tif']

resplot_ylims = [-48, 32]

fig = plt.figure(figsize=(7,5.5))
for i,file in enumerate(files):
    if i == 0:
        roi_x1, roi_x2, roi_y1, roi_y2 = br_roi_x1, br_roi_x2,br_roi_y1,br_roi_y2
        zoom_x1, zoom_x2, zoom_y1, zoom_y2 = br_zoom_x1, br_zoom_x2, br_zoom_y1, br_zoom_y2
        s = 1.65
        rect2 = patches.Rectangle((1200, 960), 238, 12, linewidth=1, edgecolor='w', facecolor='w')
        axis = [0,1500,1030,30]
        text_xy = [1150, 930]
        pix2nm = 500/238
    if i == 1:
        roi_x1, roi_x2, roi_y1, roi_y2 = cl_roi_x1, cl_roi_x2,cl_roi_y1,cl_roi_y2
        zoom_x1, zoom_x2, zoom_y1, zoom_y2 = cl_zoom_x1, cl_zoom_x2, cl_zoom_y1, cl_zoom_y2
        s = 3
        rect2 = patches.Rectangle((2050, 1520), 480, 12, linewidth=1, edgecolor='w', facecolor='w')
        axis = [782,2618,1610,390]
        text_xy = [2065, 1490]
        pix2nm = 500/480
    
    im = plt.imread(file) #  the scale bar is accross the 1464-1226= 238 pixels 500 nm
    tr = set_ROI(im, roi_y1, roi_y2,roi_x1, roi_x2) # trim the figures for a single interface
    edge = skimage.feature.canny(tr, sigma=s) # canny filter that marks the interface
    x, y = edge_image2scatter(edge)
    image, zi = zoom_in(im, zoom_x1, zoom_x2, zoom_y1, zoom_y2)


    ax1 = fig.add_subplot(4,2,(1+i,3+i))

    ax1.imshow(image, aspect="auto")
    # ax1.plot(x,y+roi_y1) This is a sanity check. It works
    ax1.add_patch(rect2)
    ax1.text(text_xy[0],text_xy[1], '500 nm',color='w')
    ax1.axis(axis)
    ax1.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 

    ax2 = fig.add_subplot(4,2,5+i)
    
    ax2.imshow(zi, cmap='gray', aspect="auto")
    fit = np.polyfit(x,y,1)
    len_zi = np.shape(zi)[1]-1 # This is the index of the last pixel lengthwise of the zoomed in (zi) image
    roi_zoom_offset = roi_y1-zoom_y1
    ax2.plot([0,len_zi],
    [fit[0]*(zoom_x1)+fit[1]+roi_zoom_offset,fit[0]*(zoom_x2-1)+fit[1]+roi_zoom_offset],'m--',linewidth=1) # Drawing a straight line from two points x = 0 f(0) and x=3000 f(3000)
    
    ### The offset here is arbitrary just to make it fit. I don't know how to solve it
    ax2.plot(range(len_zi),y[zoom_x1-12 : zoom_x2-13]+roi_zoom_offset,'b--',linewidth=1) # I cant figure out why there is a missmatch between the line and image

    ax2.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    
    ax3 = fig.add_subplot(4,2,7+i)    
    yfit = np.polyval(fit,x)
    x_nm = x*pix2nm
    y_res_nm = (yfit-y)*pix2nm
    ax3.plot(x_nm,y_res_nm)
    ax3.set_xlabel('Distance (nm)')
    if i == 0:
        ax3.set_ylabel('Residual (nm)')
    ax3.set_xlim([axis[0]*pix2nm, axis[1]*pix2nm])
    ax3.set_ylim(resplot_ylims)
    ax3.vlines(x=[zoom_x1*pix2nm, zoom_x2*pix2nm], ymin=resplot_ylims[0], ymax=resplot_ylims[1], colors='k', ls='--', linewidth=0.7)
    gof = sum(map(abs, y_res_nm))/len(yfit) # The goodness of fit is calculated by the sum of absolute value of errors over the number of pixels
    print(gof)

fig.subplots_adjust(hspace=0.5, right=0.95, top=0.96)
plt.savefig(r'Output\example.svg')
plt.show()
