import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import ndimage as ndi
import skimage
import matplotlib.patches as patches

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=14)

def set_ROI(file,x1,x2,y1,y2, show_ims=False):
    im = plt.imread(file)
    if show_ims == True:
        plt.subplot(211)
        plt.imshow(im)
        plt.subplot(212)
        plt.imshow(im[x1:x2, y1:y2])
        plt.show()
    return im[x1:x2, y1:y2]

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
    
def edge_image2scatter(edge_im,show_ims=True):
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

br_roi_x1, br_roi_x2,br_roi_y1,br_roi_y2 = 300,430,0,-1
br_zoom_x1, br_zoom_x2, br_zoom_y1, br_zoom_y2 = 800, 1400, 320, 470

cl_roi_x1, cl_roi_x2,cl_roi_y1,cl_roi_y2 = 700,880,0,-1
cl_zoom_x1, cl_zoom_x2, cl_zoom_y1, cl_zoom_y2 = 1800, 2500, 750, 920

br_im = plt.imread('images/br.tif') #  the scale bar is accross the 1464-1226= 238 pixels 500 nm
br_tr = set_ROI('images/br.tif',br_roi_x1, br_roi_x2,br_roi_y1,br_roi_y2) # trim the figures for a single interface
br_bin = binary_mask(br_tr)
br_edge = skimage.feature.canny(br_tr, sigma=1.65) # canny filter that marks the interface
br_x, br_y = edge_image2scatter(br_edge)
br_image, br_zoom_in = zoom_in(br_im, br_zoom_x1, br_zoom_x2, br_zoom_y1, br_zoom_y2)

cl_im = plt.imread('images/90_rt_cl_15-1.tif') #  the scale bar is accross the 2873-2393= 480 pixels 500 nm
cl_tr =  set_ROI('images/90_rt_cl_15-1.tif',cl_roi_x1, cl_roi_x2,cl_roi_y1,cl_roi_y2)
cl_bin = binary_mask(cl_tr)
cl_edge = skimage.feature.canny(cl_bin, sigma=2) # Canny filter detects edges.
cl_x, cl_y = edge_image2scatter(cl_edge)
cl_image, cl_zoom_in = zoom_in(cl_im, cl_zoom_x1, cl_zoom_x2, cl_zoom_y1, cl_zoom_y2)


fig = plt.figure(figsize=(7,5.5))
ax1 = fig.add_subplot(4,2,(1,3))
ax2 = fig.add_subplot(425)
ax3 = fig.add_subplot(4,2,(2,4))
ax4 = fig.add_subplot(426)
ax5 = fig.add_subplot(427)
ax6 = fig.add_subplot(428)

ax1.imshow(cl_image, aspect="auto")
rect2 = patches.Rectangle((2050, 1520), 480, 12, linewidth=1, edgecolor='w', facecolor='w')
ax1.add_patch(rect2)
ax1.text(2065, 1490, '500 nm',color='w')
ax1.axis([782,2618,1610,390])
ax1.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

ax2.imshow(cl_zoom_in, cmap='gray', aspect="auto")
print(np.shape(cl_zoom_in))
fit = np.polyfit(cl_x,cl_y,1)
ax2.plot([0,(cl_zoom_x2-cl_zoom_x1)],[fit[0]*0+fit[1],fit[0]*(cl_zoom_x2-cl_zoom_x1)+fit[1]],'m--',linewidth=1) # Drawing a straight line from two points x = 0 f(0) and x=3000 f(3000)
ax2.plot(cl_x[0:(cl_zoom_x2-cl_zoom_x1)],cl_y[0:(cl_zoom_x2-cl_zoom_x1)],'b--',linewidth=1)
# ax2.axis([1800, 1800+700, 750+170, 750])
ax2.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
                
                
yfit = np.polyval(fit,cl_x)
ax5.plot(cl_x/480*500,(yfit-cl_y)/480*500)
ax5.set_xlabel('Distance (nm)')
ax5.set_ylabel('Residual (nm)')
ax5.set_xlim([0, 3200])
ax5.set_ylim([30, -49])
ax5.vlines(x=[1800/480*500, (1800+700)/480*500], ymin=-60, ymax=35, colors='k', ls='--', linewidth=0.7)
gof = sum(map(abs, yfit-cl_y))/len(yfit) # The goodness of fit is calculated by the sum of absolute value of errors over the number of pixels
print(gof/480*500)


ax3.imshow(br_image, aspect="auto")
roi2 = [800, 1400, 470, 320]
rect2 = patches.Rectangle((1200, 960), 238, 12, linewidth=1, edgecolor='w', facecolor='w')
ax3.add_patch(rect2)
ax3.text(1150, 930, '500 nm',color='w')
ax3.axis([0,1500,1030,30])
ax3.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

ax4.imshow(br_im, cmap='gray', aspect="auto")
fit = np.polyfit(br_x,br_y+300.,1)
ax4.plot([0,3000],[fit[0]*0+fit[1],fit[0]*3000+fit[1]],'m--',linewidth=1)
ax4.plot(br_x,br_y+300 ,'b--',linewidth=0.8)
ax4.axis(roi2)
ax4.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
                
yfit = np.polyval(fit,br_x)
ax6.plot(br_x/238*500,(yfit-br_y-300)/238*500)
ax6.set_xlabel('Distance (nm)')
# ax6.set_ylabel('Residual (nm)')
ax6.set_xlim([0, 3200])
ax6.set_ylim([30, -49])
ax6.vlines(x=[800/238*500, (800+600)/238*500], ymin=-60, ymax=35, colors='k', ls='--', linewidth=0.7)
gof = sum(map(abs, yfit-br_y-300))/len(yfit) # The goodness of fit is calculated by the sum of absolute value of errors over the number of pixels
print(gof/238*500)
fig.subplots_adjust(hspace=0.5, right=0.95, top=0.96)
# plt.savefig(r'Output\example.svg')
plt.show()
