import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import ndimage as ndi
from skimage import feature
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
    return br_bin
    
def edge_image2scatter(edge_im,sho_ims=False):
    xy = np.column_stack(np.where(edge_im > 0)) # Get the x and y values of the interface for a scatter plot.
    x = xy[:,1]
    y = xy[:,0]
    unique_x = np.unique(x) # Remove values that are not single values of f(x) חד חד ערכי
    avg_y = np.array([np.mean(y[x == ux]) for ux in unique_x])
    return unique_x, avg_y
    

br_im = plt.imread('images/br.tif') #  the scale bar is accross the 1464-1226= 238 pixels 500 nm
cl_im = plt.imread('images/90_rt_cl_15-1.tif') #  the scale bar is accross the 2873-2393= 480 pixels 500 nm

br_tr = set_ROI('images/br.tif',300,430,0,-1) # trim the figures for a single interface
br_bin = binary_mask(br_tr)
br_edge = feature.canny(br_tr, sigma=1.65) # canny filter that marks the interface

cl_tr =  set_ROI('images/90_rt_cl_15-1.tif',700,880,0,-1)
cl_bin = binary_mask(cl_tr)
cl_edge = feature.canny(cl_bin, sigma=2) # Canny filter detects edges.

cl_x, cl_y = edge_image2scatter(cl_edge)

# cl_xy = np.column_stack(np.where(cl_edge > 0)) # Get the x and y values of the interface for a scatter plot.
# x = cl_xy[:,1]
# y = cl_xy[:,0]
# unique_x = np.unique(x) # Remove values that are not single values of f(x) חד חד ערכי
# avg_y = np.array([np.mean(y[x == ux]) for ux in unique_x])

fig = plt.figure(figsize=(7,5.5))
ax1 = fig.add_subplot(4,2,(1,3))
ax2 = fig.add_subplot(425)
ax3 = fig.add_subplot(4,2,(2,4))
ax4 = fig.add_subplot(426)
ax5 = fig.add_subplot(427)
ax6 = fig.add_subplot(428)

ax1.imshow(cl_im, cmap='gray', aspect="auto")
roi1 = [1800, 2500, 920, 750]
rect1 = patches.Rectangle((roi1[0], roi1[3]), roi1[1]-roi1[0], roi1[2]-roi1[3], linewidth=1, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((2050, 1520), 480, 12, linewidth=1, edgecolor='w', facecolor='w')
ax1.add_patch(rect1)
ax1.add_patch(rect2)
ax1.text(2065, 1490, '500 nm',color='w')
ax1.axis([782,2618,1610,390])
ax1.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

ax2.imshow(cl_im, cmap='gray', aspect="auto")
fit = np.polyfit(cl_x,cl_y+700.,1)
ax2.plot([0,3000],[fit[0]*0+fit[1],fit[0]*3000+fit[1]],'m--',linewidth=1) # Drawing a straight line from two points x = 0 f(0) and x=3000 f(3000)
ax2.plot(cl_x,cl_y+700. ,'b--',linewidth=1)
ax2.axis([1800, 1800+700, 750+170, 750])
ax2.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
                
                
yfit = np.polyval(fit,cl_x)
ax5.plot(cl_x/480*500,(yfit-cl_y-700)/480*500)
ax5.set_xlabel('Distance (nm)')
ax5.set_ylabel('Residual (nm)')
ax5.set_xlim([0, 3200])
ax5.set_ylim([30, -49])
ax5.vlines(x=[1800/480*500, (1800+700)/480*500], ymin=-60, ymax=35, colors='k', ls='--', linewidth=0.7)
gof = sum(map(abs, yfit-cl_y-700))/len(yfit) # The goodness of fit is calculated by the sum of absolute value of errors over the number of pixels
print(gof/480*500)

# ax5.set_xlim([1800, 1800+700])
###########################################
br_xy = np.column_stack(np.where(br_edge > 0)) # Get the x and y values of the interface for a scatter plot.
x = br_xy[:,1]
y = br_xy[:,0]
unique_x = np.unique(x) # Remove values that are not single values of f(x) חד חד ערכי
avg_y = np.array([np.mean(y[x == ux]) for ux in unique_x])

ax3.imshow(br_im, cmap='gray', aspect="auto")
roi2 = [800, 1400, 470, 320]
rect1 = patches.Rectangle((roi2[0], roi2[3]), roi2[1]-roi2[0], roi2[2]-roi2[3], linewidth=1, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((1200, 960), 238, 12, linewidth=1, edgecolor='w', facecolor='w')
ax3.add_patch(rect1)
ax3.add_patch(rect2)
ax3.text(1150, 930, '500 nm',color='w')
ax3.axis([0,1500,1030,30])
ax3.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

ax4.imshow(br_im, cmap='gray', aspect="auto")
fit = np.polyfit(unique_x,avg_y+300.,1)
ax4.plot([0,3000],[fit[0]*0+fit[1],fit[0]*3000+fit[1]],'m--',linewidth=1)
ax4.plot(unique_x,avg_y+300 ,'b--',linewidth=0.8)
ax4.axis(roi2)
ax4.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
                
yfit = np.polyval(fit,unique_x)
ax6.plot(unique_x/238*500,(yfit-avg_y-300)/238*500)
ax6.set_xlabel('Distance (nm)')
# ax6.set_ylabel('Residual (nm)')
ax6.set_xlim([0, 3200])
ax6.set_ylim([30, -49])
ax6.vlines(x=[800/238*500, (800+600)/238*500], ymin=-60, ymax=35, colors='k', ls='--', linewidth=0.7)
gof = sum(map(abs, yfit-avg_y-300))/len(yfit) # The goodness of fit is calculated by the sum of absolute value of errors over the number of pixels
print(gof/238*500)
fig.subplots_adjust(hspace=0.5, right=0.95, top=0.96)
# plt.tight_layout()
plt.savefig(r'Output\example.svg')
plt.show()