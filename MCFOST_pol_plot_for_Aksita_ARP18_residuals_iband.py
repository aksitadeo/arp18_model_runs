#!/usr/bin/env python
# coding: utf-8

# # MCFOST polarimetry

# In[1]:


# !pip install opencv-python


# In[2]:


# !pip install scikit-image


# In[3]:


import os
from astropy.io import fits
import astropy.units as u
import numpy as np
from IPython.display import display
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import cv2

plt.rc('font',   size=14)          # controls default text sizes
plt.rc('axes',   titlesize=16)     # fontsize of the axes title
plt.rc('axes',   labelsize=16)     # fontsize of the x and y labels
plt.rc('xtick',  labelsize=12)     # fontsize of the tick labels
plt.rc('ytick',  labelsize=12)     # fontsize of the tick labels
plt.rc('legend', fontsize=14)      # legend fontsize
plt.rc('figure', titlesize=16)     # fontsize of the figure title


# In[4]:


def load_obs_images(main_dir):
    dir = main_dir
    hdul = fits.open(dir)
    data = hdul[0].data
    n = data.shape[0]

    return data, n

def load_mcfost_images(main_dir, ploting=False, save_plots=False, title_addition=''):
    #loop over all folders made for individual imaging runs, excluding the thermal
    #structure folder with prefix 'th'. Also unzip any zips
    img_folders=glob.glob(main_dir+'data_[!th]*')
    for folderpath in img_folders:
        if os.path.exists(folderpath+'/RT.fits.gz') and not os.path.exists(folderpath+'/RT.fits'):
            #os.system("gunzip -k "+folderpath+'/RT.fits.gz')
            subprocess.run(
                ["gunzip", "-k", f"{folderpath}/RT.fits.gz"],
                stdout=subprocess.DEVNULL
            )
        #open the required fits file + get some header info
        hdul=fits.open(folderpath+'/RT.fits')
        header=hdul[0].header
        wave=hdul[0].header['WAVE']
        #load in all images and separate
        img_array=hdul[0].data
        #total intensity
        img_tot=img_array[:][:][0][0][0]
        #stokes Q intensity
        img_q=img_array[:][:][1][0][0]
        #stokes U intensity
        img_u=img_array[:][:][2][0][0]
        #stokes V intensity
        img_v=img_array[:][:][3][0][0]
        #direct starlight intensity
        img_star=img_array[:][:][4][0][0]
        #scattered starlight intensity
        img_star_sct=img_array[:][:][5][0][0]
        #disk thermal intensity
        img_disk_th=img_array[:][:][6][0][0]
        #disk scattered thermal intensity
        img_disk_th_sct=img_array[:][:][7][0][0]

        if ploting==True:
            #do some plotting
            fig, ax = plt.subplots(2, 4, figsize=(14,7))
            color_map = 'afmhot'
            ax[0][0].imshow(img_tot, color_map)
            ax[0][0].set_title('$I_{tot}$')
            ax[0][1].imshow(img_q, color_map)
            ax[0][1].set_title('$Q$')
            ax[0][2].imshow(img_u, color_map)
            ax[0][2].set_title('$U$')
            ax[0][3].imshow(img_v, color_map)
            ax[0][3].set_title('$V$')
            ax[1][0].imshow(img_tot-img_star, color_map)
            ax[1][0].set_title('$I_{disk}$')
            ax[1][1].imshow(img_disk_th, color_map)
            ax[1][1].set_title('$I_{disk,th}$')
            ax[1][2].imshow(img_star_sct, color_map)
            ax[1][2].set_title('$I_{disk,scat,*}$')
            ax[1][3].imshow(img_disk_th_sct, color_map)
            ax[1][3].set_title('$I_{disk,scat,th}$')
            plt.suptitle(str(wave)+r' $\mu m$, '+title_addition)
            plt.tight_layout()
            plt.show()
            #save the plots
            if save_plots==True:
                fig.savefig(save_plots+title_addition+' image_decomp_'+str(wave)+'.png', dpi= 150, bbox_inches='tight')
            plt.close()
    return img_array, img_tot, img_q, img_u, img_v, img_star, img_star_sct, img_disk_th, img_disk_th_sct

def load_mcfost_images_1wave(
        main_dir: str,
        wavelength: str or float,
        *,
        ploting: bool= False,
        save_plots: bool or str = False,
        title_addition:str = ''
) -> tuple[np.ndarray, fits.Header, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and optionally visualize MCFOST model images at a single wavelength.

        This function:
        * Locates the folder corresponding to a given wavelength (``main_dir/data_<wavelength>``),
        * Ensures the FITS file is uncompressed (unzips ``RT.fits.gz`` if only that exists),
        * Loads the multi-extension FITS file produced by MCFOST,
        * Extracts key image components: total intensity, Stokes parameters (Q, U, V),
            direct stellar emission, scattered stellar emission, thermal emission, and scattered
            thermal emission,
        * Optionally produces a quick-look decomposition plot.

        Parameters
        ----------
        main_dir : str
            Path to the parent directory containing the MCFOST run output.
        wavelength : str or float
            Wavelength identifier used by MCFOST in the subdirectory name (e.g., ``1.65`` or ``1650``).
        ploting : bool, default=False
            If True, generate a figure showing total intensity, Stokes components, and decomposed images.
        save_plots : bool or str, default=False
            If False, plots are not saved. If a string (directory path), plots are saved into this directory
            with filenames including the wavelength and `title_addition`.
        title_addition : str, optional
            Additional string to append to the figure title and saved filename.

        Returns
        -------
        img_array : np.ndarray
            Raw 5D image data array from the FITS file (contains all components).
        header_data : astropy.io.fits.Header
            Header from the FITS file containing metadata (including wavelength).
        img_tot : np.ndarray
            Total intensity image (I).
        img_q : np.ndarray
            Stokes Q image.
        img_u : np.ndarray
            Stokes U image.
        img_v : np.ndarray
            Stokes V image.
        img_star : np.ndarray
            Direct stellar light image.
        img_star_sct : np.ndarray
            Stellar scattered light image.
        img_disk_th : np.ndarray
            Disk thermal emission image.
        img_disk_th_sct : np.ndarray
            Scattered thermal emission image.

        Notes
        -----
        * The function assumes that the MCFOST output directory structure is of the form:
        ``<main_dir>/data_<wavelength>/RT.fits(.gz)``.
        * Image arrays are returned in pixel coordinates, not rescaled to angular size.
        * The optional plots use a fixed colormap ("afmhot") and a 2×4 grid layout.
        """
        folderpath=main_dir+'/data_'+str(wavelength)

        print(folderpath)

        if os.path.exists(folderpath+'/RT.fits.gz')and not os.path.exists(folderpath+'/RT.fits'):
            #os.system("gunzip -k "+folderpath+'/RT.fits.gz')
            subprocess.run(
                    ["gunzip", "-k", f"{folderpath}/RT.fits.gz"],
                    stdout=subprocess.DEVNULL
                )
        #open the required fits file + get some header info
        hdul=fits.open(folderpath+'/RT.fits')
        header_data=hdul[0].header

        wave=hdul[0].header['WAVE']
        #load in all images and separate
        img_array=hdul[0].data
        #total intensity
        img_tot=img_array[:][:][0][0][0]
        #stokes Q intensity
        img_q=img_array[:][:][1][0][0]
        #stokes U intensity
        img_u=img_array[:][:][2][0][0]
        #stokes V intensity
        img_v=img_array[:][:][3][0][0]
        #direct starlight intensity
        img_star=img_array[:][:][4][0][0]
        #scattered starlight intensity
        img_star_sct=img_array[:][:][5][0][0]
        #disk thermal intensity
        img_disk_th=img_array[:][:][6][0][0]
        #disk scattered thermal intensity
        img_disk_th_sct=img_array[:][:][7][0][0]

        if ploting==True:
            #do some plotting
            fig, ax = plt.subplots(2, 4, figsize=(14,7))
            color_map = 'viridis' #'afmhot'
            ax[0][0].imshow(img_tot, color_map, extent=[+img_tot.shape[0]/2, -img_tot.shape[0]/2, -img_tot.shape[1]/2, img_tot.shape[1]/2])
            ax[0][0].set_title('$I_{tot}$')
            ax[0][1].imshow(img_q, color_map)
            ax[0][1].set_title('$Q$')
            ax[0][2].imshow(img_u, color_map)
            ax[0][2].set_title('$U$')
            ax[0][3].imshow(img_v, color_map)
            ax[0][3].set_title('$V$')
            ax[1][0].imshow(img_tot-img_star, color_map,extent=[+img_tot.shape[0]/2, -img_tot.shape[0]/2, -img_tot.shape[1]/2, img_tot.shape[1]/2])
            ax[1][0].set_title('$I_{disk}$')
            #ax[1][0].set_xlim([-img_tot.shape[0]/6, img_tot.shape[0]/6])
            #ax[1][0].set_ylim([-img_tot.shape[1]/6, img_tot.shape[1]/6])
            ax[1][1].imshow(img_disk_th, color_map)
            ax[1][1].set_title('$I_{disk,th}$')
            ax[1][2].imshow(img_star_sct, color_map)
            ax[1][2].set_title('$I_{disk,scat,*}$')
            ax[1][3].imshow(img_disk_th_sct, color_map)
            ax[1][3].set_title('$I_{disk,scat,th}$')
            plt.suptitle(str(wave)+r' $\mu m$, '+title_addition)
            plt.tight_layout()
            plt.show()
            #save the plots
            if save_plots:
                fig.savefig(save_plots+title_addition+' image_decomp_'+str(wave)+'.png', dpi= 150, bbox_inches='tight')
            plt.close()
        return img_array, header_data, img_tot, img_q, img_u, img_v, img_star, img_star_sct, img_disk_th, img_disk_th_sct


# # start here

# In[5]:


from pathlib import Path
import sys, json


# In[6]:


mod_dir = Path(sys.argv[1])
mod_dir=str(mod_dir)


# mod_dir = "/Users/aksitadeo/mcfost_outputs/OZstar/ar_pup18/alpha=0.1/h0=0.93/fl=0.8"#zones_test/surface_density_exp/"#surface_density_exp #tappered_edge. #image_size_test
folders = [name for name in os.listdir(mod_dir) if os.path.isdir(os.path.join(mod_dir, name)) and name != "figures" and name != "old"]

# print(mod_dir+folder)

fig_dir = mod_dir+"/figures_residuals_map/"
os.makedirs(fig_dir, exist_ok=True)  # no error if it already exists
for folder in folders:
    img_array, header, img_tot, img_q, img_u, img_v, img_star, img_star_sct, img_disk_th, img_disk_th_sct = load_mcfost_images_1wave(mod_dir,'0.8168',ploting=True, save_plots=fig_dir, title_addition=folder)


# In[7]:


import numpy as np

def radial_flux(image, center=None, dr=1):
    """
    Compute the total flux in concentric annuli (rings) of a 2D image.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image (e.g. intensity map).
    center : tuple (x, y), optional
        Center coordinates (x0, y0). Defaults to the image center.
    dr : float, optional
        Radial bin width in pixels. Default is 1.

    Returns
    -------
    r_bins : np.ndarray
        Radii (pixel units) corresponding to each annulus center.
    flux_in_ring : np.ndarray
        Total flux (sum of pixel values) in each annulus.
    """
    y, x = np.indices(image.shape)
    if center is None:
        center = (image.shape[1] / 2, image.shape[0] / 2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    r_max = np.max(r)
    bins = np.arange(0, r_max + dr, dr)
    flux_in_ring = np.zeros(len(bins) - 1)

    for i in range(len(bins) - 1):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if np.any(mask):
            flux_in_ring[i] = np.sum(image[mask])
        else:
            flux_in_ring[i] = np.nan

    r_bins = 0.5 * (bins[1:] + bins[:-1])
    return r_bins, flux_in_ring


# In[8]:


# Compute the radial profile of total intensity
# r, I_r = radial_flux(img_tot, center=None, dr=1)
#
# # Optionally plot the profile
# plt.figure()
# plt.plot(r, I_r, label='Normalised $I_{tot}$') # / np.max(I_r)
# plt.xlabel('Radius [pixels]')
# plt.ylabel('Normalised Intensity')
# plt.title(f'Radial Profile - heating')
# plt.legend()
# plt.grid(True)
# plt.savefig('/Users/aksitadeo/mcfost_outputs/ARPup_2018/data_0.8168/TRUE_non_norm_profile')
# plt.show()


# In[9]:


# for folder in folders:
#     img_array,header, img_tot, img_q, img_u, img_v, img_star, img_star_sct, img_disk_th, img_disk_th_sct = load_mcfost_images_1wave(mod_dir, '0.8173',ploting=False, title_addition=folder)
#
#
#     pi_sum = np.sum(np.sqrt(img_q**2 + img_u**2))
#     pi_frac = pi_sum/np.sum(img_tot)
#     print(folder, pi_frac*100)  # print polarization info


# In[10]:


folder =folders[2]
#folder='tapereddisc_surfdenexpin_-1_-gamma_exp_-1d25_dust_MW89'

print(folder)
img_array, header, img_tot, img_q, img_u, img_v, img_star, img_star_sct, img_disk_th, img_disk_th_sct = load_mcfost_images_1wave(mod_dir, '0.8168',ploting=False, title_addition=folder)

pi_sum = np.sum(np.sqrt(img_q**2 + img_u**2))
pi_frac = pi_sum/np.sum(img_tot)
print(img_q.shape)
pixel_scale=header['CDELT2']*u.deg.to(u.arcsec)*1000

print(pixel_scale)


# In[11]:


from skimage.transform import rescale, resize, downscale_local_mean
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft, AiryDisk2DKernel
import fnmatch

def plotImage(image1, lim1,ps,savepath=None, title=''):
    n = image1.shape[0]

    fig, ax = plt.subplots()
    image = image1#np.arcsinh(image1)#np.copy(image1)#
    if ps==12.27:
        lim=lim1*1.0
    else:
        lim=lim1*12.27/3.6
    max = np.max(image[int(n/2-2*lim/2):int(n/2+2*lim/2),int(n/2-2*lim/2):int(n/2+2*lim/2)])
    min=np.min(image[int(n/2-2*lim/2):int(n/2+2*lim/2),int(n/2-2*lim/2):int(n/2+2*lim/2)])
    d = n * ps / 2
    #plt.imshow(image, extent=(-d, d, d, -d))
    plt.imshow(image,vmin=min, vmax=max, extent=(-d, d, d, -d))
    plt.xlim(-lim * ps, lim * ps)
    plt.ylim(-lim * ps, lim * ps)
    if title != '':
        plt.title(title)
    plt.xlabel('mas')
    plt.ylabel("mas")
    plt.colorbar()
    plt.tight_layout(pad=3.0)

    # if savepath:
    #     plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

def synthetic_psf(ps, psf_FWHM):
    # ps in mas/pixel
    # psf_FWHM in mas
    sigma = psf_FWHM / (ps * 2*np.sqrt(2*np.log(2)))
    psf = Gaussian2DKernel(sigma,x_size=int(15*sigma),y_size=int(15*sigma))
    return psf

def Loadimage(dirdat,filename):
    dir =dirdat
    psfile =  filename
    files = os.listdir(dir)
    for file in files:
        if fnmatch.fnmatch(file, psfile):
            hdulPSF = fits.open(dir + file)
            fit = hdulPSF[0].data


    return fit

band='I'

if band=='I':
    ps=3.6
    psf_FWHM=30 #mas
elif band=='V':
    ps=3.6
    psf_FWHM=30 #mas
elif band=='H':
    ps=12.27
    psf_FWHM=40

print('Simulation MCFOST')
n_model=img_q.shape[0]
shift=int((n_model)/2.)
x = np.linspace(-shift+1, shift, num=n_model)
y = np.linspace(-shift+1, shift, num=n_model)
if n_model%2==0:
    x=x-0.5
    y=y-0.5
X, Y = np.meshgrid(x, y)
R_mcfost = np.sqrt(X**2 + Y**2)
#calculating angle for azimuthal polarisation
phi_mcfost= np.arctan(Y/X)+np.pi/2

pi= np.sqrt(img_q**2 + img_u**2)
pi_sum = np.sum(pi)
pi_frac = pi_sum/np.sum(img_tot)
q_phi=-img_q*np.cos(2*phi_mcfost)-img_u*np.sin(2*phi_mcfost)
u_phi=img_q*np.sin(2*phi_mcfost)-img_u*np.cos(2*phi_mcfost)

print(f'sum Q_phi: {np.sum(q_phi)}, sum U_phi: {np.sum(u_phi)}, sum PI: {pi_sum}')
print(f'frac Q_phi:{np.sum(q_phi)/np.sum(img_tot)*100}, frac U_phi:{np.sum(u_phi)/np.sum(img_tot)*100}, frac PI: {pi_frac*100}')
plotImage(img_q, 10*ps/pixel_scale, pixel_scale,fig_dir+'/Qmcfost.png', title='Q MCFOST')
plotImage(q_phi, 10*ps/pixel_scale, pixel_scale,fig_dir+'/Qphimcfost.png', title='Q_phi MCFOST')

print('Rescaling to instrument pixel scale')
img_q_rescaled=rescale(img_q, pixel_scale/ps, anti_aliasing=True)
img_u_rescaled=rescale(img_u, pixel_scale/ps, anti_aliasing=True)
img_total_rescaled=rescale(img_tot, pixel_scale/ps, anti_aliasing=True)
pi_rescaled=np.sqrt(img_q_rescaled**2+img_u_rescaled**2)
pi_rescaled_sum = np.sum(pi_rescaled)
pi_rescaled_frac = pi_rescaled_sum/np.sum(img_total_rescaled)
print(f'new shape:{img_q_rescaled.shape}, 1/coefficient: {ps/pixel_scale}')


n_model=img_q_rescaled.shape[0]
shift=int((n_model)/2.)
x = np.linspace(-shift+1, shift, num=n_model)
y = np.linspace(-shift+1, shift, num=n_model)
if n_model%2==0:
    x=x-0.5
    y=y-0.5
X, Y = np.meshgrid(x, y)
R_rescaled = np.sqrt(X**2 + Y**2)
#calculating angle for azimuthal polarisation
phi_rescaled= np.arctan(Y/X)+np.pi/2

q_phi_rescaled=-img_q_rescaled*np.cos(2*phi_rescaled)-img_u_rescaled*np.sin(2*phi_rescaled)
u_phi_rescaled=img_q_rescaled*np.sin(2*phi_rescaled)-img_u_rescaled*np.cos(2*phi_rescaled)


print(f'sum Q_phi: {np.sum(q_phi_rescaled)}, sum U_phi: {np.sum(u_phi_rescaled)}, sum PI: {pi_rescaled_sum}, sum I: {np.sum(img_total_rescaled)}')
print(f'frac Q_phi:{np.sum(q_phi_rescaled)/np.sum(img_total_rescaled)*100}, frac U_phi:{np.sum(u_phi_rescaled)/np.sum(img_total_rescaled)*100}, frac PI: {pi_rescaled_frac*100}')

plotImage(img_q_rescaled, 10, ps,fig_dir+'/Qrescaledmcfost.png',title='Q Rescaled')
plotImage(q_phi_rescaled, 10, ps,fig_dir+'/Qphirescaledmcfost.png', title='Q_phi Rescaled')


print('Convolving with synthetic PSF')

print(f'PSF FWHM: {psf_FWHM} mas')


psf = synthetic_psf(ps, psf_FWHM)
print(f'PSF shape: {psf.array.shape}, PSF sum: {np.sum(psf.array)}')
Q_conv=convolve_fft(img_q_rescaled, psf, boundary='wrap')
U_conv=convolve_fft(img_u_rescaled, psf, boundary='wrap')
img_tot_conv=convolve_fft(img_total_rescaled, psf, boundary='wrap')
pi_conv=np.sqrt(Q_conv**2+U_conv**2)
pi_conv_sum = np.sum(pi_conv)
pi_conv_frac = pi_conv_sum/np.sum(img_tot_conv)
print(f'new shape after convolution:{Q_conv.shape}')


Q_phi_conv=-Q_conv*np.cos(2*phi_rescaled)-U_conv*np.sin(2*phi_rescaled)
U_phi_conv=Q_conv*np.sin(2*phi_rescaled)-U_conv*np.cos(2*phi_rescaled)
print(f'sum Q_phi: {np.sum(Q_phi_conv)}, sum U_phi: {np.sum(U_phi_conv)}, sum PI: {pi_conv_sum}')
print(f'frac Q_phi:{np.sum(Q_phi_conv)/np.sum(img_tot_conv)*100} %, frac U_phi:{np.sum(U_phi_conv)/np.sum(img_tot_conv)*100} %, frac PI: {pi_conv_frac*100} %')
plotImage(Q_conv, 10, ps,fig_dir+'/Qconvmcfost.png', title='Q Conv')
plotImage(Q_phi_conv, 10, ps,fig_dir+'/Qphiconvmcfost.png', title='Q_phi Conv')
plotImage(img_tot_conv, 10, ps,fig_dir+'/Iconvmcfost.png', title='I Conv')
plotImage(pi_conv, 10, ps,fig_dir+'/PIconvmcfost.png', title='PI Conv')


print('Convolving with real PSF from observations')
# ARPup
star_psf='REF_HD75885'
figfolder_psf='/Users/aksitadeo/PycharmProjects/PythonProject/SPHERE_data/REF_HD75885_old/filtered/'

psf_full=Loadimage(figfolder_psf,star_psf+'_'+'I'+'_'+'I'+'_meancombined.fits')                       # write output
n=psf_full.shape[0]
psf_cut=100
psf=psf_full[int(n/2)-psf_cut:int(n/2)+psf_cut,int(n/2)-psf_cut:int(n/2)+psf_cut]
psf=psf/np.sum(psf)
print(f'PSF shape: {psf.shape}, PSF sum: {np.sum(psf)}')

Q_conv=convolve_fft(img_q_rescaled, psf, boundary='wrap')
U_conv=convolve_fft(img_u_rescaled, psf, boundary='wrap')
img_tot_conv=convolve_fft(img_total_rescaled, psf, boundary='wrap')
pi_conv=np.sqrt(Q_conv**2+U_conv**2)
pi_conv_sum = np.sum(pi_conv)
pi_conv_frac = pi_conv_sum/np.sum(img_tot_conv)
print(f'new shape after convolution:{Q_conv.shape}')


Q_phi_conv=-Q_conv*np.cos(2*phi_rescaled)-U_conv*np.sin(2*phi_rescaled)
U_phi_conv=Q_conv*np.sin(2*phi_rescaled)-U_conv*np.cos(2*phi_rescaled)
print(f'sum Q_phi: {np.sum(Q_phi_conv)}, sum U_phi: {np.sum(U_phi_conv)}, sum PI: {pi_conv_sum}')
print(f'frac Q_phi:{np.sum(Q_phi_conv)/np.sum(img_tot_conv)*100} %, frac U_phi:{np.sum(U_phi_conv)/np.sum(img_tot_conv)*100} %, frac PI: {pi_conv_frac*100} %')
plotImage(Q_conv, 10, ps,fig_dir+'/Qconv2mcfost.png', title='Q Conv')
plotImage(Q_phi_conv, 10,ps, fig_dir+'/Qphiconv2mcfost.png', title='Q_phi Conv')
plotImage(img_tot_conv, 10, ps,fig_dir+'/Iconv2mcfost.png', title='I Conv')
plotImage(pi_conv, 10, ps,fig_dir+'/PIconv2mcfost.png', title='PI Conv')


# In[12]:


from astropy.io import fits
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from scipy import interpolate
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from textwrap import wrap
import scipy.ndimage as ndimage
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import pandas as pd
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties

def gaus(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))


def find_FWHM (PSF,n):             #resolution
    middle=int(n/2)

    y1=PSF[middle,:]
    y2=PSF[:,middle]

    xdata = np.linspace(0,n, num=len(y1))


    n_gauss = len(xdata) #the number of data
    amp=np.max(y1)
    mean = np.sum(xdata * y1) / sum(y1)
    sigma = np.sqrt(sum(y1 * (xdata - mean)**2) / sum(y1))

    popt1,pcov1 = curve_fit(gaus,xdata,y1,p0=[amp,mean,sigma])
    popt2,pcov2 = curve_fit(gaus,xdata,y2,p0=[amp,mean,sigma])

    fwhm1=2*np.sqrt(2*math.log(2))*popt1[2]
    fwhm2=2.355*popt2[2]


    fwhm=(abs(fwhm1)+abs(fwhm2))/2

    return fwhm



def calculate_unresolved(correction_radius, q, u,i,ps,R,normlim):
    # Calculates degree and angle of unresolved polarisation
    #resulting values are in fraction (not %) for dolp, and in degrees for aolp


    mask=(R<=correction_radius)

    normalisation=np.sum(i[R<=1500/ps])
    q_over_i=np.divide(q,i,where=i!=0)
    cq=np.median(q_over_i[mask]) #for median normal as in IRDIS
    u_over_i=np.divide(u,i,where=i!=0)
    cu=np.median(u_over_i[mask]) #for median normal as in IRDIS
    aolp_unres=np.rad2deg(0.5*np.arctan2(cu, cq))
    #print(aolp_unres)
    if aolp_unres<0 :
        aolp_unres=aolp_unres+180
    dolp_unres=np.sum(np.sqrt(cu*i*cu*i+ cq*i*cq*i)*(R<=normlim))/normalisation

    q_corr=q-cq*i
    u_corr=u-cu*i
    return dolp_unres, aolp_unres,q,u

print(find_FWHM(img_tot, img_tot.shape[0])*pixel_scale)
print(find_FWHM(img_total_rescaled, img_total_rescaled.shape[0])*ps)
print(find_FWHM(img_tot_conv, img_tot_conv.shape[0])*ps)

plotImage(img_tot, 10*ps/pixel_scale, pixel_scale, title='I tot')
plotImage(img_total_rescaled, 10, ps, title='I tot rescaled')
plotImage(img_tot_conv, 10, ps, title='I tot conv')

correction_radius=3
dolp_unres, aolp_unres,q_corr,u_corr=calculate_unresolved(correction_radius, img_q_rescaled, img_u_rescaled,img_total_rescaled,pixel_scale,R_rescaled,100)
q_phi_corr=-q_corr*np.cos(2*phi_rescaled)-u_corr*np.sin(2*phi_rescaled)
u_phi_corr=q_corr*np.sin(2*phi_rescaled)+u_corr*np.cos(2*phi_rescaled)
pi_corr=np.sqrt(q_corr*q_corr+u_corr*u_corr)
aolp_corr=0.5*np.arctan2(u_corr, q_corr)

print(f'Unresolved pol: {dolp_unres*100} %, angle: {aolp_unres} deg')

dolp_unres_conv, aolp_unres_conv,q_corr_conv,u_corr_conv=calculate_unresolved(correction_radius, Q_conv, U_conv,img_tot_conv,ps,R_rescaled,100)
q_phi_corr_conv=-q_corr_conv*np.cos(2*phi_rescaled)-u_corr_conv*np.sin(2*phi_rescaled)
u_phi_corr_conv=q_corr_conv*np.sin(2*phi_rescaled)+u_corr_conv*np.cos(2*phi_rescaled)
pi_corr_conv=np.sqrt(q_corr_conv*q_corr_conv+u_corr_conv*u_corr_conv)
aolp_corr_conv=0.5*np.arctan2(u_corr_conv, q_corr_conv)
print(f'Unresolved pol after conv: {dolp_unres_conv*100} %, angle: {aolp_unres_conv} deg')



# #### 4 figure

# In[13]:


n=img_q_rescaled.shape[0]
d = (n-1) * ps / 2
lim=30

fig, axs = plt.subplots(2, 4, figsize=(12, 6))
# images_list=[q_phi_rescaled, pi_rescaled, q_phi_corr,pi_corr,Q_phi_conv, pi_conv, q_phi_corr_conv,pi_corr_conv, decQphi, decPI, unresQphi, unresPI]
images_list=[q_phi_rescaled, pi_rescaled, q_phi_corr,pi_corr,Q_phi_conv, pi_conv, q_phi_corr_conv,pi_corr_conv]

# bandlist=['V','V','V','V','V','V','V','V']
bandlist=['I','I','I','I','I','I','I','I','I','I','I','I']
bandlist=['I','I','I','I','I','I','I','I']

i_im=0
for ax, image,band in zip(axs.flat,images_list,bandlist):
    image = np.arcsinh(image)
    max = np.max(image[int(n/2-lim/2):int(n/2+lim/2),int(n/2-lim/2):int(n/2+lim/2)])
    min=np.min(image[int(n/2-lim/2):int(n/2+lim/2),int(n/2-lim/2):int(n/2+lim/2)])


    ax.imshow(image, vmin=min, vmax=max, extent=(-d, d, d, -d))
    #plt.plot(0, 0, "+", color="red")
    ax.set_xlim(-lim * ps, lim * ps)
    ax.set_ylim(-lim * ps, lim * ps)
    ax.set_xlabel('mas',fontsize=14)
    ax.set_ylabel('mas',fontsize=14)
    #ax.set_yticks([-130,-100,-50,0,50,100])
    #ax.set_yticks(fontsize=14)
    ax.tick_params(axis='both',labelsize=14)

    i_im+=1
col_titles = ['Q$_\phi$', 'I$_{\mathrm{pol}}$', 'Q$_\phi$', 'I$_{\mathrm{pol}}$']


for ax, col_title in zip(axs[0], col_titles):
    ax.set_title(col_title, fontsize=16)


for i in range(3):
    axs[0,i+1].axis('off')
    axs[1,i+1].get_yaxis().set_visible(False)
axs[0,0].get_xaxis().set_visible(False)

plt.tight_layout()
#fig.text(0.5, 1.05, starnames[star], fontsize=16, ha='center')

fig.text(0.31, 1, 'With unresolved', fontsize=16, ha='center')
fig.text(0.76, 1, "without unresolved", fontsize=16, ha='center')


# plt.savefig(fig_dir+'unresolved.png',bbox_inches='tight')
plt.show()
plt.close()


# In[14]:


plt.imshow(pi_corr_conv)
plt.xlim(40, 120)
plt.ylim(40, 120)


# In[15]:


def plot_AoLP(ps,Q,U,R,I,Q_PHI,title,save,plot,noise,lim,aolp_plot=False):

    n = Q_PHI.shape[0]
    Q=Q[int(n/2-lim):int(n/2+lim),int(n/2-lim):int(n/2+lim)]
    U=U[int(n/2-lim):int(n/2+lim),int(n/2-lim):int(n/2+lim)]
    R=R[int(n/2-lim):int(n/2+lim),int(n/2-lim):int(n/2+lim)]
    I=I[int(n/2-lim):int(n/2+lim),int(n/2-lim):int(n/2+lim)]
    Q_PHI=Q_PHI[int(n/2-lim):int(n/2+lim),int(n/2-lim):int(n/2+lim)]

    # First, we plot the background image
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    n = Q_PHI.shape[0]
    d = n * ps / 2

    im1=ax.imshow(np.arcsinh(Q_PHI), origin='lower',extent=(-d, d, d, -d))
    #plt.xlim(-lim * ps, lim * ps)
    #plt.ylim(-lim * ps, lim * ps)
    fig.colorbar(im1, orientation='vertical',shrink=0.75)

    plt.xlabel('mas',fontsize=14)
    plt.ylabel("mas",fontsize=14)

    ax.tick_params(axis='both',labelsize=14)
    plt.tight_layout(pad=0.1)     

    # ranges of the axis
    xx0, xx1 = ax.get_xlim()
    yy0, yy1 = ax.get_ylim()

    # binning factor
    factor = [4, 4]

    # re-binned number of points in each axis
    nx_new = Q_PHI.shape[1] // factor[0]
    ny_new = Q_PHI.shape[0] // factor[1]

    # These are the positions of the quivers
    X,Y = np.meshgrid(np.linspace(xx0,xx1,nx_new,endpoint=True),
                      np.linspace(yy0,yy1,ny_new,endpoint=True))
    # bin the data
    Q_bin = Q.reshape(nx_new, factor[0], ny_new, factor[1]).sum(3).sum(1)
    U_bin = U.reshape(nx_new, factor[0], ny_new, factor[1]).sum(3).sum(1)
    I_bin = I.reshape(nx_new, factor[0], ny_new, factor[1]).sum(3).sum(1)
    Q_phi_bin = Q_PHI.reshape(nx_new, factor[0], ny_new, factor[1]).sum(3).sum(1)
    R_bin=R.reshape(nx_new, factor[0], ny_new, factor[1]).sum(3).sum(1)

    #Here you have to recalculate the AoLP (marked as psi) for the binned data. If you try to just bin AoLP it wil mess up angles
    psi=0.5*np.arctan2(U_bin, Q_bin)
    #psi=ndimage.gaussian_filter(psi, sigma=(1, 1), order=0) #smooting by gaussian filter   

    # polarization fraction
    frac =Q_phi_bin/I_bin

    # mask to show only alighned

    #mask1=maskcrit(psi,R_bin)
    mask2=Q_phi_bin>=noise*5
    mask=mask2#*mask1


    #print('max DoLP in region %.3f percent'%(np.max(frac[mask])*100))

    #+pi/2 because quiver function start counting from the horizontal axis counterclockwise 
    #and the AoLP have to start from North to East (which is also counterclockvise)
    if aolp_plot:
        pixX = frac*np.cos(psi+np.pi/2) # X-vector 
        pixY = frac*np.sin(psi+np.pi/2) # Y-vector

        # keyword arguments for quiverplots
        quiveropts = dict(headlength=0, headwidth=1, pivot='middle', color='w')
        ax.quiver(X[mask], Y[mask], pixX[mask], pixY[mask],scale=0.1, **quiveropts)


    mask=mask.astype(int)
    levels = [0,1]  # Adjust this as needed
    CS = ax.contour(X, Y, mask, levels=levels, colors=['white'], extent=(-d, d, d, -d))
    #ax.clabel(CS, inline=True, fontsize=10)

    plt.title(title,fontsize=16)
    if save!=False:
        print("false")
        # plt.savefig(save,bbox_inches='tight', pad_inches=0.1)
    if plot!=False:
        plt.show()
    plt.close()


plot_AoLP(ps,q_corr_conv,u_corr_conv,R_rescaled,img_tot_conv,q_phi_corr_conv,"convolved, unresolved corrected Q phi",save=False,plot=True,noise=3e-19,lim=30,aolp_plot=True)
plot_AoLP(ps,img_q_rescaled,img_u_rescaled,R_rescaled,img_total_rescaled,pi_rescaled,"Q phi",save=False,plot=True,noise=2e-17,lim=30,aolp_plot=True)



# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def compute_grid(img: np.ndarray) -> np.ndarray:
        n_img=img.shape[0]
        shift=int((n_img)/2.)
        x = np.linspace(-shift+1, shift, num=n_img)
        y = np.linspace(-shift+1, shift, num=n_img)
        if n_img%2==0:
            x=x-0.5
            y=y-0.5
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        return R, x, y, X, Y

def plot_polarimetric_image(
    image_to_plot: np.ndarray,
    ps_mas: Optional[float],
    *,
    image_scale: str = "linear",
    title: str = "",
    save: Optional[str] = None,
    show: bool = True,
    roi_half_size: Optional[int] = None,
    roi_center: Optional[Tuple[int, int]] = None,
    cmap_image: str = "viridis",
    cbar_label: Optional[str] = None,
    return_fig_ax: bool = False,
    aolp_quiver: bool = False,
    bin_factor: Tuple[int, int] = (4, 4),
    Q: Optional[np.ndarray]=None,
    U: Optional[np.ndarray]=None,
    I: Optional[np.ndarray]=None,
    snr_threshold: Optional[float] = 1,
    noise_level: Optional[float] = None,
    quiver_scale: Optional[float] = 0.1
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Status: not fully verified

    Plot an image with optional AoLP quiver field overlaid. To overplot AoLP quivers,
    Stokes Q, U and I must be provided. The quivers are oriented by AoLP and scaled by
    polarization fraction (estimated as image/I on binned maps).

    The function:
      1. Displays the image in pixel units or milliarcseconds (if `ps_mas` is given),
      2. Optionally rois to a square region of interest (ROI),
      3. Optionally bins the image (and Stokes maps, if provided) by integer factors,
      4. Optionally overlays quivers oriented by AoLP and scaled by polarization fraction,
      5. Optionally applies SNR masking for quiver vectors,
      6. Optionally outlines the mask region with contours.


    Parameters
    ----------
    image_to_plot : np.ndarray
        2D array to use as the background image (e.g. Q_phi, polarized intensity, or total intensity).
    ps_mas : float or None
        Pixel scale in milliarcseconds per pixel. If None, axes are shown in pixel units.
    image_scale : {"linear", "log", "asinh"}, default="linear"
        Scaling applied to the background image:
          * "linear" : raw values
          * "log"    : logarithmic scaling (values <= 0 will be masked)
          * "asinh"  : inverse hyperbolic sine stretch, useful for high dynamic range
    title : str, optional
        Title for the plot.
    save : str or None, optional
        Path to save the figure. If None, the figure is not saved.
    show : bool, default=True
        If True, display the figure with `plt.show()`.
    roi_half_size : int or None, optional
        Half-size of the square ROI. If provided, roi to a region
        of size (2*roi_half_size) × (2*roi_half_size) around `roi_center`.
    roi_center : (int, int) or None, optional
        Center (y, x) of the ROI. If None, the image center is used.
    cmap_image : str, default="viridis"
        Colormap for the background image.
    cbar_label : str, optional
        Label for the colorbar.
    return_fig_ax : bool, default=False
        If True, return (fig, ax) for further modification.
    aolp_quiver : bool, default=False
        If True, overlay AoLP quivers computed from Stokes Q and U.
        Requires Q, U, and I to be provided.
    bin_factor : (int, int), default=(4, 4)
        Integer binning factors (by_y, by_x) for plotting the AoLP quivers. 
        Must evenly divide the roiped image dimensions.
    Q, U, I : np.ndarray or None, optional
        Stokes Q, U, and total intensity maps. Needed only if `aolp_quiver=True`.
    snr_threshold : float or None, optional
        SNR threshold for masking quivers. Vectors are kept where
        image_to_plot >= snr_threshold * noise_level. Requires `noise_level`.
    noise_level : float or None, optional
        Noise estimate in the same units as `image_to_plot`. Used with `snr_threshold`.
    quiver_scale : float or None, default=0.1
        Scaling factor for quiver lengths passed to `ax.quiver`. If None, use Matplotlib defaults.
    Returns
    -------
    (fig, ax) or None
        Matplotlib Figure and Axes objects if `return_fig_ax=True`, otherwise None.

    Notes
    -----
    * AoLP is always recomputed from binned Q and U, not from pre-binned angles.
    * Polarization fraction is estimated as (image_to_plot / I) on binned data.
    * If `ps_mas` is given, axis labels are in mas; otherwise pixel coordinates are used.
    * This function is not yet fully verified.
    """

    # --- helpers ---
    def _center_crop(arr, half, center=None):
        if half is None:
            return arr
        ny, nx = arr.shape
        cy, cx = (ny // 2, nx // 2) if center is None else center
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        return arr[y0:y1, x0:x1]

    def _block_reduce_sum(a: np.ndarray, by: int, bx: int) -> np.ndarray:
        """Sum-reduce in (y,x) blocks of (by,bx). Assumes divisibility."""
        ny, nx = a.shape
        if ny % by != 0 or nx % bx != 0:
            raise ValueError(f"Array shape {a.shape} not divisible by bin_factor {(by, bx)}")
        return a.reshape(ny // by, by, nx // bx, bx).sum(axis=(1, 3))

    ################################################################################
    # --- input checks ---

    allowed_scales = {"linear", "log", "asinh"}
    if image_scale not in allowed_scales:
        raise ValueError(
            f"Invalid image_scale '{image_scale}'. "
            f"Choose one of {allowed_scales}."
        )

    if aolp_quiver:
        if Q is None or U is None or I is None:
            raise ValueError("To plot AoLP quivers, Stokes Q, U and I must be provided.")
        if not (Q.shape == U.shape == I.shape == image_to_plot.shape):
            raise ValueError("All input maps must have the same shape.")

    # ---   p (optional) ---
    if roi_half_size is not None:
        image_to_plot = _center_crop(image_to_plot, roi_half_size, roi_center)
        if aolp_quiver:
            Q = _center_crop(Q, roi_half_size, roi_center)
            U = _center_crop(U, roi_half_size, roi_center)
            I = _center_crop(I, roi_half_size, roi_center)



    ny, nx = image_to_plot.shape

    # --- figure/axes ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- extent in mas or pixels ---
    if ps_mas is not None:
        half_w_mas = (nx * ps_mas) / 2.0
        half_h_mas = (ny * ps_mas) / 2.0
        extent = (-half_w_mas, half_w_mas, -half_h_mas, half_h_mas)  # (x_min, x_max, y_min, y_max)
        xlabel, ylabel = "mas", "mas"
    else:
        extent = (-(nx / 2), (nx / 2), -(ny / 2), (ny / 2))
        xlabel, ylabel = "pixel", "pixel"

    # --- background plot ---
    if image_scale == "linear":
        img_display = image_to_plot
        if cbar_label is None:
            cbar_label = "Intensity (linear)"
    elif image_scale == "log":
        img_display = np.where(image_to_plot > 0, np.log10(image_to_plot), np.nan)
        if cbar_label is None:
            cbar_label = "Intensity (log10)"
    elif image_scale == "asinh":
        img_display = np.arcsinh(image_to_plot)
        if cbar_label is None:
            cbar_label = "Intensity (asinh)"

    im = ax.imshow(img_display, origin="lower", cmap=cmap_image, extent=extent, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.73)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)


    # --- quiver overlay (optional) ---
    if aolp_quiver:
        by, bx = bin_factor

        # --- bin maps (sum) ---
        # Use sums for Q/U/I so that AoLP is computed from vector sums; fraction computed from binned Qphi/I.
        image_to_plot_b = _block_reduce_sum(image_to_plot, by, bx)
        if aolp_quiver:
            Q_b = _block_reduce_sum(Q, by, bx)
            U_b = _block_reduce_sum(U, by, bx)
            I_b = _block_reduce_sum(I, by, bx)

        ny_b, nx_b = image_to_plot_b.shape

        # --- coordinates for quiver centers in the same units as the extent ---
        if ps_mas is not None:
            xs = np.linspace(-half_w_mas + (ps_mas * bx) / 2, half_w_mas - (ps_mas * bx) / 2, nx_b)
            ys = np.linspace(-half_h_mas + (ps_mas * by) / 2, half_h_mas - (ps_mas * by) / 2, ny_b)
        else:
            xs = np.linspace(-(nx / 2) + bx / 2, (nx / 2) - bx / 2, nx_b)
            ys = np.linspace(-(ny / 2) + by / 2, (ny / 2) - by / 2, ny_b)

        Xc, Yc = np.meshgrid(xs, ys)


        # --- recompute AoLP (psi) from binned Q/U ---
        psi = 0.5 * np.arctan2(U_b, Q_b)  # radians

        # --- polarization fraction (simple estimator) ---
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(I_b != 0.0, image_to_plot_b / I_b, 0.0)

        # --- masking ---
        mask = np.ones_like(image_to_plot_b, dtype=bool)
        if snr_threshold is not None:
            if noise_level is None:
                raise ValueError("`noise_level` must be provided when using `snr_threshold`.")
            mask &= (image_to_plot_b >= snr_threshold * noise_level)


        # Rotate by +pi/2 to follow usual convention (vectors start from North and increase to East).
        dx = frac * np.cos(psi + np.pi / 2.0)
        dy = frac * np.sin(psi + np.pi / 2.0)

        quiv_kwargs = dict(headlength=0, headwidth=1, pivot="middle", color="w")
        if quiver_scale is not None:
            quiv_kwargs["scale"] = quiver_scale

        ax.quiver(Xc[mask], Yc[mask], dx[mask], dy[mask], **quiv_kwargs)

        # optional: outline the mask
        try:
            # Build an integer mask for contouring (0/1)
            mask_int = mask.astype(int)
            ax.contour(
                Xc, Yc, mask_int,
                levels=[0.5], colors=["white"], linewidths=0.8
            )
        except Exception:
            pass

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    #
    # if save:
    #     plt.savefig(save, bbox_inches="tight", pad_inches=0.1)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig_ax:
        return fig, ax
    return None




plot_polarimetric_image(q_phi_corr_conv,ps,Q=q_corr_conv,U=u_corr_conv,I=img_tot_conv,title="convolved, unresolved corrected Q phi",bin_factor=(4,4),save=False,snr_threshold=3,noise_level=5e-19,roi_half_size=30,aolp_quiver=True, quiver_scale=0.1)
plot_polarimetric_image(pi_rescaled,ps,Q=img_q_rescaled,U=img_u_rescaled,I=img_total_rescaled,title="Q phi",bin_factor=(4,4),save=False,snr_threshold=3,noise_level=2e-17,roi_half_size=30,aolp_quiver=True, quiver_scale=5)
plot_polarimetric_image(q_phi_corr_conv, ps, roi_half_size=30, image_scale="asinh")
plot_polarimetric_image(Q_conv, ps, roi_half_size=30, image_scale="linear")
plot_polarimetric_image(img_tot_conv, ps, roi_half_size=30, image_scale="linear")

plot_polarimetric_image(q_phi,pixel_scale,Q=img_q,U=img_u,I=img_tot,title="Q phi",bin_factor=(4,4),image_scale="asinh",save=False,snr_threshold=3,noise_level=1e-17,roi_half_size=100,aolp_quiver=True, quiver_scale=3)



# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_image_grid(
    images: List[np.ndarray],
    ps_mas: float,
    *,
    nrows: int,
    ncols: int,
    titles: Optional[List[str]] = None,
    group_headers: Optional[List[Tuple[float, str]]] = None,
    scale: str = "linear",           # {"linear","log","asinh"}
    roi_half_size: Optional[int] = 30,  # in pixels, for autoscaling window; None = full frame
    roi_center: Optional[Tuple[int, int]] = None,  # (y,x) in pixels; None = image center
    per_panel_autoscale: bool = True,    # vmin/vmax from roi per image
    cmap: str = "viridis",
    colorbar: str = "none",  # {"shared","individual","none"}
    cbar_label: Optional[str] = None,
    cbar_kwargs: Optional[dict] = None,  # extra kwargs passed to color
    fontsize_axes: int = 14,
    fontsize_titles: int = 16,
    figsize: Tuple[float, float] = (12, 6),
    hide_axis_rules: Optional[callable] = None,  # function(ax, row, col) -> None to customise axis visibility
    tight: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Status: not fully verified

    Plot a grid of images with consistent angular axes (mas) and flexible intensity scaling.

    Parameters
    ----------
    images : list of 2D np.ndarray
        Images to plot; length must equal nrows * ncols. All images must share the same shape.
    ps_mas : float
        Pixel scale in milliarcseconds per pixel.
    nrows, ncols : int
        Grid layout.
    titles : list of str, optional
        Per-panel titles; length should be <= len(images). Extra panels ignore titles.
    group_headers : list of (x_position, text), optional
        Figure-level headers placed at fig coords (x, 1.0). Useful for labeling column groups.
    scale : {"linear","log","asinh"}
        Intensity transform applied for display.
    roi_half_size : int or None, optional
        Half-size of the square ROI. If provided, crop to a region
        of size (2*roi_half_size) × (2*roi_half_size) around `roi_center`.
    roi_center : (int, int) or None, optional
        Center (y, x) of the ROI. If None, the image center is used.
    per_panel_autoscale : bool
        If True, compute vmin/vmax per panel; otherwise use global vmin/vmax from first image (after transform).
    cmap : str
        Colormap for imshow.
    colorbar : {"shared","individual","none"}
        Add one shared colorbar, one per subplot, or none.
    cbar_label : str or None
        Label for the colorbar(s).
    cbar_kwargs : dict, optional
        Extra kwargs passed to colorbar (e.g., dict(fraction=0.02, pad=0.02, shrink=0.9))
    fontsize_axes, fontsize_titles : int
        Font sizes for axes and titles.
    figsize : (float, float)
        Figure size in inches.
    hide_axis_rules : callable or None
        Optional hook called as hide_axis_rules(ax, row, col) to customise which spines/ticks/labels are shown.
    tight : bool
        If True, apply tight_layout().
    show : bool
        If True, call plt.show().

    Returns
    -------
    fig, axs : matplotlib Figure and Axes array.
    """

    # --- validate
    if len(images) != nrows * ncols:
        raise ValueError(f"Expected {nrows*ncols} images, got {len(images)}.")
    shapes = {im.shape for im in images}
    if len(shapes) != 1:
        raise ValueError("All images must have the same shape.")


    # --- choose transform
    allowed_scales = {"linear", "log", "asinh"}
    if scale not in allowed_scales:
        raise ValueError(f"scale must be one of {allowed_scales}, got '{scale}'.")

    def transform(img: np.ndarray) -> np.ndarray:
        if scale == "log":
            # mask non-positive values for log display
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.full_like(img, np.nan, dtype=float)
                pos = img > 0
                out[pos] = np.log10(img[pos])
            return out
        if scale == "asinh":
             return np.arcsinh(img)
        return img

    # --- extent in mas (keep your original orientation: extent=(-d, d, d, -d))
    ny, nx = images[0].shape
    if roi_half_size:
        d=roi_half_size * ps_mas
    else:
        d = (np.max([nx, ny]) - 1) * ps_mas / 2.0

    extent = (-d, d, d, -d)

    # --- helper for central crop min/max
    def crop_minmax(img_t: np.ndarray) -> Tuple[float, float]:
        if roi_half_size is None:
            sub = img_t
        else:
            if roi_center is not None:
                cy, cx = roi_center
            else:
                cy, cx = ny // 2, nx // 2
            y0, y1 = int(cy - roi_half_size/2), int(cy + roi_half_size/2)
            x0, x1 = int(cx - roi_half_size/2), int(cx + roi_half_size/2)
            sub = img_t[y0:y1, x0:x1]
        # robust min/max even if NaNs present
        return (np.nanmin(sub), np.nanmax(sub))

    # --- precompute transforms and min/max
    images_t = [transform(im) for im in images]

    if per_panel_autoscale:
        minmax = [crop_minmax(imt) for imt in images_t]
    else:
        rois = [crop_minmax(imt) for imt in images_t]
        global_vmin = np.min([mn for (mn, mx) in rois])
        global_vmax = np.max([mx for (mn, mx) in rois])
        minmax = [(global_vmin, global_vmax) for _ in images_t]

    # --- plot
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = np.atleast_2d(axs)

    if hide_axis_rules is None:
        def hide_axis_rules(ax, r, c):
            # Example: hide x labels on top row; hide y labels on right 3 panels per row
            if r == 0:
                ax.get_xaxis().set_visible(False)
            if c > 0:
                ax.get_yaxis().set_visible(False)

    im_handles = []  # store imshow handles for colorbars
    for idx, (ax, imt, (vmin, vmax)) in enumerate(zip(axs.flat, images_t, minmax)):
        im = ax.imshow(imt, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
        im_handles.append(im)
        ax.set_xlim(-d,d)
        ax.set_ylim( d, -d)  
        ax.set_xlabel("mas", fontsize=fontsize_axes)
        ax.set_ylabel("mas", fontsize=fontsize_axes)
        ax.tick_params(axis='both', labelsize=fontsize_axes)
        r, c = divmod(idx, ncols)
        hide_axis_rules(ax, r, c)
        if titles and idx < len(titles) and titles[idx]:
            ax.set_title(titles[idx], fontsize=fontsize_titles)

    # --- colorbars
    cbar_kwargs = cbar_kwargs or {}
    if colorbar == "shared":
        if per_panel_autoscale:
            print("Warning: shared colorbar with per_panel_autoscale=True corresponds only to last image.")
        # compute the tight bounding box of all subplots
        boxes = np.array([ax.get_position().extents for ax in axs.ravel()])
        left, bottom = boxes[:, 0].min(), boxes[:, 1].min()
        right, top   = boxes[:, 2].max(), boxes[:, 3].max()

        # space for colorbar to the right of the whole grid
        cbar_pad   = 0.01   # gap between grid and colorbar (figure coords)
        cbar_width = 0.02   # colorbar width (figure coords)
        cbar_ax = fig.add_axes([right + cbar_pad, bottom, cbar_width, top - bottom])

        # one shared colorbar (use any image handle; they share the same cmap & norm if global scale is used)
        im_ref = im_handles[-1]
        cbar = fig.colorbar(im_ref, cax=cbar_ax) 
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=fontsize_axes)
        cbar.ax.tick_params(labelsize=fontsize_axes)

    elif colorbar == "individual":
        # one colorbar per axes
        for ax, im in zip(axs.flat, im_handles):
            cbar = plt.colorbar(im, ax=ax, **({"orientation": "vertical", "fraction": 0.046, "pad": 0.04} | cbar_kwargs))
            if cbar_label: cbar.set_label(cbar_label, fontsize=fontsize_axes-2)
            cbar.ax.tick_params(labelsize=fontsize_axes-2)
    elif colorbar == "none":
        pass
    else:
        raise ValueError("colorbar must be one of {'shared','individual','none'}")

    # group headers at top (figure coords)
    if group_headers:
        for x, text in group_headers:
            if colorbar == "shared":
                fig.text(x, .95, text, fontsize=fontsize_titles, ha='center', va='bottom')
            else:
                fig.text(x, 1.0, text, fontsize=fontsize_titles, ha='center', va='bottom')

    if tight and colorbar != "shared":
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.01)
    if show:
        plt.show()

    return fig, axs


images_list = [
    q_phi_rescaled, pi_rescaled, q_phi_corr, pi_corr,
    Q_phi_conv, pi_conv, q_phi_corr_conv, pi_corr_conv
]
titles = ['Q$_\\phi$', 'I$_{\\mathrm{pol}}$', 'Q$_\\phi$', 'I$_{\\mathrm{pol}}$',
          'Q$_\\phi$', 'I$_{\\mathrm{pol}}$', 'Q$_\\phi$', 'I$_{\\mathrm{pol}}$']

print(ps, type(ps))

fig, axs = plot_image_grid(
    images=images_list,
    ps_mas=ps,
    nrows=2,
    ncols=4,
    titles=titles,
    group_headers=[(0.31, 'With unresolved'), (0.72, 'Without unresolved')],
    scale="asinh",
    roi_half_size=30,          
    per_panel_autoscale=True,
    colorbar="individual",
    figsize=(12, 6),
    show=True
)


# In[18]:


from scipy.ndimage import rotate

def transformtodisk(Q, U, angle):
    '''
    Transforms Stokes parameters from sky coordinates to disk coordinates using a geometric rotation.

    We load Stokes Q and U images in sky coordinates and use a geometric rotation
    (Equations 7 and 8 from Schmid 2021) to transform the Q and U into the disk coordinate system.
    This preserves the brightness distribution of the disk's lobes. The function then
    plots and saves the transformed image.

    The paper for each equation is found here: https://www.aanda.org/articles/aa/pdf/2021/11/aa40405-21.pdf, titled Quadrant polarization parameters for the scattered light of circumstellar disks by H. M. Schmid (2021).

    The angle, omega, aligns the disk's major and minor axes with the plot's x and y
    axes, with the "front" of the disk (brightest/closest part) located at the bottom.

    Input parameters:
    dir (str): The directory path where the FITS images are located.
    angle (float): The rotation angle in degrees (omega). The transformation is based on this angle.
    stokes (str): The specific Stokes parameter to compute and return ('Q' or 'U').
    band (str): The photometric band of the observation (e.g., 'I', 'V', 'H').
                Used to form the input filenames.
    lim (int): The half-width in pixels for the plot's x and y limits, centered on the image.
    verbose (bool): If True, displays the plot on the screen.

    Outputs:
    rotated_stokes (arr): The transformed image data as a NumPy array.
    '''

    # load data
    # rotate image
    w = np.deg2rad(angle)               # convert omega to radians
    # pre-calculate cos(2*omega) and sin(2*omega)
    cos2w = np.cos(2*w)
    sin2w = np.sin(2*w)

    cw, sw = np.cos(w), np.sin(w)
    R, x_s, y_s, _, _ =compute_grid(Q)

    x = x_s * cw + y_s * sw
    y = y_s * cw - x_s * sw
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    phi=np.arctan2(Y, X)+np.pi/2  # azimuthal angle in disk coordinates (0 at North, increasing to East)

    # perform the geometric transformation to disk coordinates based on the selected Stokes parameter
    Q = rotate(Q, angle, reshape=False)
    U = rotate(U, angle, reshape=False)
    rotated_Q= Q * cos2w + U * sin2w       # equation for Q_disk (7)

    rotated_U = U * cos2w - Q * sin2w       # equation for U_disk (8)




    return rotated_Q, rotated_U, x, y, R, phi

plot_polarimetric_image(q_phi,pixel_scale, roi_half_size=100)

Q_d, U_d, x_d, y_d, R_d, phi_d = transformtodisk(img_q, img_u, 90)
plot_polarimetric_image(img_q,pixel_scale, roi_half_size=100)
#plot_polarimetric_image(img_u,pixel_scale, roi_half_size=100)

plot_polarimetric_image(Q_d,pixel_scale, roi_half_size=100)
#plot_polarimetric_image(U_d,pixel_scale, roi_half_size=100)



R, x, y, X,Y=compute_grid(Q_d)
phi=np.arctan2(Y, X)-np.pi/2 # azimuthal angle in disk coordinates (0 at North, increasing to East)

pi_d=np.sqrt(Q_d**2+U_d**2)
#pi_d[(phi>0)&(phi<np.deg2rad(10))]=1

plot_polarimetric_image(pi_d,pixel_scale, roi_half_size=100)


q_phi_disc=-Q_d*np.cos(2*phi)-U_d*np.sin(2*phi)
u_phi_disc=Q_d*np.sin(2*phi)-U_d*np.cos(2*phi)

plot_polarimetric_image(q_phi_disc,pixel_scale, roi_half_size=100)


# In[19]:


from scipy.ndimage import rotate


def transformtodisk(Q, U, angle):
    '''
    Transforms Stokes parameters from sky coordinates to disk coordinates using a geometric rotation.

    We load Stokes Q and U images in sky coordinates and use a geometric rotation
    (Equations 7 and 8 from Schmid 2021) to transform the Q and U into the disk coordinate system.
    This preserves the brightness distribution of the disk's lobes. The function then
    plots and saves the transformed image.

    The paper for each equation is found here: https://www.aanda.org/articles/aa/pdf/2021/11/aa40405-21.pdf, titled Quadrant polarization parameters for the scattered light of circumstellar disks by H. M. Schmid (2021).

    The angle, omega, aligns the disk's major and minor axes with the plot's x and y
    axes, with the "front" of the disk (brightest/closest part) located at the bottom.

    Input parameters:
    dir (str): The directory path where the FITS images are located.
    angle (float): The rotation angle in degrees (omega). The transformation is based on this angle.
    stokes (str): The specific Stokes parameter to compute and return ('Q' or 'U').
    band (str): The photometric band of the observation (e.g., 'I', 'V', 'H').
                Used to form the input filenames.
    lim (int): The half-width in pixels for the plot's x and y limits, centered on the image.
    verbose (bool): If True, displays the plot on the screen.

    Outputs:
    rotated_stokes (arr): The transformed image data as a NumPy array.
    '''

    # load data
    # rotate image
    w = np.deg2rad(angle)               # convert omega to radians
    # pre-calculate cos(2*omega) and sin(2*omega)
    cos2w = np.cos(2*w)
    sin2w = np.sin(2*w)

    cw, sw = np.cos(w), np.sin(w)
    R, x_s, y_s, _, _ =compute_grid(Q)

    x = x_s * cw + y_s * sw
    y = y_s * cw - x_s * sw
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    phi=np.arctan2(Y, X)+np.pi/2  # azimuthal angle in disk coordinates (0 at North, increasing to East)

    # perform the geometric transformation to disk coordinates based on the selected Stokes parameter

    rotated_Q= Q * cos2w + U * sin2w       # equation for Q_disk (7)

    rotated_U = U * cos2w - Q * sin2w       # equation for U_disk (8)

    rotated_Q = rotate(rotated_Q, angle, reshape=False)
    rotated_U = rotate(rotated_U, angle, reshape=False)



    return rotated_Q, rotated_U, x, y, R, phi


plot_polarimetric_image(q_phi,pixel_scale, roi_half_size=100)

Q_d, U_d, x_d, y_d, R_d, phi_d = transformtodisk(img_q, img_u, 128)
plot_polarimetric_image(img_q,pixel_scale, roi_half_size=100)
#plot_polarimetric_image(img_u,pixel_scale, roi_half_size=100)

plot_polarimetric_image(Q_d,pixel_scale, roi_half_size=100)
#plot_polarimetric_image(U_d,pixel_scale, roi_half_size=100)



R, x, y, X,Y=compute_grid(Q_d)
phi=np.arctan2(Y, X)-np.pi/2 # azimuthal angle in disk coordinates (0 at North, increasing to East)

pi_d=np.sqrt(Q_d**2+U_d**2)
#pi_d[(phi>0)&(phi<np.deg2rad(10))]=1

plot_polarimetric_image(pi_d,pixel_scale, roi_half_size=100)


q_phi_disc=-Q_d*np.cos(2*phi)-U_d*np.sin(2*phi)
u_phi_disc=Q_d*np.sin(2*phi)-U_d*np.cos(2*phi)

plot_polarimetric_image(q_phi_disc,pixel_scale, roi_half_size=100)


# In[20]:


R, x, y, X,Y=compute_grid(q_phi_rescaled)
phi=np.arctan2(Y, X) # azimuthal angle in disk coordinates (0 at North, increasing to East)

q_from_qphi=q_phi_rescaled*np.cos(2*phi)
u_from_qphi=q_phi_rescaled*np.sin(2*phi)
plot_polarimetric_image(u_from_qphi,ps, roi_half_size=20)

phi= np.arctan(Y/X)
q_from_qphi=q_phi_rescaled*np.cos(2*phi)
u_from_qphi=q_phi_rescaled*np.sin(2*phi)
plot_polarimetric_image(q_from_qphi,ps, roi_half_size=20)

q_phi_rescaled=rotate(q_phi_rescaled, 90, reshape=False)

q_from_qphi=q_phi_rescaled*np.cos(2*phi)
u_from_qphi=q_phi_rescaled*np.sin(2*phi)
plot_polarimetric_image(q_from_qphi,ps, roi_half_size=20)



# In[ ]:





# # Testing Residual maps

# In[21]:


# U MON
AR_obs_PI_I, n_AR_obs_PI_I = load_obs_images('/Users/aksitadeo/PycharmProjects/PythonProject/SPHERE_data/01.SCI_AR_Pup_old/Unres+PSFcorr/01.SCI_AR_Pup_I_PI_corr_tel+unres.fits')


# In[22]:


try:
    del min, max
except:
    pass


# In[23]:


AR_sim_PI = pi_corr_conv
n_AR_sim_PI = AR_sim_PI.shape[0]
center = (n_AR_sim_PI/2, n_AR_sim_PI/2)


# In[24]:


def normalize_and_crop(image, center, r_min, r_max, crop_radius=None):
    """
    Normalize an image within a radial annulus and optionally crop it.

    Parameters
    ----------
    image : 2D array
        Input image.
    center : tuple of floats
        (xc, yc) center coordinates in pixels.
    r_min, r_max : float
        Inner and outer radii (in pixels) used for normalization.
    crop_radius : float or None
        If given, crop the image to a square region of size ±crop_radius around center.

    Returns
    -------
    image_crop : 2D array
        Normalized (and cropped) image.
    """
    xc, yc = center
    y, x = np.indices(image.shape)
    R = np.sqrt((x - xc)**2 + (y - yc)**2)

    # Create mask for normalization
    mask = (R >= r_min) & (R <= r_max)

    # Avoid division by zero
    if np.any(mask):
        norm_factor = np.nanmax(image[mask])
    else:
        raise ValueError("Normalization mask is empty — check r_min and r_max.")

    image_norm = image / norm_factor

    # Crop region if requested
    if crop_radius is not None:
        x_min = int(max(0, xc - crop_radius))
        x_max = int(min(image.shape[1], xc + crop_radius))
        y_min = int(max(0, yc - crop_radius))
        y_max = int(min(image.shape[0], yc + crop_radius))
        image_crop = image_norm[y_min:y_max, x_min:x_max]
    else:
        image_crop = image_norm

    return image_crop


# In[25]:


from skimage.filters import threshold_triangle
from skimage import exposure

def residual_fit(obs_image, model_image, title, plot=True):
    """
    Compare observed and model images morphologically using binary thresholding.

    Parameters
    ----------
    obs_image : 2D array
        Observed image (e.g., SPHERE data)
    model_image : 2D array
        Model image (e.g., MCFOST output)
    plot : bool
        If True, show comparison plots.

    Returns
    -------
    score : float
        Goodness-of-fit score (lower is better).
    diff_map : 2D array
        Binary difference map.
    """

    # https://iopscience.iop.org/article/10.3847/1538-4357/adfa15/pdf
    # Normalise images to [0,1]
    # normalise(image,center,r_min,r_max)
    center_obs = (obs_image.shape[0]/2,obs_image.shape[0]/2)
    center_mod = (model_image.shape[0]/2,model_image.shape[0]/2)
    #
    # ycr, xcr = np.unravel_index(np.nanargmax(obs_image), obs_image.shape)
    # center_obs = (xcr, ycr)
    #
    # ycr, xcr = np.unravel_index(np.nanargmax(model_image), model_image.shape)
    # center_mod = (xcr, ycr)

    print(center_obs, center_mod)
    # obs_norm = normalise(obs_image,center_obs,3,50)
    # mod_norm = normalise(model_image,center_mod,3,50)

    crop_to = 60

    obs_norm = normalize_and_crop(obs_image, center=center_obs, r_min=0, r_max=30, crop_radius=crop_to)
    mod_norm = normalize_and_crop(model_image, center=center_mod, r_min=0, r_max=30, crop_radius=crop_to)

    # compute difference map
    # compute difference map
    diff_map = np.abs(obs_norm.astype(float) - mod_norm.astype(float))

    # goodness of fit metric: sum of absolute differences, same as sum of the squared-difference map
    score = np.sum(diff_map)

    # _,_,d = calc_lims(n_UM_sim_PI,50,3.6)
    d = n_AR_sim_PI * ps / 2
    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(14, 5))
        axes[0].imshow(np.log10(obs_norm), cmap='gray')
        axes[0].set_title('Observed')
        axes[0].set_xlim(0, crop_to * 2)
        axes[0].set_ylim(0, crop_to * 2)
        axes[1].imshow(np.log10(mod_norm), cmap='gray')
        axes[1].set_title('Model')
        axes[1].set_xlim(0, crop_to * 2)
        axes[1].set_ylim(0, crop_to * 2)
        axes[2].imshow(np.log10(obs_norm), cmap='Reds')
        axes[2].imshow(np.log10(mod_norm), cmap='Blues', alpha=0.5)
        axes[2].set_xlim(0, crop_to * 2)
        axes[2].set_ylim(0, crop_to * 2)
        axes[2].set_title('Overlay')
        im = axes[3].imshow(np.arcsinh(diff_map), cmap='hot')
        axes[3].set_title(f'Residual map\nScore = {score:.2f}')
        axes[3].set_xlim(0, crop_to * 2)
        axes[3].set_ylim(0, crop_to * 2)

        # Proper colorbar
        cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label("Residual intensity", fontsize=10)

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(fig_dir+title, dpi=150)
        plt.show()

    return score, diff_map


# In[26]:

params = mod_dir.split('/')[-3:]  # ['alpha=0.001', 'h0=0.93', 'fl=1.0']

# Join with underscores for a neat filename or title
param_str = '_'.join(params)

score, _ = residual_fit(AR_obs_PI_I, AR_sim_PI, title = f'arp18_{param_str}_residual_map_iband.png', plot=True)


# In[27]:


result = {
    "path": str(mod_dir),
    "diff_score": score,
}

print(json.dumps(result))


# In[ ]:





# In[ ]:





# # Testing Modelling fits
# 

# In[28]:


# def calc_lims(shape,lim1,ps):
#
#     ps = ps
#     if ps==12.27:
#         lim=lim1*1.0
#     else:
#         lim=lim1*12.27/3.6
#
#     max = np.max(image[int(n/2-2*lim/2):int(n/2+2*lim/2),int(n/2-2*lim/2):int(n/2+2*lim/2)])
#     min=np.min(image[int(n/2-2*lim/2):int(n/2+2*lim/2),int(n/2-2*lim/2):int(n/2+2*lim/2)])
#
#     d = shape * ps / 2
#
#     return min,max,d


# In[29]:


# second_ring = False
# def giveR(min=1e-5, max=12, n=300):
#     return np.linspace(1e-5, 12, n)
#
# def sampledir(angle, n):
#     xd, yd = [], []
#     R = giveR()
#     for r in R:
#         xd.append( r*np.cos(angle*np.pi/180.) )
#         yd.append( r*np.sin(angle*np.pi/180.) )
#     return np.array(xd), np.array(yd)
#
# def giveProfile(img, x, y, angle):
#
#     fimg = interpolate.RectBivariateSpline(x, y, img)
#     xd, yd = sampledir(angle, 100)
#     profile = []
#     for xx, yy in zip(xd, yd):
#         profile.append(fimg(xx, yy)[0][0])
#     #fig, ax = plt.subplots(1, 1,figsize=(8,6))
#     #plt.plot(giveR(), profile, '.')
#     return np.array(profile)
#
# def Gauss1D(R, amp, r0, size):
#     model = amp * np.exp( -(R-r0)**2 / size )
#     return model
#
# def fitProfile(profile, r):
#     amp = amp_dict[star]
#     r0 = r0_dict[star]
#     size = size_dict[star]
#     a = [amp, r0, size]
#     sigma = 1 / (profile**2 +1e-10)
#     try:
#         popt, pcov = optimize.curve_fit( Gauss1D, giveR(), profile, p0=a, sigma=sigma, maxfev=10000)
#     except:
#         popt=a
#         popt[0]=2.5e-4
#
#     #plt.plot(giveR(), Gauss1D(giveR(), *popt))
#     #plt.show()
#     #plt.close()
#
#     return popt[1], popt[0]
#
# def GetRingCoord(img, x, y, nangles = 360):
#     ## Gives the radius, PA, amplitude of the fitte gaussian to nangles angles with a center chose to be (x,y)
#     angles = np.linspace(0, 360, nangles)
#     radii, amps = [], []
#     for ang in angles:
#         profile = giveProfile(img, x, y, ang)
#         rad, amp = fitProfile(profile, giveR())
#         radii.append(rad)
#         amps.append(amp)
#     #print(profile)
#     return np.array(radii), 90-np.array(angles), np.array(amps)
#
# # def gauss_ell_fit(img,x,y,ps,savefig,savefig1, star,annulus,Rlimit, imgru_cen):
# def gauss_ell_fit(img,x,y,ps, star,annulus,Rlimit, imgru_cen):
#     n=img.shape[0]
#     shift=n/2-0.5
#
#     rad, theta, amps = GetRingCoord(img, x, y, nangles = 360)
#
#     d=n*ps/2
#     fig, ax = plt.subplots(1, 1,figsize=(8,6))
#     plt.imshow(np.arcsinh(img), extent=[-d, d, d, -d])
#     plt.xlim(-lim * ps, lim * ps)
#     plt.ylim(-lim * ps, lim * ps)
#     cbar=plt.colorbar()
#     cbar.ax.tick_params(labelsize=20)
#     xring = rad*np.cos(theta*np.pi/180.)
#     yring = rad*np.sin(theta*np.pi/180.)
#
#     #print(amps)
#
#     xrok, yrok = [], []
#
#     for xr, yr, amp in zip(xring, yring, amps):
#         Rr = np.sqrt(xr**2+yr**2)
#         if Rr<Rlimit and second_ring==False:
#             # if amp > 2.5e-4:
#             plt.plot( xr*ps, yr*ps, 's', color='green' )
#             xrok.append(xr)
#             yrok.append(yr)
#             # else:
#             # plt.plot( xr*ps, yr*ps, 'x', color='red' )
#
#     ring = np.vstack((xrok, yrok)).transpose()
#
#     print("ring shape:", ring.shape)
#     print("NaN count:", np.isnan(ring).sum())
#
#     ell = EllipseModel()
#     # success = ell.estimate(ring)
#     #
#     # if not success or ell.params is None:
#     #     raise RuntimeError(f"Ellipse fit failed for {star}, insufficient valid points or bad geometry.")
#
#     ell.estimate(ring)
#     xc, yc, a, b, theta = ell.params
#
#     points=ell.predict_xy(np.linspace(0, 2 * np.pi, 50),params=(xc,yc,a,b,theta))
#     sigma_2=np.sum(ell.residuals(ring)**2)
#     ac,bc=a,b
#     reverse=False
#     if a<b:
#         ac,bc=b,a
#         print('reverse')
#         reverse=True
#     cosi=bc/ac
#     ecc=np.sqrt(1-bc*bc/ac/ac)
#     print('theta %.2f' % np.rad2deg(theta))
#     angle=np.rad2deg(theta)
#     #if angle<0: angle=angle+360
#     if (a<b):
#         if np.rad2deg(theta)<90:
#             angle=angle+90
#         else:
#             angle=angle-90
#
#
#     print('angle %.2f' % angle)
#     plt.plot(points[:,0]*ps,points[:,1]*ps, color='cyan')
#     plt.plot(xc*ps, yc*ps, '+', color='cyan')
#     plt.title("\n".join(wrap(star+' ellipse gaussian fit for original values. a=%.2f, b=%.2f, i=%.2f' % (a, b,np.rad2deg(np.arccos(cosi))), 60)))
#     # northeast2(lim,ps,coef=3)
#     plt.show()
#     # if second_ring:
#     #     plt.savefig(savefig1+ star+'_'+annulus+ "_ellipse_gauss_fit_2.jpeg",bbox_inches='tight', pad_inches=0.1)
#     # else:
#     #     plt.savefig(savefig1+ star+'_'+annulus+ "_ellipse_gauss_fit.jpeg",bbox_inches='tight', pad_inches=0.1)
#
#     plt.close()
#
#
#     return  points, xc, yc, a, b, theta, cosi, ecc, sigma_2, reverse, angle


# In[30]:


# import numpy as np
# from scipy.ndimage import map_coordinates
# import matplotlib.pyplot as plt
#
# def deproject_image(img, center, PA_deg, incl_deg):
#     """
#     Deproject a disc image.
#
#     Parameters
#     ----------
#     img : 2D numpy array
#         Input image (e.g., UM_obs_PI)
#     center : tuple
#         (xc, yc) center of the disc in pixels
#     PA_deg : float
#         Position angle of the disc (degrees, East of North)
#     incl_deg : float
#         Inclination of the disc (degrees)
#
#     Returns
#     -------
#     img_deproj : 2D numpy array
#         Deprojected image
#     """
#
#     xc, yc = center
#
#     # Shift coordinates to center
#     y, x = np.indices(img.shape)
#     x_shifted = x - xc
#     y_shifted = y - yc
#
#     # Rotate by PA to align major axis along x-axis
#     PA_rad = np.deg2rad(PA_deg)
#     x_rot = x_shifted * np.cos(PA_rad) + y_shifted * np.sin(PA_rad)
#     y_rot = -x_shifted * np.sin(PA_rad) + y_shifted * np.cos(PA_rad)
#
#     # Deproject inclination
#     incl_rad = np.deg2rad(incl_deg)
#     y_deproj = y_rot / np.cos(incl_rad)
#     x_deproj = x_rot
#
#     # Shift back to original image coordinates
#     deproj_coords = np.array([y_deproj + yc, x_deproj + xc])
#
#     # Interpolate
#     img_deproj = map_coordinates(img, deproj_coords, order=1, mode='reflect')
#
#     return img_deproj


# In[31]:


# import numpy as np
# import matplotlib.pyplot as plt
#
# def radial_profile_deproj(img_deproj, center, incl_deg, ps_mas, r_min=0, r_max=None, dr=1, plot=True, vmax=None, title=''):
#     """
#     Compute the radial profile of a deprojected image.
#
#     Parameters
#     ----------
#     img_deproj : 2D numpy array
#         Deprojected image (e.g., PI or Q_phi)
#     center : tuple
#         (xc, yc) pixel coordinates of the star/disc center
#     incl_deg : float
#         Inclination in degrees
#     r_min : float
#         Minimum radius (pixels) for profile
#     r_max : float
#         Maximum radius (pixels) for profile (default: half image size)
#     dr : float
#         Radial bin width in pixels
#     plot : bool
#         Whether to plot the radial profile
#     vmax : float or None
#         Max value for image plotting
#     title : str
#         Title for plots
#
#     Returns
#     -------
#     r_bins : list of float
#         Radial bin centers
#     mean_vals : list of float
#         Mean pixel values in each annulus
#     std_errs : list of float
#         Standard error in each annulus
#     """
#
#     xc, yc = center
#     cosi = np.cos(np.deg2rad(incl_deg))
#
#     ny, nx = img_deproj.shape
#     half_w_mas = (nx * ps_mas) / 2.0
#     half_h_mas = (ny * ps_mas) / 2.0
#     extent = (-half_w_mas, half_w_mas, -half_h_mas, half_h_mas)
#
#     if r_max is None:
#         r_max = min(nx, ny) / 2
#
#     # Deprojected radius grid
#     y, x = np.indices(img_deproj.shape)
#     x_shifted = x - xc
#     y_shifted = y - yc
#     R_deproj = np.sqrt(x_shifted**2 + (y_shifted / cosi)**2)
#
#
#     # Mask for the radial range of interest
#     mask_r = (R_deproj >= r_min) & (R_deproj <= r_max)
#
#     # Normalise within the masked region only
#     norm_factor = np.max(img_deproj[mask_r])
#     img_deproj = img_deproj / norm_factor
#
#
#     r_bins = []
#     mean_vals = []
#     std_errs = []
#
#     for r in np.arange(r_min, r_max, dr):
#         mask = (R_deproj >= r) & (R_deproj < r + dr)
#         pixels = img_deproj[mask]
#         if len(pixels) > 0:
#             mean_vals.append(np.mean(pixels))
#             std_errs.append(np.std(pixels) / np.sqrt(len(pixels)))
#             r_bins.append(r * 1)  # keep in pixels
#         else:
#             mean_vals.append(np.nan)
#             std_errs.append(np.nan)
#             r_bins.append(r * 1)
#
#     if plot:
#
#         plt.figure(figsize=(8,5))
#         plt.errorbar(r_bins, mean_vals, yerr=std_errs, fmt='o', color='black', ecolor='blue')
#         plt.xlabel('Radius')
#         plt.ylabel('Intensity (normalised)')
#         plt.title(title + ' - Radial Profile')
#         plt.grid(True)
#         plt.show()
#
#
#         # Optional: show deprojected image
#         # _,_,d = calc_lims(ny,50,ps)
#
#         plt.figure(figsize=(6,6))
#         plt.imshow(img_deproj, extent=(-nx,nx,-ny,ny), origin='lower', cmap='viridis', vmax=vmax)
#         plt.xlim(-r_max , r_max)
#         plt.ylim(-r_max, r_max)
#         # plt.scatter([xc], [yc], color='red', marker='*', s=100)
#         plt.title(title + ' - Deprojected Image')
#         plt.colorbar()
#         plt.show()
#
#     return r_bins, mean_vals, std_errs
#
# from matplotlib.patches import Circle
#
# def cart2polar_for_mask_defining(X, Y):
#     """Converts Cartesian coordinates to polar (radius, position angle)."""
#     r = np.sqrt(X**2 + Y**2)
#     # 0° = North (Y), increasing East (X)
#     pa = np.degrees(np.arctan2(-X, Y))  # negative X because East is positive X
#     pa[pa < 0] += 360  # ensure 0-360 deg
#     return r, pa
#
# def azimuthal_profile_deproj(image_deproj, center, radius, r_min,r_max, npoints=360):
#     """
#     Compute the azimuthal profile at a fixed radius for a deprojected image.
#
#     Parameters
#     ----------
#     image_deproj : 2D array
#         Deprojected image (PI, Q_phi, etc.)
#     xc, yc : float
#         Center coordinates in pixels
#     radius : float
#         Radius at which to sample the azimuthal profile (pixels)
#     npoints : int
#         Number of azimuthal points
#
#     Returns
#     -------
#     pos_angles_deg : np.ndarray
#         Position angles (0-360 deg)
#     profile_vals : np.ndarray
#         Interpolated pixel values along the circle
#     """
#
#     theta = np.linspace(0, 2*np.pi, npoints)
#     xc, yc = center
#
#     # Deprojected radius grid
#     y, x = np.indices(image_deproj.shape)
#     x_shifted = x - xc
#     y_shifted = y - yc
#     R_deproj = np.sqrt(x_shifted**2 + (y_shifted)**2)
#
#     # Mask for the radial range of interest
#     mask_r = (R_deproj >= r_min) & (R_deproj <= r_max)
#
#     # Normalise within the masked region only
#     norm_factor = np.max(image_deproj[mask_r])
#     image_deproj = image_deproj / norm_factor
#     print(norm_factor)
#
#     # Circular coordinates
#     x = xc + radius * np.cos(theta)
#     y = yc + radius * np.sin(theta)
#
#     # image_deproj = image_deproj/np.max(image_deproj)
#
#     # y, x = np.indices(image_deproj.shape)
#     # R = np.sqrt((x - xc)**2 + (y - yc)**2)
#     # image_plot = np.copy(image_deproj).astype(float)
#
#     # Interpolate pixel values (bilinear)
#     profile_vals = np.zeros(npoints)
#     profile_errs = np.zeros(npoints)
#
#     for i, (xi, yi) in enumerate(zip(x, y)):
#         ix = int(np.floor(xi))
#         iy = int(np.floor(yi))
#         if ix >= 0 and ix + 1 < image_deproj.shape[1] and iy >= 0 and iy + 1 < image_deproj.shape[0]:
#             dx = xi - ix
#             dy = yi - iy
#             f00 = image_deproj[iy, ix]
#             f10 = image_deproj[iy, ix+1]
#             f01 = image_deproj[iy+1, ix]
#             f11 = image_deproj[iy+1, ix+1]
#             profile_vals[i] = f00*(1-dx)*(1-dy) + f10*dx*(1-dy) + f01*(1-dx)*dy + f11*dx*dy
#             profile_errs[i] = np.std([f00, f10, f01, f11])
#         else:
#             profile_vals[i] = np.nan
#             profile_errs[i] = np.nan
#
#     # profile_errs = np.std(profile_vals)
#
#     circle_X = x - xc  # East offset
#     circle_Y = y - yc  # North offset
#
#     # Compute PA using astronomical convention
#     _, pos_angles_deg = cart2polar_for_mask_defining(circle_X, circle_Y)
#     pos_angles_deg = np.sort(pos_angles_deg)
#     # pos_angles_deg = np.rad2deg(theta)
#
#     ny, nx = image_deproj.shape
#     # half_w_mas = (nx * ps) / 2.0
#     # half_h_mas = (ny * ps) / 2.0
#     # extent = (-half_w_mas, half_w_mas, -half_h_mas, half_h_mas)
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
#
#     # Plot azimuthal profile
#     ax1.plot(pos_angles_deg, profile_vals, '-', color='blue')
#     ax1.set_xlabel('Position angle [deg]')
#     ax1.set_ylabel('Pixel value')
#     # ax1.set_title(f'Azimuthal profile at r = {radius} ')
#     ax1.grid(True)
#
#     # Plot image with circle
#     im = ax2.imshow(image_deproj, extent=(-nx,nx,-ny,ny), origin='lower', cmap='viridis')
#     plt.xlim(-r_max,r_max)
#     plt.ylim(-r_max,r_max)
#     cbar = plt.colorbar(im, ax=ax2)
#     cbar.set_label('Normalised intensity')
#     circle = Circle((0, 0), radius, color='red', fill=False, lw=2)
#     ax2.add_patch(circle)
#     ax2.set_xlabel('pix')
#     ax2.set_ylabel('pix')
#     # ax2.set_title(f'Radius = {radius}')
#
#     plt.tight_layout()
#     plt.show()
#     return pos_angles_deg, profile_vals, profile_errs
#
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.measure import profile_line


# In[32]:


# def normalise(image,center,r_min,r_max):
#     # Create a radial distance grid
#     xc, yc = center
#     y, x = np.indices(image.shape)
#     R = np.sqrt((x - xc)**2 + (y - yc)**2)
#
#     # Create a mask for the annulus
#     mask = (R >= r_min) & (R <= r_max)
#
#     # Normalize only using pixels inside the mask
#     image_norm = image / np.nanmax(image[mask])
#
#     return image_norm


# In[ ]:





# In[33]:


# amp_dict={'HR4049_20190108':120,'HR4049_20190107':120,'HR4049_combined':120,'V709_Car':100,'HR4226':100,'UMon':100}
# r0_dict={'HR4049_20190108':10,'HR4049_20190107':10,'HR4049_combined':10,'V709_Car':10,'HR4226':5,'UMon':8}
# size_dict={'HR4049_20190108':10,'HR4049_20190107':10,'HR4049_combined':10,'V709_Car':10,'HR4226':5,'UMon':5}
# starnames = {'HD75885':'HD75885','AR_Pup_dc_notnorm':'AR Pup','HR4049_combined':'HR4049','HR4049_20190108':'HR4049/2019-01-08','HR4049_20190107':'HR4049/2019-01-07','IRAS08544-4431':'IRAS08544-4431','UMon':'U Mon','AR_Pup_flat4':'AR_Pup_flat4','V709_Car':'V709 Car','UMon_calibV390':'UMon_calibV390','HR4226':'HR4226','UMon':'U Mon'}
#
# starsdict = {'HR4049_20190108':1449,'HR4049_20190107':1449,'HR4049_combined':3471,'UMon':800,'V709_Car':0,'HR4226':0}


# ## I Band

# In[34]:


# ps = 3.6
# Rlimit=100
# img_in=UM_obs_PI_I*1.0
# n=img_in.shape[0]
# d_in=(n-1)/2
# d=d_in*ps
# x = np.linspace(-d, d, n)
# y = np.linspace(-d, d, n)
# x2, y2 = np.meshgrid(x, y)
# R = np.sqrt(x2**2+y2**2)
#
# lim=35
# img = img_in * (R<Rlimit) * (R>1) ##mask in radius
# image_plot=np.arcsinh(img)
# n=img.shape[0]
#
# fig, ax = plt.subplots(1, 1,figsize=(8,6))
#
#
#
# print(R.shape)
#
# plt.imshow(image_plot,vmax=np.max(image_plot),extent=(-d, d, d, -d))
# plt.xlim(-lim * ps, lim * ps)
# plt.ylim(-lim * ps, lim * ps)
# x0=(511.5)*ps-d
# y0=(511.5)*ps-d
# plt.plot(x0,y0,'+',color='white')
# cbar=plt.colorbar()
# cbar.ax.tick_params(labelsize=20)
#
# plt.title("\n".join(wrap('I-band', 60)), fontsize=28)  #+' ellipse fit with gauss+arcs'
# plt.xlabel('mas', fontsize=24)
# plt.ylabel("mas", fontsize=24)
# ax.xaxis.set_tick_params(labelsize=20)
# ax.yaxis.set_tick_params(labelsize=20)
#
# plt.show()


# In[35]:


# img2 = img
# Rlimit2 = Rlimit
# star = 'UMon'
# points, xc, yc, a, b, theta, cosi, ecc, sigma_2, reverse, angle = gauss_ell_fit(img2, x, y, ps, 'UMon', 'I', Rlimit2, img)
# xc_in = xc * 1.0
# yc_in = yc * 1.0
# print(xc, yc, a, b, theta, str(np.rad2deg(np.arccos(cosi))), ecc)
#
# #height of arc
# starposition = [(511.5 - n / 2), (511.5 - n / 2)]  #center in python array index
# ringshift = math.dist(starposition, [xc, yc])
# sini = np.sqrt(1 - cosi * cosi)


# In[36]:


# deprojection = True
#
# print(xc,yc)
#
# prexc = img.shape[0]/2 + xc
# preyc = img.shape[0]/2 + yc
#
# def rotate_points_el(x,y,angle,xc,yc):
#     rotx=xc+(x-xc)*np.cos(angle)-(y-yc)*np.sin(angle)
#     roty=yc+(y-yc)*np.cos(angle)+(x-xc)*np.sin(angle)
#     return rotx,roty
#
# if deprojection:
#
#     img=img_in*1.0
#     print(np.nanmin(img_in), np.nanmax(img_in), np.count_nonzero(img_in))
#
#     angle_rot=np.deg2rad(-angle) #because image is plotted upside down (result of pythons array numeration) and rotate_image looks like working in the opposite direction
#
#     ell = EllipseModel()
#     image_rotated=f.rotate_image(img,angle,prexc,preyc)
#     print(image_rotated.shape, np.nanmin(image_rotated), np.nanmax(image_rotated), np.count_nonzero(image_rotated))
#
#
#     points_rotx,points_roty=rotate_points_el(points[:,0],points[:,1],angle_rot,prexc-d,preyc-d)
#     points_rotx=points_rotx+d
#     points_roty=points_roty+d
#
#     d = (n-1)*ps/2
#     lim=35
#     fig, ax = plt.subplots(1, 1,figsize=(8,6))
#     #image_rotated=np.sinh(image_rotated)*1.0*R*R
#
#     plt.imshow(np.arcsinh(image_rotated),vmax=np.max(np.arcsinh(image_rotated)), extent=(-d, d, d,-d))
#     plt.plot(points_rotx*ps-d,points_roty*ps-d, lw=2, color='red')
#
#     ax.plot(prexc*ps-d, preyc*ps-d, "+", ms=10, color='red')
#     plt.xlim(-lim*ps, lim*ps)
#     plt.ylim(-lim*ps, lim*ps)
#     plt.xlabel('mas', fontsize=24)
#     plt.ylabel("mas",fontsize=24)
#
#     cbar=plt.colorbar()
#     cbar.ax.tick_params(labelsize=20)
#     plt.tight_layout()
#     # northeast_rotated(angle,xc,yc,lim,ps,coef=3)
#     plt.title('I', fontsize=28)
#     ax.xaxis.set_tick_params(labelsize=20)
#     ax.yaxis.set_tick_params(labelsize=20)
#     plt.title('I')
#     plt.show()
#     plt.close()
#
#
#     image_rotated_plot=image_rotated*1.0
#
#     fig, ax = plt.subplots(1, 1,figsize=(8,6))
#
#     plt.imshow(np.arcsinh(image_rotated_plot),vmax=np.max(np.arcsinh(image_rotated_plot)), extent=(-d, d, d/cosi,-d/cosi))
#     ax.plot(prexc*ps-d, preyc*ps-d, "*", color='red')
#     plt.plot(points_rotx*ps-d,(points_roty*ps-d)/cosi, lw=2, color='red')
#     plt.xlim(-lim*ps, lim*ps)
#     plt.ylim(-lim*ps, lim*ps)
#     plt.xlabel('mas', fontsize=24)
#     plt.ylabel("mas", fontsize=24)
#     cbar=plt.colorbar()
#     cbar.ax.tick_params(labelsize=20)
#     ax.xaxis.set_tick_params(labelsize=20)
#     ax.yaxis.set_tick_params(labelsize=20)
#     plt.tight_layout()
#     # northeast_rotated_depr(angle,xc,yc,35,cosi,ps,coef=3)
#     plt.title('I', fontsize=28)#+' '+'deprojected')
#     # plt.savefig(savefig1+star+'_'+annulus+'_deproj.jpeg',bbox_inches='tight', pad_inches=0.1)
#     plt.title('I')
#     plt.show()
#     plt.close()


# In[37]:


# UM_obs_PI_I_deproj = image_rotated.copy()


# In[38]:


# # --- Parameters from your stats ---
#
# # Get array center
# yc_img, xc_img = np.array(UM_obs_PI_I.shape) / 2
#
# # Corrected center in absolute pixel coordinates
# center = (xc_img, yc_img)
# PA = 131         # position angle in degrees
# incl = 41        # inclination in degrees
#
# # --- Apply deprojection ---
# UM_obs_PI_I_deproj = deproject_image(UM_obs_PI_I, center, PA, incl)
#
# # --- Quick check ---
# _,_,d = calc_lims(n_UM_obs_PI_I,50,3.6)
# lims = 100
#
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(UM_obs_PI_I,extent=[-d,d,-d,d], origin='lower', cmap='viridis')
# plt.xlim(-lims,lims)
# plt.ylim(-lims,lims)
# plt.title('Original PI I-band')
# plt.colorbar()
#
# plt.subplot(1,2,2)
# plt.imshow(UM_obs_PI_I_deproj,extent=[-d,d,-d,d], origin='lower', cmap='viridis')
# plt.xlim(-lims,lims)
# plt.ylim(-lims,lims)
# plt.title('Deprojected PI I-band')
# plt.colorbar()
# plt.show()


# In[39]:


# r_bins_i, mean_PI_i, err_PI_i = radial_profile_deproj(
#     np.arcsinh(UM_obs_PI_I_deproj),
#     center=center,
#     incl_deg=incl,
#     ps_mas=3.6,
#     r_min=3,
#     r_max=50,   # adjust as needed
#     dr=1,
#     plot=True,
#     title='UMon I-band PI'
# )


# In[40]:


# radius_pix = 10
#
# pos_angle_i, profile_i, profile_i_err = azimuthal_profile_deproj(
#     np.arcsinh(UM_obs_PI_I_deproj), center, radius_pix, 2,50, npoints=360
# )


# In[41]:


# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.measure import profile_line
#
# try:
#     del min, max
# except:
#     pass
#
# def linear_profile_axes(image, center, r_min=None, r_max=None, ps=1.0):
#     """
#     Compute linear brightness profiles along the central axes (X and Y) of an image.
#
#     Parameters
#     ----------
#     image : 2D array
#         Input image (e.g., PI, Q_phi)
#     center : tuple of floats
#         (xc, yc) center coordinates in pixels
#     r_min : float or None
#         Minimum radius for normalization mask (optional)
#     r_max : float or None
#         Maximum radius for normalization mask (optional)
#     ps : float
#         Pixel scale in mas/pixel (default=1.0)
#
#     Returns
#     -------
#     dist_x : 1D array
#         Distances along the X-axis from center [mas]
#     profile_x : 1D array
#         Brightness along X-axis
#     err_x : 1D array
#         Local pixel standard deviation along X-axis
#
#     dist_y : 1D array
#         Distances along the Y-axis from center [mas]
#     profile_y : 1D array
#         Brightness along Y-axis
#     err_y : 1D array
#         Local pixel standard deviation along Y-axis
#     """
#     xc, yc = center
#     ny, nx = image.shape
#
#     # Deprojected radius grid
#     y, x = np.indices(image.shape)
#     x_shifted = x - xc
#     y_shifted = y - yc
#     R_deproj = np.sqrt(x_shifted**2 + (y_shifted)**2)
#
#     # Mask for the radial range of interest
#     mask_r = (R_deproj >= r_min) & (R_deproj <= r_max)
#
#     # Normalise within the masked region only
#     norm_factor = np.max(image[mask_r])
#     image_norm = image / norm_factor
#     print(norm_factor)
#
#     # X-axis profile (horizontal line through center)
#     x_start = int(max(xc - r_max, 0))
#     x_end = int(min(xc + r_max, nx - 1))
#     y_fixed = int(yc)
#     profile_x = profile_line(image_norm, (y_fixed, x_start), (y_fixed, x_end),
#                              linewidth=1, order=1, mode='nearest')
#     # Approximate error using 3x3 local pixels along the line
#     err_x = np.zeros_like(profile_x)
#     for i, xi in enumerate(np.linspace(x_start, x_end, len(profile_x))):
#         ix = int(np.floor(xi))
#         iy = int(y_fixed)
#         patch = image_norm[max(0, iy-1):min(ny, iy+2), max(0, ix-1):min(nx, ix+2)]
#         err_x[i] = np.nanstd(patch)
#
#     x_coords = np.linspace(x_start, x_end, len(profile_x))
#     dist_x = (x_coords - xc) * ps
#
#     # Y-axis profile (vertical line through center)
#     y_start = int(max(yc - r_max, 0))
#     y_end = int(min(yc + r_max, ny - 1))
#     x_fixed = int(xc)
#     profile_y = profile_line(image_norm, (y_start, x_fixed), (y_end, x_fixed),
#                              linewidth=1, order=1, mode='nearest')
#     err_y = np.zeros_like(profile_y)
#     for i, yi in enumerate(np.linspace(y_start, y_end, len(profile_y))):
#         ix = int(x_fixed)
#         iy = int(np.floor(yi))
#         patch = image_norm[max(0, iy-1):min(ny, iy+2), max(0, ix-1):min(nx, ix+2)]
#         err_y[i] = np.nanstd(patch)
#
#     y_coords = np.linspace(y_start, y_end, len(profile_y))
#     dist_y = (y_coords - yc) * ps
#
#     # Plot
#     fig, ax = plt.subplots(figsize=(10,6))
#     ax.errorbar(dist_x, profile_x, yerr=err_x, fmt='-o', markersize=3, color='blue', label='X-axis')
#     ax.errorbar(dist_y, profile_y, yerr=err_y, fmt='-o', markersize=3, color='orange', label='Y-axis')
#     # plt.xlim(-r_max, r_max)
#     # plt.ylim(-r_max, r_max)
#     ax.set_xlabel('Distance [pix]')
#     ax.set_ylabel('Normalised intensity')
#     ax.set_title('Linear brightness profile')
#     ax.legend()
#     ax.grid(True)
#     plt.show()
#
#     fig2, ax2 = plt.subplots(figsize=(8,8))
#     im2 = ax2.imshow(image,extent=(-nx,nx,-ny,ny), origin='lower', cmap='viridis')
#     plt.xlim(-r_max, r_max)
#     plt.ylim(-r_max, r_max)
#     ax2.axhline(y=0, color='blue', lw=2, label='X-axis line')
#     ax2.axvline(x=0, color='orange', lw=2, label='Y-axis line')
#     # ax2.set_title('Image with central axes lines')
#     ax2.set_xlabel('Pixels')
#     ax2.set_ylabel('Pixels')
#     plt.colorbar(im2, ax=ax2, label='Intensity')
#     # ax2.legend()
#     plt.show()
#
#     return dist_x, profile_x, err_x, dist_y, profile_y, err_y


# In[42]:


# dist_x_i, profile_x_i, err_x_i, dist_y_i, profile_y_i, err_y_i = linear_profile_axes(np.arcsinh(UM_obs_PI_I_deproj), center, r_min=3, r_max=50)


# In[43]:


# # --- Parameters from your stats ---
#
# # Get array center
# yc_img, xc_img = np.array(UM_obs_PI_V.shape) / 2
#
# # Corrected center in absolute pixel coordinates
# center = (xc_img, yc_img)
# PA = 128         # position angle in degrees
# incl = 48        # inclination in degrees
#
# # --- Apply deprojection ---
# UM_obs_PI_V_deproj = deproject_image(UM_obs_PI_V, center, PA, incl)
#
# # --- Quick check ---
# _,_,d = calc_lims(n_UM_obs_PI_V,50,3.6)
# lims = 100
#
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(UM_obs_PI_V,extent=[-d,d,-d,d], origin='lower', cmap='viridis')
# plt.xlim(-lims,lims)
# plt.ylim(-lims,lims)
# plt.title('Original PI V-band')
# plt.colorbar()
#
# plt.subplot(1,2,2)
# plt.imshow(UM_obs_PI_V_deproj,extent=[-d,d,-d,d], origin='lower', cmap='viridis')
# plt.xlim(-lims,lims)
# plt.ylim(-lims,lims)
# plt.title('Deprojected PI V-band')
# plt.colorbar()
# plt.show()


# In[44]:


# r_bins_v, mean_PI_v, err_PI_v = radial_profile_deproj(
#     UM_obs_PI_V_deproj,
#     center=center,
#     incl_deg=incl,
#     r_min=0,
#     r_max=100,   # adjust as needed
#     dr=1,
#     plot=True,
#     title='UMon V-band PI'
# )


# In[ ]:





# ## Simulation

# In[45]:


# UM_sim_PI = pi_corr_conv
# n_UM_sim_PI = UM_sim_PI.shape[0]
# center = (n_UM_sim_PI/2, n_UM_sim_PI/2)
# UM_sim_PI_deproj = deproject_image(UM_sim_PI, center, 144, 25)
#
# # --- Quick check ---
# # _,_,d = calc_lims(n_UM_sim_PI,100,3.6)
# d = n_UM_sim_PI*ps / 2
# lims = 30
#
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(UM_sim_PI,extent=[-d,d,-d,d], origin='lower', cmap='viridis')
# plt.xlim(-lims*ps,lims*ps)
# plt.ylim(-lims*ps,lims*ps)
# plt.title('Original PI')
# plt.colorbar()
#
# plt.subplot(1,2,2)
# plt.imshow(np.arcsinh(UM_sim_PI_deproj),extent=[-d,d,-d,d], origin='lower', cmap='viridis')
# plt.xlim(-lims*ps,lims*ps)
# plt.ylim(-lims*ps,lims*ps)
# plt.title('Deprojected PI')
# plt.colorbar()
# plt.show()


# In[46]:


# n_UM_sim_PI


# In[47]:


# r_bins_sim, mean_PI_sim, err_PI_sim = radial_profile_deproj(
#     np.arcsinh(UM_sim_PI_deproj),
#     center=center,
#     incl_deg=incl,
#     ps_mas=3.6,
#     r_min=3,
#     r_max=50,   # adjust as needed
#     dr=1,
#     plot=True,
#     title='UMon Model PI'
# )


# In[48]:


# radius_pix = 10
#
# pos_angle_sim, profile_sim,profile_sim_err = azimuthal_profile_deproj(
#     np.arcsinh(UM_sim_PI_deproj), center, radius_pix, 2,50, npoints=360
# )


# In[49]:


# dist_x_sim, profile_x_sim, err_x_sim, dist_y_sim, profile_y_sim, err_y_sim = linear_profile_axes(np.arcsinh(UM_sim_PI_deproj), center, r_min=3, r_max=50)


# ## Reduced chi_sq

# In[50]:


# import numpy as np
#
# def reduced_chi_squared(obs, model, obs_err, p=3):
#
#     obs = np.array(obs)
#     model = np.array(model)
#     obs_err = np.array(obs_err)
#
#     # Only keep bins with finite error > 0
#     mask = np.isfinite(obs_err) & (obs_err != 0)
#     chi2 = np.sum(((obs[mask] - model[mask])**2 / obs_err[mask]**2))
#     N = np.sum(mask)
#
#     return chi2 / (N - p)


# In[51]:


# chi_sq_i_rad = reduced_chi_squared(obs=mean_PI_sim, model=mean_PI_i, obs_err=err_PI_i, p=3)
# print(chi_sq_i_rad)
#
# plt.figure(figsize=(10,6))
#
# plt.plot(r_bins_i, mean_PI_i, color='red', alpha=0.6, lw=2, label=('Observational PI I-band'))
# plt.plot(r_bins_sim, mean_PI_sim, color='black', alpha=1.0, lw=2, label=f'Model PI (χ²={chi_sq_i_rad:.2f})')
#
# plt.xlabel('Radius [pix]', fontsize=14)
# plt.ylabel('Polarised Intensity [normalised]', fontsize=14)
# plt.title('Radial Profiles: Simulated vs Observed Models for {}'.format(mod_dir), fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
#
# # plt.savefig(fig_dir+'CHI_sq_radial_I-band.png', dpi=150)
# plt.show()


# In[52]:


# chi_sq_i_az = reduced_chi_squared(obs=profile_sim, model=profile_i, obs_err=profile_i_err, p=3)
# print(chi_sq_i_az)
#
# plt.figure(figsize=(10,6))
# plt.plot(pos_angle_i, profile_i, color='red', alpha=0.6, lw=2, label='Observational PI I-band')
# plt.plot(pos_angle_sim, profile_sim, color='black', alpha=1.0, lw=2, label=f'Model PI (χ²={chi_sq_i_az:.2f})')
#
#
# plt.xlabel('Radius [pix]', fontsize=14)
# plt.ylabel('Polarised Intensity [normalised]', fontsize=14)
# plt.title('Azimuthal Profiles: Simulated vs Observed Models for {}'.format(mod_dir), fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
#
# # plt.savefig(fig_dir+'CHI_sq_azim_I-band.png', dpi=150)
# plt.show()


# In[53]:


# chi_sq_i_lin_x = reduced_chi_squared(obs=profile_x_sim, model=profile_x_i, obs_err=err_x_i, p=3)
# chi_sq_i_lin_y = reduced_chi_squared(obs=profile_y_sim, model=profile_y_i, obs_err=err_y_sim, p=3)
# sum_chi_sq_i_lin = chi_sq_i_lin_x + chi_sq_i_lin_y
# print(sum_chi_sq_i_lin)
#
# plt.figure(figsize=(10,6))
#
# plt.plot(dist_x_sim, profile_x_sim, color='black', alpha=1.0, lw=2, label=f'Model PI (χ²={chi_sq_i_lin_x:.2f})')
# # plt.plot(dist_y_sim, profile_y_sim, color='black', alpha=1.0, lw=2)
#
# plt.plot(dist_x_i, profile_x_i, color='red', alpha=0.6, lw=2, label='Observational PI I-band')
# # plt.plot(dist_y_i, profile_y_i, color='red', alpha=0.6, lw=2)
#
# plt.xlabel('Radius [pix]', fontsize=14)
# plt.ylabel('Polarised Intensity [normalised]', fontsize=14)
# plt.title('Azimuthal Profiles X: Simulated vs Observed Models for {}'.format(mod_dir), fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
#
# # plt.savefig(fig_dir+'CHI_sq_linX_I-band.png', dpi=150)
# plt.show()


# In[54]:


# # chi_sq_i_lin_x = reduced_chi_squared(obs=profile_x_sim, model=profile_x_i, obs_err=err_x_sim, p=3)
# chi_sq_i_lin_y = reduced_chi_squared(obs=profile_y_sim, model=profile_y_i, obs_err=err_y_i, p=3)
# sum_chi_sq_i_lin = chi_sq_i_lin_x + chi_sq_i_lin_y
# print(sum_chi_sq_i_lin)
#
# plt.figure(figsize=(10,6))
#
# # plt.plot(dist_x_sim, profile_x_sim, color='black', alpha=1.0, lw=2, label=f'Model PI (χ²={sum_chi_sq_i_lin:.2f})')
# plt.plot(dist_y_sim, profile_y_sim, color='black', alpha=1.0, lw=2, label=f'Model PI (χ²={chi_sq_i_lin_y:.2f})')
#
# # plt.plot(dist_x_i, profile_x_i, color='red', alpha=0.6, lw=2, label='Observational PI I-band')
# plt.plot(dist_y_i, profile_y_i, color='red', alpha=0.6, lw=2, label='Observational PI I-band')
#
# plt.xlabel('Radius [pix]', fontsize=14)
# plt.ylabel('Polarised Intensity [normalised]', fontsize=14)
# plt.title('Azimuthal Profiles Y: Simulated vs Observed Models for {}'.format(mod_dir), fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
#
# # plt.savefig(fig_dir+'CHI_sq_linY_I-band.png', dpi=150)
# plt.show()


# In[55]:


# def normalize_and_crop(image, center, r_min, r_max, crop_radius=None):
#     """
#     Normalize an image within a radial annulus and optionally crop it.
#
#     Parameters
#     ----------
#     image : 2D array
#         Input image.
#     center : tuple of floats
#         (xc, yc) center coordinates in pixels.
#     r_min, r_max : float
#         Inner and outer radii (in pixels) used for normalization.
#     crop_radius : float or None
#         If given, crop the image to a square region of size ±crop_radius around center.
#
#     Returns
#     -------
#     image_crop : 2D array
#         Normalized (and cropped) image.
#     """
#     xc, yc = center
#     y, x = np.indices(image.shape)
#     R = np.sqrt((x - xc)**2 + (y - yc)**2)
#
#     # Create mask for normalization
#     mask = (R >= r_min) & (R <= r_max)
#
#     # Avoid division by zero
#     if np.any(mask):
#         norm_factor = np.nanmax(image[mask])
#     else:
#         raise ValueError("Normalization mask is empty — check r_min and r_max.")
#
#     image_norm = image / norm_factor
#
#     # Crop region if requested
#     if crop_radius is not None:
#         x_min = int(max(0, xc - crop_radius))
#         x_max = int(min(image.shape[1], xc + crop_radius))
#         y_min = int(max(0, yc - crop_radius))
#         y_max = int(min(image.shape[0], yc + crop_radius))
#         image_crop = image_norm[y_min:y_max, x_min:x_max]
#     else:
#         image_crop = image_norm
#
#     return image_crop


# In[56]:


# from skimage.filters import threshold_triangle
# from skimage import exposure
#
# def morphological_fit(obs_image, model_image, plot=True):
#     """
#     Compare observed and model images morphologically using binary thresholding.
#
#     Parameters
#     ----------
#     obs_image : 2D array
#         Observed image (e.g., SPHERE data)
#     model_image : 2D array
#         Model image (e.g., MCFOST output)
#     plot : bool
#         If True, show comparison plots.
#
#     Returns
#     -------
#     score : float
#         Goodness-of-fit score (lower is better).
#     diff_map : 2D array
#         Binary difference map.
#     """
#
#     # https://iopscience.iop.org/article/10.3847/1538-4357/adfa15/pdf
#     # Normalise images to [0,1]
#     # normalise(image,center,r_min,r_max)
#     center_obs = (obs_image.shape[0]/2,obs_image.shape[0]/2)
#     center_mod = (model_image.shape[0]/2,model_image.shape[0]/2)
#
#     print(center_obs, center_mod)
#     # obs_norm = normalise(obs_image,center_obs,3,50)
#     # mod_norm = normalise(model_image,center_mod,3,50)
#
#     obs_norm = normalize_and_crop(obs_image, center=center_obs, r_min=3, r_max=30, crop_radius=45)
#     mod_norm = normalize_and_crop(model_image, center=center_mod, r_min=3, r_max=30, crop_radius=45)
#
#     # apply the triangle autothreshold
#     t_obs = threshold_triangle(obs_norm)
#     t_mod = threshold_triangle(mod_norm)
#
#     obs_bin = obs_norm > t_obs
#     mod_bin = mod_norm > t_mod
#
#     # compute difference map
#     diff_map = np.abs(obs_bin.astype(float) - mod_bin.astype(float))
#
#     # goodness of fit metric: sum of absolute differences, same as sum of the squared-difference map
#     score = np.sum(diff_map)
#
#     # _,_,d = calc_lims(n_UM_sim_PI,50,3.6)
#     d = n_UM_sim_PI * ps / 2
#     if plot:
#         fig, axes = plt.subplots(1, 4, figsize=(14,5))
#         axes[0].imshow(obs_norm, cmap='gray')
#         axes[0].set_title('Observed')
#         axes[0].set_xlim(0,89)
#         axes[0].set_ylim(0,89)
#         axes[1].imshow(mod_norm, cmap='gray')
#         axes[1].set_title('Model')
#         axes[1].set_xlim(0,89)
#         axes[1].set_ylim(0,89)
#         axes[2].imshow(obs_bin, cmap='Reds')
#         axes[2].imshow(mod_bin, cmap='Blues', alpha=0.5)
#         axes[2].set_xlim(0,89)
#         axes[2].set_ylim(0,89)
#         axes[2].set_title('Binary overlay')
#         axes[3].imshow(diff_map, cmap='hot')
#         axes[3].set_title(f'Difference map\nScore = {score:.2f}')
#         axes[3].set_xlim(0,89)
#         axes[3].set_ylim(0,89)
#         for ax in axes: ax.axis('off')
#         plt.tight_layout()
#         # plt.savefig(fig_dir+'diff_map_I.png', dpi=150)
#         plt.show()
#
#     return score, diff_map


# In[57]:


# score, diff = morphological_fit(UM_obs_PI_I_deproj, UM_sim_PI_deproj)
# print(f"Morphological fit score: {score:.3f}")


# # Saving to desktop

# In[58]:


# chi_sq_total = chi_sq_i_rad + chi_sq_i_az + sum_chi_sq_i_lin


# In[59]:


# result = {
#     "path": str(mod_dir),
#     "chi_sq": chi_sq_total,
#     "diff_score": score,
# }
#
# print(json.dumps(result))


# In[60]:


# _,_,d = calc_lims(n_UM_obs_Qphi,50,3.6)
# lims = 100
#
# plt.imshow(UM_obs_PI,extent=[-d,d,-d,d])
# plt.xlim(-lims,lims)
# plt.ylim(-lims,lims)


# In[61]:


# PA_UM_obs_PI = rotate(UM_obs_PI,131,reshape=False)
#
# plt.imshow(PA_UM_obs_PI,extent=[-d,d,-d,d])
# plt.scatter(0,0,marker='x',color='r')
# plt.xlim(-lims,lims)
# plt.ylim(-lims,lims)
#
# plt.colorbar()


# In[62]:


# PA_UM_sim_PI = rotate(pi_corr_conv,144,reshape=False)
# n_PA_UM_sim_PI = PA_UM_sim_PI.shape[0]
# n_PA_UM_sim_PI = n_PA_UM_sim_PI*3.6


# In[63]:


# pi_corr_conv.shape


# In[64]:


# n_PA_UM_sim_PI


# In[65]:


# _,_,d = calc_lims(n_PA_UM_sim_PI,50,3.6)
#
# plt.imshow(PA_UM_sim_PI) # ,extent=[-d,d,-d,d]
# plt.scatter(n_PA_UM_sim_PI,n_PA_UM_sim_PI,marker='x',color='r')
# plt.xlim(-lims,lims)
# plt.ylim(-lims,lims)
#
# plt.colorbar()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[66]:


# data_dir = "/Users/aksitadeo/PycharmProjects/PythonProject/SPHERE_data/01.SCI_AR_Pup_old/"
#
# unresI, nuI = load_obs_images(data_dir+"Unres+PSFcorr/01.SCI_AR_Pup/01.SCI_AR_Pup_I_I_meancombined.fits")
# unresPI, nuPI = load_obs_images(data_dir+"Unres+PSFcorr/01.SCI_AR_Pup_I_PI_corr_tel+unres.fits")
# unresQphi, nuQphi = load_obs_images(data_dir+"Unres+PSFcorr/01.SCI_AR_Pup_I_Q_phi_corr_tel+unres.fits")
#
# decI, ndI = load_obs_images(data_dir+"Deconvolution_corr_tel/ZIMPOL/deconvolved_I/01.SCI_AR_Pup_I_decon.fits")
# decPI, ndPI = load_obs_images(data_dir+"Deconvolution_corr_tel/ZIMPOL/deconvolved_PI/01.SCI_AR_Pup_I_decon.fits")
# decQphi, ndQphi = load_obs_images(data_dir+"/Deconvolution_corr_tel/ZIMPOL/deconvolved_Q_phi/01.SCI_AR_Pup_I_decon.fits")


# In[67]:


# model = img_tot_conv.copy() #img_tot_conv.copy(), pi_conv.copy()
# obs = unresI.copy() #unresI.copy(), unresPI.copy()


# In[68]:


# model_norm = model.copy() #(model - np.min(model)) / (np.max(model) - np.min(model))
# obs_norm_noc = obs.copy() #(obs - np.min(obs)) / (np.max(obs) - np.min(obs))


# In[69]:


# model_norm = rotate(model_norm,0,reshape=True)
# # model_norm = np.arcsinh(model_norm)
#
# shape_modelx,shape_modely = model_norm.shape
#
# plt.imshow(model_norm,vmin=0)
# plt.xlim(0,shape_modelx)
# plt.ylim(0,shape_modely)
# plt.colorbar()


# In[70]:


# offset = shape_modelx/2
# shift = 7
# left = int(nuQphi/2-offset)
# right = int(nuQphi/2+offset)
# obs_norm = obs_norm_noc[left:right,left+shift:right+shift]
# plt.imshow(np.arcsinh(obs_norm))
# plt.xlim(0,offset*2)
# plt.ylim(0,offset*2)
# plt.colorbar()
# plt.xlim(nuQphi/2-offset,nuQphi/2+offset)
# plt.ylim(nuQphi/2-offset,nuQphi/2+offset)


# In[71]:


# if model_norm.shape==obs_norm.shape:
#     print("Both have same shape.")
# else:
#     print("Model and observation have different shape:")
#     print("Model = ", model_norm.shape)
#     print("Observation = ", obs_norm.shape)


# In[72]:


# bg_mask = obs_norm < np.percentile(obs_norm, 10)  # bottom 5% as background
# sigma = np.std(obs_norm[bg_mask])
#
# chi_sq = np.sum(((obs_norm - model_norm)**2) / sigma**2)
# reduced_chi_sq = chi_sq / (len(obs_norm) - 1)
#
# rms = np.sqrt(np.mean((obs_norm - model_norm)**2))
#
# corr = np.corrcoef(obs_norm, model_norm)[0, 1]
#
# print(f"Reduced χ²: {reduced_chi_sq:.3f}")
# print(f"RMS difference: {rms:.4f}")
# print(f"Correlation coefficient: {corr:.3f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




