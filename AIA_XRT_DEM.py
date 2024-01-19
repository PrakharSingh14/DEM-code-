#-----------------------------------------------------------------
#Author: Prakhar
#Purpose: Calculate DEM and generate temperature Map from AIA Images
#Method: Using established Hannah & Kontar (2012) 
#Data of creation: 8/12/2023

#Modification History
# 
#-----------------------------------------------------------------

#--Package importing---
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as io
from demregpy import dn2dem
import demregpy
import glob
from sunpy.net import Fido, attrs as a
from astropy.visualization import wcsaxes
from sys import path as sys_path
import astropy.time as atime
from astropy.coordinates import SkyCoord
from astropy import units as u
import sunpy.map
import sys
from sunpy.net import Fido, attrs
from sunpy.map import Map
import pathlib
from astropy import units as u, time as time
from astropy.io import fits
from aiapy.calibrate import degradation
from aiapy.calibrate.util import get_correction_table
from aiapy.calibrate import register, update_pointing
from aiapy.calibrate import degradation, register, update_pointing, correct_degradation
from aiapy.calibrate.util import get_correction_table
import warnings
warnings.simplefilter('ignore')

# Change to your local copy's location...
sys_path.append('/home/adithya/Desktop/AIA_temp/DEM/demreg-master/python/')
from dn2dem_pos import dn2dem_pos

#--------$$$-----------

#Required files

trin=io.readsav('/home/adithya/Desktop/AIA_temp/DEM/aia_tresp_en.dat')#AIA Response file

xrt_fls=np.loadtxt('/home/adithya/Desktop/AIA_temp/DEM/XRT_HMI_files_list.dat',dtype='str')

pathlib.Path("Temp_Fits").mkdir(parents=True, exist_ok=True) #
pathlib.Path("Temp_maps").mkdir(parents=True, exist_ok=True)

print(len(xrt_fls))
tot_n_fls=len(xrt_fls)
rej=[]
for k in range(tot_n_fls):
    try:
        image=xrt_fls[k]
        f_name=image[8:25]
        fdir = '/home/adithya/Desktop/AIA_temp/AIA_1K_Data/' +image+'/'
        
        Temp_Bins=31

        #--------$$$-------------

        files = sorted(glob.glob(fdir + 'AIA*.fits'))
        #print(files)
        if len(files) ==0:
            print('No AIA files in folder')
            continue

        if len(files) <6:
            print('Insufficient files')
            continue

        maps = [sunpy.map.Map(file) for file in files]
        maps = [Map(file) for file in files]
        maps = sorted(maps, key=lambda x: x.wavelength)
        #maps = [aiaprep(m) for m in maps]
        maps = [correct_degradation(m)/m.exposure_time for m in maps]

        wvn0 = [m.meta['wavelnth'] for m in maps]
        srt_id = sorted(range(len(wvn0)), key=wvn0.__getitem__)

        maps = [maps[i] for i in srt_id]
        #print([m.meta['wavelnth'] for m in maps])
    
        wvn = [m.meta['wavelnth'] for m in maps]
        worder=np.argsort(wvn)

        durs = [m.meta['exptime'] for m in maps]

        durs=np.array(durs)
        wvn=np.array(wvn)
        #print('duration -',durs)

        # Get the temperature response functions in the correct form for demreg
        tresp_logt=np.array(trin['logt'])
        nt=len(tresp_logt)
        nf=len(trin['tr'][:])
        trmatrix=np.zeros((nt,nf))
        for i in range(0,nf):
            trmatrix[:,i]=trin['tr'][i]

        #print(nt,nf)
        temps=np.logspace(5.7,7.0,num=Temp_Bins)
        # Temperature bin mid-points for DEM plotting
        mlogt=([np.mean([(np.log10(temps[i])),np.log10((temps[i+1]))]) \
                for i in np.arange(0,len(temps)-1)])

        mtemps=([np.mean([(temps[i]),(temps[i+1])]) \
                    for i in np.arange(0,len(temps)-1)])
        log_temps = np.log10(temps)

        nx = int(maps[0].dimensions.x.value)
        ny = int(maps[0].dimensions.y.value)
        nf = len(files)
        data = np.zeros([nx, ny, nf])

        #convert from our list to an array of data
        for j in np.arange(nf):
            data[:,:,j]=maps[j].data
        data[data < 0]=0
        serr_per=10.0
        #errors in dn/px/s

        npix=1024.**2/(nx*ny)
        edata=np.zeros([nx,ny,nf])
        gains=np.array([18.3,17.6,17.7,18.3,18.3,17.6])
        dn2ph=gains*[94,131,171,193,211,335]/3397.0
        rdnse=1.15*np.sqrt(npix)/npix
        drknse=0.17
        qntnse=0.288819*np.sqrt(npix)/npix
        for j in np.arange(nf):
            etemp=np.sqrt(rdnse**2.+drknse**2.+qntnse**2.+(dn2ph[j]*abs(data[:,:,j]))/(npix*dn2ph[j]**2))
            esys=serr_per*data[:,:,j]/100.
            edata[:,:,j]=np.sqrt(etemp**2. + esys**2.)
        

        print('Calculating DEM')
        #------Full disk DEM----------------
        dem0, edem0, elogt0, chisq0, dn_reg0 = dn2dem_pos(data, edata, trmatrix, tresp_logt, temps,max_iter=100, rgt_fact=1.5)

        em0 = np.zeros_like(dem0)
        em0_temp= np.zeros_like(dem0)

        for j in range(0,30):
            em0[:, :, j] = dem0[:, :, j] *(temps[j + 1] - temps[j])
            em0_temp[:, :, j] = em0[:, :, j]*(mtemps[j])
                #if you want to measure for particular bins i.e not for all  bins then mention like em0[:, :,1:5] for bin 1 to 4
        #total_em = np.sum(em0[:, :], axis=2)
        #total_weighted_em = np.sum(em0_temp[:, :],axis=2)
        #mean_temp = (total_weighted_em)/(total_em)


        total_em = np.sum(em0[:, :,0:30], axis=2)
        total_weighted_em = np.sum(em0_temp[:, :,0:30],axis=2)
        mean_temp = (total_weighted_em)/(total_em)


        hdu=fits.PrimaryHDU(mean_temp)
        hdu.writeto('Temp_Fits/Temp_AIA_{}.fits'.format(f_name),overwrite=True)

        
        plt.imshow(mean_temp, 'jet',origin='lower',vmin=1000000,vmax=4000000)
        #plt.imshow(np.log10(mean_temp), 'hinodexrt', origin='lower')
        plt.title('Mean Temperature Map')
        plt.axis('off')
        plt.colorbar()
        plt.savefig('Temp_maps/Temp_{}.png'.format(f_name), dpi=1000, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        print('[ ',k,' / ',tot_n_fls,' ]' )

        

    except:
        print('skipped the files')
        print('error is:', sys.exc_info()[0])
        rej.append(image)

np.savetxt('Skipped_files.dat',rej,fmt='%s')
