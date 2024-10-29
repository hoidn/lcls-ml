# import numpy as np

# from pathlib import Path
# import tables
# import math
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from scipy.optimize import curve_fit
# #################### Define the Experiment ########################
# exp = 'xppx1003221'
# h5dir = Path('/sdf/data/lcls/ds/xpp/xppx1003221/hdf5/smalldata/')

# def projection_fit_func(x,a,c,w,k1,k0):
#     # Function to model the projection curve, gaussian + a linear background
#     return gaus(x,a,c,w)+k1*x+k0

# def CDW_Optimizer(Run_Number, ROI, Energy_Filter, I0_Threshold):
#     # This is a function used for visualizing the CDW signal on an area detector
#     # One can play with different parameters to optimize the signal
#     # Currently it involves the following parameters:
#     # 1. Run_Number: The LCLS Run Number
#     # 2. ROI: a ROI used to crop the detector, to focus on the CDW relevant pixels.
#     # **Note that this is NOT the ROI used for SmallData production
#     # 3. Energy_Filter: A two-element list containing the incident x-ray energy, and width
#     # 4. I0_Threshold: Minimum IPM reading to accept an event

#     rr = SMD_Loader(Run_Number) # Small Data Import

#     I0 = rr.ipm2.sum[:] # Getting the I0 detector

#     plt.hist(I0,bins=500)
#     plt.axvline(I0_Threshold,color='red')
#     plt.minorticks_on()
#     plt.grid(True,'both')
#     plt.xlabel('IPM2 Amplitude')
#     plt.ylabel('Number of Obervations')
#     plt.yscale('log')
#     plt.show()

#     imgs = EnergyFilter(rr,Energy_Filter,ROI) # Energy thresholding

#     # Getting the event numbers of non-lasing events, above the prescribed I0 threshold
#     evt_num = np.where((np.array(rr.evr.code_91)==1.)&(I0>=I0_Threshold))[0]

#     aI0 = I0[evt_num].mean() # Average I0

#     aimg = imgs[evt_num].mean(axis=0) # Average Image
#     simg = imgs[evt_num].std(axis=0)/np.sqrt(len(evt_num)) # STD Image

#     naimg = aimg/aI0 # Normalized
#     nsimg = simg/aI0

#     return naimg,nsimg # Return the normalized image

# def PP_lazy(imgs_on,imgs_off,mask,delay):
#     ### Quick pump probe analysis
#     Intensity_on,Intensity_off = [],[]
#     npixels = (mask==1).sum()
#     for i in range(imgs_on.shape[0]):
#         Intensity_on.append(imgs_on[i][mask==1].mean())
#         Intensity_on.append(imgs_on[i][mask==1].std()/np.sqrt(npixels))
#         Intensity_off.append(imgs_off[i][mask==1].mean())
#         Intensity_off.append(imgs_off[i][mask==1].std()/np.sqrt(npixels))
#     Intensity_on = np.array(Intensity_on).reshape(imgs_on.shape[0],2)
#     Intensity_off = np.array(Intensity_off).reshape(imgs_on.shape[0],2)
#     fig,axs = plt.subplots(1,2,figsize=[12,5])
#     axs[0].errorbar(delay,Intensity_on[:,0],Intensity_on[:,1],fmt='rs-')
#     axs[0].errorbar(delay,Intensity_off[:,0],Intensity_off[:,1],fmt='ks-',mec='k',mfc='white',alpha=0.2)
#     axs[0].set_xlabel('Delay (ps)')
#     axs[0].set_ylabel('Intensity (arb. unit)')
#     axs[0].minorticks_on()
#     axs[1].errorbar(delay,2*(Intensity_on[:,0]-Intensity_off[:,0])/(Intensity_on[:,0]+Intensity_off[:,0]),2*Intensity_on[:,1]/(Intensity_on[:,0]+Intensity_off[:,0]),fmt='rs-')  # the error is a lazy one.
#     axs[1].set_xlabel('Delay (ps)')
#     axs[1].set_ylabel('2$\\times$(I$_\mathrm{on}$-I$_\mathrm{off}$)/(I$_\mathrm{on}$+I$_\mathrm{off}$)')
#     axs[1].minorticks_on()
#     axs[1].axhline(0.,color='k',linestyle='--',lw=2.4)
#     return Intensity_on,Intensity_off


# def mask_go(aimg,row1,row2,col1,col2):
#     # produce the mask
#     # row1,row2,col1,col2: boundaries of the ROI

#     mask = np.zeros_like(aimg)
#     mask[row1:row2,col1:col2] = 1.
#     plt.imshow(aimg,cmap='jet',clim=[aimg.mean()+aimg.std()*0.5,aimg.mean()+aimg.std()*3.5])
#     plt.colorbar(shrink=0.5)
#     plt.imshow(mask,alpha=0.1)
#     plt.axhline(row1,color='white')
#     plt.axhline(row2-1,color='white')
#     plt.axvline(col1,color='white')
#     plt.axvline(col2-1,color='white')
#     plt.show()
#     return mask

# def CDW_PP(Run_Number, ROI, Energy_Filter, I0_Threshold, IPM_pos_Filter, Time_bin, TimeTool):
#     # This is a function used for visualizing the pump/probe CDW signal
#     # Currently it involves the following parameters:
#     # 1. Run_Number: The LCLS Run Number
#     # 2. ROI: a ROI used to crop the detector, to focus on the CDW relevant pixels.
#     # **Note that this is NOT the ROI used for SmallData production
#     # 3. Energy_Filter: A two-element list containing the incident x-ray energy, and width
#     # 4. I0_Threshold: Minimum IPM reading to accept an event
#     # 5. IPM_pos_Filter: IPM Positional filter
#     # 6. Time_bin: Time interval of the delay scan
#     # 7. TimeTool: two-element list, first element is the if using timetool argument (0 = no, 1 = yes), the second is threshold

#     rr = SMD_Loader(Run_Number) # Small Data Import


#     I0 = rr.ipm2.sum[:] # Getting the I0 detector
#     arg_I0 = (I0>=I0_Threshold) # I0 thresholding argument

#     I0_x = rr.ipm2.xpos[:] # ipm relative position calibrator, x
#     I0_y = rr.ipm2.ypos[:] # ipm relative position calibrator, y
#     arg = (abs(I0_x)<2.)&(abs(I0_y)<3.)
#     I0_x_mean,I0_y_mean = I0_x[arg].mean(),I0_y[arg].mean() # Mean position
#     arg_I0_x = (I0_x<(I0_x_mean+IPM_pos_Filter[0]))&(I0_x>(I0_x_mean-IPM_pos_Filter[0]))
#     arg_I0_y = (I0_y<(I0_y_mean+IPM_pos_Filter[1]))&(I0_y>(I0_y_mean-IPM_pos_Filter[1]))

#     plot_ipm_pos(I0_x,I0_y,arg,IPM_pos_Filter,I0_x_mean,I0_y_mean)


#     tt_arg = TimeTool[0] # Time tool switch, currently, turned off because the tt for this experiment was not great
#     delay = np.array(rr.enc.lasDelay) + np.array(rr.tt.FLTPOS_PS)*tt_arg # in picosecond
#     arg_delay_nan = np.isnan(delay) # Some events report NaN
#     delay = delay_bin(delay,np.array(rr.enc.lasDelay),Time_bin,arg_delay_nan) # Time binning

#     arg_tt_amplidude = (np.array(rr.tt.AMPL)>TimeTool[1]) # Timetool amplitude thresholding

#     if tt_arg==1.:
#         plot_tt_amplidue(np.array(rr.tt.AMPL),TimeTool[1])

#     arg_laser_on = (np.array(rr.evr.code_90)==1.) # Laser on argument
#     arg_laser_off = (np.array(rr.evr.code_91)==1.) # Laser off argument

#     imgs = EnergyFilter(rr,Energy_Filter,ROI) # Energy thresholding

#     delay,imgs_on,imgs_off = imgs_grouping(delay,imgs,I0,mask,arg_delay_nan,arg_I0,arg_I0_x,arg_I0_y,arg_laser_on,arg_laser_off,arg_tt_amplidude,TimeTool,ROI)

#     return delay,imgs_on,imgs_off

# def SMD_Loader(Run_Number):
#     # Load the Small Data
#     fname = '{}_Run{:04d}.h5'.format(exp,Run_Number)
#     fname = h5dir / fname
#     rr = tables.open_file(fname).root # Small Data
#     return rr
# def EnergyFilter(rr,Energy_Filter,ROI):
#     # Thresholding the detector images
#     E0,dE = Energy_Filter[0],Energy_Filter[1]
#     thresh_1,thresh_2 = E0-dE,E0+dE
#     thresh_3,thresh_4 = 2*E0-dE,2*E0+dE
#     thresh_5,thresh_6 = 3*E0-dE,3*E0+dE

#     imgs_temp = rr.jungfrau1M.ROI_0_area[:10000,ROI[0]:ROI[1],ROI[2]:ROI[3]].ravel()

#     imgs_cleaned = rr.jungfrau1M.ROI_0_area[:,ROI[0]:ROI[1],ROI[2]:ROI[3]]
#     imgs_cleaned[(imgs_cleaned<thresh_1)
#                  |((imgs_cleaned>thresh_2)&(imgs_cleaned<thresh_3))
#                  |((imgs_cleaned>thresh_4)&(imgs_cleaned<thresh_5))
#                  |(imgs_cleaned>thresh_6)] = 0

#     fig, axs = plt.subplots(1,2,figsize=[15,7])
#     axs[0].set_title('Before Energy Thresholding')
#     axs[0].hist(imgs_temp, bins=np.arange(-5,30,0.1))
#     axs[0].set_xlabel('Pixel intensity (keV)')
#     axs[0].set_ylabel('Counts')
#     axs[0].set_yscale('log')
#     axs[0].minorticks_on()
#     axs[0].grid(True,'both')
#     axs[0].set_xlim([-5,30])
#     axs[0].axvline(thresh_1, color='green')
#     axs[0].axvline(thresh_2, color='green')
#     axs[0].axvline(thresh_3, color='green')
#     axs[0].axvline(thresh_4, color='green')
#     axs[0].axvline(thresh_5, color='green')
#     axs[0].axvline(thresh_6, color='green')
#     axs[1].set_title('After Energy Thresholding')
#     axs[1].hist(imgs_cleaned[:10000].ravel(), bins=np.arange(-5,30,0.1))
#     axs[1].set_xlabel('Pixel intensity (keV)')
#     axs[1].set_ylabel('Counts')
#     axs[1].set_yscale('log')
#     axs[1].minorticks_on()
#     axs[1].grid(True,'both')
#     axs[1].set_xlim([-5,30])
#     plt.show()
#     return imgs_cleaned

# def fit_LS(func,x,y,initial_guess):  # fit the curve
#     popt,pcov = curve_fit(func, x, y, p0=initial_guess, maxfev=1000000)
#     popt,pcov = curve_fit(func, x, y, p0=popt, maxfev=1000000)
#     perr = np.sqrt(np.diag(pcov))
#     fitted_parameters = np.zeros([len(initial_guess), 2])
#     fitted_parameters[:, 0] = popt
#     fitted_parameters[:, 1]= perr
#     return fitted_parameters
# def gaus(x,area,center,width):
#     return abs(area)*np.exp(-np.power(x - center, 2.) / (2 * np.power(width, 2.))) / (width * np.sqrt(2 * np.pi))

# def SNR_analysis(aimg,ROI_FG):
#     # A function to generate the signal-to-noise ratio (SNR) by fitting
#     mask_FG,mask_BG = np.zeros_like(aimg),np.zeros_like(aimg)
#     mask_FG[ROI_FG[0]:ROI_FG[1],ROI_FG[2]:ROI_FG[3]] = 1.
#     plt.title('ROI-Cropped image after energy/I0 thresholding...')
#     plt.imshow(aimg,cmap='jet',clim=[aimg.mean()+aimg.std()*0.5,aimg.mean()+aimg.std()*3.5])
#     plt.colorbar(shrink=0.6)
#     plt.imshow(mask_FG,alpha=0.3)
#     plt.minorticks_on()
#     plt.grid(True,'both',color='white')
#     plt.show()
#     projection = aimg[:,ROI_FG[2]:ROI_FG[3]].mean(axis=1)
#     pixel_num = np.arange(len(projection))

#     p = fit_LS(projection_fit_func,pixel_num,projection,[1.66e-4,1.53e1,10.32,5.9e-9,projection[-5:].mean()])
#     a,c,w,k1,k0 = p[:,0] # fit parameters

#     plt.title('Projected intensity between column {0:d} and {1:d}'.format(ROI_FG[2],ROI_FG[3]))
#     plt.plot(pixel_num,projection,'ko')
#     plt.plot(pixel_num,projection_fit_func(pixel_num,*p[:,0]),'r--',lw=2.4)
#     plt.fill_between(pixel_num,projection_fit_func(pixel_num,*p[:,0]),k1*pixel_num+k0,color='r',alpha=0.2)
#     plt.xlabel('Pixel')
#     plt.ylabel('Intensity (arb. unit)')
#     plt.minorticks_on()
#     plt.grid(True,'both')
#     plt.show()
#     signal = gaus(c,a,c,w)
#     noise = k1*c+k0
#     SNR_a = signal/noise   # signal over background
#     SNR_b = p[0,0]/p[0,1]  # confidence of fitting
#     print('Signal over background ratio is: {:.3f}'.format(SNR_a))
#     print('Signal over uncertainity ratio is: {:.3f}'.format(SNR_b))
#     return SNR_a,SNR_b

# def plot_ipm_pos(I0_x,I0_y,arg,IPM_pos_Filter,I0_x_mean,I0_y_mean):
#     arg_I0_x = (I0_x<(I0_x_mean+IPM_pos_Filter[0]))&(I0_x>(I0_x_mean-IPM_pos_Filter[0]))
#     arg_I0_y = (I0_y<(I0_y_mean+IPM_pos_Filter[1]))&(I0_y>(I0_y_mean-IPM_pos_Filter[1]))
#     plt.title('Beam location on BPM...')
#     plt.hist2d(I0_x[arg],I0_y[arg],bins=(100,100),norm=LogNorm(),cmap='jet')
#     plt.xlabel('ipm2_posx (percentage)')
#     plt.ylabel('ipm2_posy (percentage)')
#     plt.axvline(I0_x_mean-IPM_pos_Filter[0],color='k')
#     plt.axvline(I0_x_mean+IPM_pos_Filter[0],color='k')
#     plt.axhline(I0_y_mean+IPM_pos_Filter[1],color='r')
#     plt.axhline(I0_y_mean-IPM_pos_Filter[1],color='r')
#     plt.colorbar()
#     plt.minorticks_on()
#     plt.show()

# def plot_tt_amplidue(amplitude,threshold):
#     plt.hist(amplitude,bins=100)
#     plt.xlabel('TimeTool amplitude')
#     plt.ylabel('Number of appearances')
#     plt.axvline(threshold,color='k')
#     plt.minorticks_on()
#     plt.yscale('log')
#     plt.show()

# def delay_bin(delay,delay_raw,Time_bin,arg_delay_nan):
#     delay_min,delay_max = delay_raw[arg_delay_nan==False].min(),delay_raw[arg_delay_nan==False].max()
#     num_delays = int((delay_max-delay_min)/Time_bin)
#     for i in range(num_delays+1):
#         idx = np.where((delay<=(delay_min+(i+1)*Time_bin))&(delay>=(delay_min+i*Time_bin)))[0]
#         delay[idx] = np.around(delay[idx].mean(),2)
#     print('Number of laser delays is: {0:d}, with an interval of {1:.2f} ps.'.format(num_delays,Time_bin))
#     return delay

# def imgs_grouping(delay,imgs,I0,mask,arg_delay_nan,arg_I0,arg_I0_x,arg_I0_y,arg_laser_on,arg_laser_off,arg_tt_amplidude,TimeTool,ROI):

#     delay_output = list(set(delay[arg_delay_nan==False]))
#     delay_output = np.sort(np.array(delay_output))
#     ims_group_on,ims_group_off,scan_motor = [],[],[] # laser on/off image groups
#     for i in range(len(delay_output)):
#         if TimeTool[0]==0:
#             idx_on = np.where((arg_I0==True)&(delay==delay_output[i])&(arg_I0_x==True)&(arg_I0_y==True)&(arg_laser_on==True))[0]
#             idx_off = np.where((arg_I0==True)&(delay==delay_output[i])&(arg_I0_x==True)&(arg_I0_y==True)&(arg_laser_off==True))[0]
#         elif TimeTool[0]==1.:
#             idx_on = np.where((arg_I0==True)&(delay==delay_output[i])&(arg_I0_x==True)&(arg_I0_y==True)&(arg_laser_on==True)&(arg_tt_amplidude==True))[0]
#             idx_off = np.where((arg_I0==True)&(delay==delay_output[i])&(arg_I0_x==True)&(arg_I0_y==True)&(arg_laser_off==True))[0]
#         if (len(idx_on)>20)&(len(idx_off)>20):
#             print('Working on the data of delay {:.2f} ps...'.format(delay_output[i]))
#             ims_group_on.append((imgs[idx_on].mean(axis=0)/I0[idx_on].mean(axis=0))*mask[ROI[0]:ROI[1],ROI[2]:ROI[3]])
#             ims_group_off.append((imgs[idx_off].mean(axis=0)/I0[idx_off].mean(axis=0))*mask[ROI[0]:ROI[1],ROI[2]:ROI[3]])
#             print('Number of laser on and off events after filtering are {0:d}/{1:d}.'.format(len(idx_on),len(idx_off)))
#             scan_motor.append(delay_output[i])
#     ims_group_on = np.array(ims_group_on)
#     ims_group_off = np.array(ims_group_off)
#     scan_motor = np.array(scan_motor)

#     return scan_motor,ims_group_on,ims_group_off

# def projection(imgs_on,imgs_off,direction,boundary):
#     aimg = imgs_on.mean(axis=0)
#     plt.imshow(aimg,cmap='jet',clim=[aimg[5:55,5:55].mean()-3*aimg[5:55,5:55].std(),aimg[5:55,5:55].mean()+5*aimg[5:55,5:55].std()])
#     if direction==1:
#         plt.axvline(boundary[0],color='white',lw=3.)
#         plt.axvline(boundary[1],color='white',lw=3.)
#         Intensity_on = imgs_on[:,:,boundary[0]:boundary[1]].mean(axis=direction+1)
#         Intensity_off = imgs_off[:,:,boundary[0]:boundary[1]].mean(axis=direction+1)
#     if direction==0:
#         plt.axhline(boundary[0],color='white',lw=3.)
#         plt.axhline(boundary[1],color='white',lw=3.)
#         Intensity_on = imgs_on[:,boundary[0]:boundary[1],:].mean(axis=direction+1)
#         Intensity_off = imgs_off[:,boundary[0]:boundary[1],:].mean(axis=direction+1)
#     plt.minorticks_on()
#     plt.show()

#     return Intensity_on,Intensity_off

import numpy as np

from pathlib import Path
import tables
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
#################### Define the Experiment ########################
exp = 'xppx1003221'
h5dir = Path('/sdf/data/lcls/ds/xpp/xppx1003221/hdf5/smalldata/')

def projection_fit_func(x,a,c,w,k1,k0):
    # Function to model the projection curve, gaussian + a linear background
    return gaus(x,a,c,w)+k1*x+k0

#def CDW_Optimizer(Run_Number, ROI, Energy_Filter, I0_Threshold):
#    # This is a function used for visualizing the CDW signal on an area detector
#    # One can play with different parameters to optimize the signal
#    # Currently it involves the following parameters:
#    # 1. Run_Number: The LCLS Run Number
#    # 2. ROI: a ROI used to crop the detector, to focus on the CDW relevant pixels.
#    # **Note that this is NOT the ROI used for SmallData production
#    # 3. Energy_Filter: A two-element list containing the incident x-ray energy, and width
#    # 4. I0_Threshold: Minimum IPM reading to accept an event
#
#    rr = SMD_Loader(Run_Number) # Small Data Import
#
#    I0 = rr.ipm2.sum[:] # Getting the I0 detector
#
#    plt.hist(I0,bins=500)
#    plt.axvline(I0_Threshold,color='red')
#    plt.minorticks_on()
#    plt.grid(True,'both')
#    plt.xlabel('IPM2 Amplitude')
#    plt.ylabel('Number of Obervations')
#    plt.yscale('log')
#    plt.show()
#
#    imgs = EnergyFilter(rr,Energy_Filter,ROI) # Energy thresholding
#
#    # Getting the event numbers of non-lasing events, above the prescribed I0 threshold
#    evt_num = np.where((np.array(rr.evr.code_91)==1.)&(I0>=I0_Threshold))[0]
#
#    aI0 = I0[evt_num].mean() # Average I0
#
#    aimg = imgs[evt_num].mean(axis=0) # Average Image
#    simg = imgs[evt_num].std(axis=0)/np.sqrt(len(evt_num)) # STD Image
#
#    naimg = aimg/aI0 # Normalized
#    nsimg = simg/aI0
#
#    return naimg,nsimg, imgs[evt_num] # Return the normalized image

def CDW_Optimizer(Run_Number, ROI, Energy_Filter, I0_Threshold):
    # This is a function used for visualizing the CDW signal on an area detector
    # One can play with different parameters to optimize the signal
    # Currently it involves the following parameters:
    # 1. Run_Number: The LCLS Run Number
    # 2. ROI: a ROI used to crop the detector, to focus on the CDW relevant pixels.
    # **Note that this is NOT the ROI used for SmallData production
    # 3. Energy_Filter: A two-element list containing the incident x-ray energy, and width
    # 4. I0_Threshold: Minimum IPM reading to accept an event

    rr = SMD_Loader(Run_Number) # Small Data Import

    I0 = rr.ipm2.sum[:] # Getting the I0 detector

    plt.hist(I0,bins=500)
    plt.axvline(I0_Threshold,color='red')
    plt.minorticks_on()
    plt.grid(True,'both')
    plt.xlabel('IPM2 Amplitude')
    plt.ylabel('Number of Obervations')
    plt.yscale('log')
    plt.show()

    imgs = EnergyFilter(rr,Energy_Filter,ROI) # Energy thresholding

    # Getting the event numbers of non-lasing events, above the prescribed I0 threshold
    evt_num = np.where((np.array(rr.evr.code_91)==1.)&(I0>=I0_Threshold))[0]

    aI0 = I0[evt_num].mean() # Average I0

    aimg = imgs[evt_num].mean(axis=0) # Average Image
    simg = imgs[evt_num].std(axis=0)/np.sqrt(len(evt_num)) # STD Image

    naimg = aimg/aI0 # Normalized
    nsimg = simg/aI0

    return naimg,nsimg # Return the normalized image

#def PP_lazy(imgs_on,imgs_off,mask,delay):
#    ### Quick pump probe analysis
#    Intensity_on,Intensity_off = [],[]
#    npixels = (mask==1).sum()
#    for i in range(imgs_on.shape[0]):
#        Intensity_on.append(imgs_on[i][mask==1].mean())
#        Intensity_on.append(imgs_on[i][mask==1].std()/np.sqrt(npixels))
#        Intensity_off.append(imgs_off[i][mask==1].mean())
#        Intensity_off.append(imgs_off[i][mask==1].std()/np.sqrt(npixels))
#    Intensity_on = np.array(Intensity_on).reshape(imgs_on.shape[0],2)
#    Intensity_off = np.array(Intensity_off).reshape(imgs_on.shape[0],2)
#    fig,axs = plt.subplots(1,2,figsize=[12,5])
#    axs[0].errorbar(delay,Intensity_on[:,0],Intensity_on[:,1],fmt='rs-')
#    axs[0].errorbar(delay,Intensity_off[:,0],Intensity_off[:,1],fmt='ks-',mec='k',mfc='white',alpha=0.2)
#    axs[0].set_xlabel('Delay (ps)')
#    axs[0].set_ylabel('Intensity (arb. unit)')
#    axs[0].minorticks_on()
#    axs[1].errorbar(delay,2*(Intensity_on[:,0]-Intensity_off[:,0])/(Intensity_on[:,0]+Intensity_off[:,0]),2*Intensity_on[:,1]/(Intensity_on[:,0]+Intensity_off[:,0]),fmt='rs-')  # the error is a lazy one.
#    axs[1].set_xlabel('Delay (ps)')
#    axs[1].set_ylabel('2$\\times$(I$_\mathrm{on}$-I$_\mathrm{off}$)/(I$_\mathrm{on}$+I$_\mathrm{off}$)')
#    axs[1].minorticks_on()
#    axs[1].axhline(0.,color='k',linestyle='--',lw=2.4)
#    return Intensity_on,Intensity_off
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_relative_p_values(Intensity_on, Intensity_off, assume_photon_counts=False):
    p_values = []
    for i in range(len(Intensity_on)):
        signal_on = Intensity_on[i, 0]
        signal_off = Intensity_off[i, 0]

        # If intensities are photon counts, use sqrt of intensity as std deviation
        if assume_photon_counts:
            std_dev_on = np.sqrt(signal_on)
            std_dev_off = np.sqrt(signal_off)
        else:
            std_dev_on = Intensity_on[i, 1]
            std_dev_off = Intensity_off[i, 1]

        delta_signal = abs(signal_on - signal_off)
        combined_std_dev = np.sqrt(std_dev_on**2 + std_dev_off**2)
        z_score = delta_signal / combined_std_dev
        p_value = 2 * (1 - norm.cdf(z_score))  # Two-tailed test
        p_values.append(p_value)
    return np.array(p_values)


def PP_lazy(imgs_on, imgs_off, mask, delay):
    ### Quick pump probe analysis
    Intensity_on, Intensity_off = [], []
    npixels = (mask == 1).sum()
    for i in range(imgs_on.shape[0]):
        Intensity_on.append(imgs_on[i][mask == 1].mean())
        Intensity_on.append(imgs_on[i][mask == 1].std() / np.sqrt(npixels))
        Intensity_off.append(imgs_off[i][mask == 1].mean())
        Intensity_off.append(imgs_off[i][mask == 1].std() / np.sqrt(npixels))
    Intensity_on = np.array(Intensity_on).reshape(imgs_on.shape[0], 2)
    Intensity_off = np.array(Intensity_off).reshape(imgs_on.shape[0], 2)

    # Calculate relative p-values
    p_values = calculate_relative_p_values(Intensity_on, Intensity_off)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[12, 10])
    ax1.errorbar(delay, Intensity_on[:, 0], Intensity_on[:, 1], fmt='rs-', label='Intensity On')
    ax1.errorbar(delay, Intensity_off[:, 0], Intensity_off[:, 1], fmt='ks-', mec='k', mfc='white', alpha=0.2, label='Intensity Off')
    ax1.set_xlabel('Delay (ps)')
    ax1.set_ylabel('Intensity (arb. unit)')
    ax1.minorticks_on()
    ax1.legend()

    # Add a new subplot for relative p-values
#     ax2 = ax1.twinx()
    ax2.scatter(delay, -np.log10(p_values), color='blue', label='Relative P-Values')
    ax2.set_ylabel('-log10(P-value)')
    ax2.legend(loc='upper right')

    plt.show()
    return {
        'Intensity_on': Intensity_on,
        'Intensity_off': Intensity_off,
        'p_values': p_values
    }



def mask_go(aimg,row1,row2,col1,col2):
    # produce the mask
    # row1,row2,col1,col2: boundaries of the ROI

    mask = np.zeros_like(aimg)
    mask[row1:row2,col1:col2] = 1.
    plt.imshow(aimg,cmap='jet',clim=[aimg.mean()+aimg.std()*0.5,aimg.mean()+aimg.std()*3.5])
    plt.colorbar(shrink=0.5)
    plt.imshow(mask,alpha=0.1)
    plt.axhline(row1,color='white')
    plt.axhline(row2-1,color='white')
    plt.axvline(col1,color='white')
    plt.axvline(col2-1,color='white')
    plt.show()
    return mask

def CDW_PP(Run_Number, ROI, Energy_Filter, I0_Threshold, IPM_pos_Filter, Time_bin, TimeTool):
    # This is a function used for visualizing the pump/probe CDW signal
    # Currently it involves the following parameters:
    # 1. Run_Number: The LCLS Run Number
    # 2. ROI: a ROI used to crop the detector, to focus on the CDW relevant pixels.
    # **Note that this is NOT the ROI used for SmallData production
    # 3. Energy_Filter: A two-element list containing the incident x-ray energy, and width
    # 4. I0_Threshold: Minimum IPM reading to accept an event
    # 5. IPM_pos_Filter: IPM Positional filter
    # 6. Time_bin: Time interval of the delay scan
    # 7. TimeTool: two-element list, first element is the if using timetool argument (0 = no, 1 = yes), the second is threshold

    rr = SMD_Loader(Run_Number) # Small Data Import

    # Mask for bad pixels, this needs to be changed manually, depending on which detector tile used.
    idx_tile = rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][0,0]
    mask = rr.UserDataCfg.jungfrau1M.mask[idx_tile][rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,1],rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,1]]

    I0 = rr.ipm2.sum[:] # Getting the I0 detector
    arg_I0 = (I0>=I0_Threshold) # I0 thresholding argument

    I0_x = rr.ipm2.xpos[:] # ipm relative position calibrator, x
    I0_y = rr.ipm2.ypos[:] # ipm relative position calibrator, y
    arg = (abs(I0_x)<2.)&(abs(I0_y)<3.)
    I0_x_mean,I0_y_mean = I0_x[arg].mean(),I0_y[arg].mean() # Mean position
    arg_I0_x = (I0_x<(I0_x_mean+IPM_pos_Filter[0]))&(I0_x>(I0_x_mean-IPM_pos_Filter[0]))
    arg_I0_y = (I0_y<(I0_y_mean+IPM_pos_Filter[1]))&(I0_y>(I0_y_mean-IPM_pos_Filter[1]))

    plot_ipm_pos(I0_x,I0_y,arg,IPM_pos_Filter,I0_x_mean,I0_y_mean)


    tt_arg = TimeTool[0] # Time tool switch, currently, turned off because the tt for this experiment was not great
    delay = np.array(rr.enc.lasDelay) + np.array(rr.tt.FLTPOS_PS)*tt_arg # in picosecond
    arg_delay_nan = np.isnan(delay) # Some events report NaN
    delay = delay_bin(delay,np.array(rr.enc.lasDelay),Time_bin,arg_delay_nan) # Time binning

    arg_tt_amplidude = (np.array(rr.tt.AMPL)>TimeTool[1]) # Timetool amplitude thresholding

    if tt_arg==1.:
        plot_tt_amplidue(np.array(rr.tt.AMPL),TimeTool[1])

    arg_laser_on = (np.array(rr.evr.code_90)==1.) # Laser on argument
    arg_laser_off = (np.array(rr.evr.code_91)==1.) # Laser off argument

    imgs = EnergyFilter(rr,Energy_Filter,ROI) # Energy thresholding

    delay,imgs_on,imgs_off = imgs_grouping(delay,imgs,I0,mask,arg_delay_nan,arg_I0,arg_I0_x,arg_I0_y,arg_laser_on,arg_laser_off,arg_tt_amplidude,TimeTool,ROI)

    return delay,imgs_on,imgs_off

def CDW_PP_old(Run_Number, ROI, Energy_Filter, I0_Threshold, IPM_pos_Filter, Time_bin, TimeTool):
    # This is a function used for visualizing the pump/probe CDW signal
    # Currently it involves the following parameters:
    # 1. Run_Number: The LCLS Run Number
    # 2. ROI: a ROI used to crop the detector, to focus on the CDW relevant pixels.
    # **Note that this is NOT the ROI used for SmallData production
    # 3. Energy_Filter: A two-element list containing the incident x-ray energy, and width
    # 4. I0_Threshold: Minimum IPM reading to accept an event
    # 5. IPM_pos_Filter: IPM Positional filter
    # 6. Time_bin: Time interval of the delay scan
    # 7. TimeTool: two-element list, first element is the if using timetool argument (0 = no, 1 = yes), the second is threshold

    rr = SMD_Loader(Run_Number) # Small Data Import

    # Mask for bad pixels, this needs to be changed manually, depending on which detector tile used.
    idx_tile = rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][0,0]
    mask = rr.UserDataCfg.jungfrau1M.mask[idx_tile][rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,1],rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,1]]

    I0 = rr.ipm2.sum[:] # Getting the I0 detector
    arg_I0 = (I0>=I0_Threshold) # I0 thresholding argument

    I0_x = rr.ipm2.xpos[:] # ipm relative position calibrator, x
    I0_y = rr.ipm2.ypos[:] # ipm relative position calibrator, y
    arg = (abs(I0_x)<2.)&(abs(I0_y)<3.)
    I0_x_mean,I0_y_mean = I0_x[arg].mean(),I0_y[arg].mean() # Mean position
    arg_I0_x = (I0_x<(I0_x_mean+IPM_pos_Filter[0]))&(I0_x>(I0_x_mean-IPM_pos_Filter[0]))
    arg_I0_y = (I0_y<(I0_y_mean+IPM_pos_Filter[1]))&(I0_y>(I0_y_mean-IPM_pos_Filter[1]))

    plot_ipm_pos(I0_x,I0_y,arg,IPM_pos_Filter,I0_x_mean,I0_y_mean)


    tt_arg = TimeTool[0] # Time tool switch, currently, turned off because the tt for this experiment was not great
    delay = np.array(rr.enc.lasDelay) + np.array(rr.tt.FLTPOS_PS)*tt_arg # in picosecond
    arg_delay_nan = np.isnan(delay) # Some events report NaN
    delay = delay_bin(delay,np.array(rr.enc.lasDelay),Time_bin,arg_delay_nan) # Time binning

    arg_tt_amplidude = (np.array(rr.tt.AMPL)>TimeTool[1]) # Timetool amplitude thresholding

    if tt_arg==1.:
        plot_tt_amplidue(np.array(rr.tt.AMPL),TimeTool[1])

    arg_laser_on = (np.array(rr.evr.code_90)==1.) # Laser on argument
    arg_laser_off = (np.array(rr.evr.code_91)==1.) # Laser off argument

    imgs = EnergyFilter(rr,Energy_Filter,ROI) # Energy thresholding

    delay,imgs_on,imgs_off = imgs_grouping(delay,imgs,I0,mask,arg_delay_nan,arg_I0,arg_I0_x,arg_I0_y,arg_laser_on,arg_laser_off,arg_tt_amplidude,TimeTool,ROI)

    return delay,imgs_on,imgs_off

def SMD_Loader(Run_Number):
    # Load the Small Data
    fname = '{}_Run{:04d}.h5'.format(exp,Run_Number)
    fname = h5dir / fname
    rr = tables.open_file(fname).root # Small Data
    return rr
def EnergyFilter(rr,Energy_Filter,ROI):
    # Thresholding the detector images
    E0,dE = Energy_Filter[0],Energy_Filter[1]
    thresh_1,thresh_2 = E0-dE,E0+dE
    thresh_3,thresh_4 = 2*E0-dE,2*E0+dE
    thresh_5,thresh_6 = 3*E0-dE,3*E0+dE

    imgs_temp = rr.jungfrau1M.ROI_0_area[:10000,ROI[0]:ROI[1],ROI[2]:ROI[3]].ravel()

    imgs_cleaned = rr.jungfrau1M.ROI_0_area[:,ROI[0]:ROI[1],ROI[2]:ROI[3]]
    imgs_cleaned[(imgs_cleaned<thresh_1)
                 |((imgs_cleaned>thresh_2)&(imgs_cleaned<thresh_3))
                 |((imgs_cleaned>thresh_4)&(imgs_cleaned<thresh_5))
                 |(imgs_cleaned>thresh_6)] = 0

    fig, axs = plt.subplots(1,2,figsize=[15,7])
    axs[0].set_title('Before Energy Thresholding')
    axs[0].hist(imgs_temp, bins=np.arange(-5,30,0.1))
    axs[0].set_xlabel('Pixel intensity (keV)')
    axs[0].set_ylabel('Counts')
    axs[0].set_yscale('log')
    axs[0].minorticks_on()
    axs[0].grid(True,'both')
    axs[0].set_xlim([-5,30])
    axs[0].axvline(thresh_1, color='green')
    axs[0].axvline(thresh_2, color='green')
    axs[0].axvline(thresh_3, color='green')
    axs[0].axvline(thresh_4, color='green')
    axs[0].axvline(thresh_5, color='green')
    axs[0].axvline(thresh_6, color='green')
    axs[1].set_title('After Energy Thresholding')
    axs[1].hist(imgs_cleaned[:10000].ravel(), bins=np.arange(-5,30,0.1))
    axs[1].set_xlabel('Pixel intensity (keV)')
    axs[1].set_ylabel('Counts')
    axs[1].set_yscale('log')
    axs[1].minorticks_on()
    axs[1].grid(True,'both')
    axs[1].set_xlim([-5,30])
    plt.show()
    return imgs_cleaned

def fit_LS(func,x,y,initial_guess):  # fit the curve
    popt,pcov = curve_fit(func, x, y, p0=initial_guess, maxfev=1000000)
    popt,pcov = curve_fit(func, x, y, p0=popt, maxfev=1000000)
    perr = np.sqrt(np.diag(pcov))
    fitted_parameters = np.zeros([len(initial_guess), 2])
    fitted_parameters[:, 0] = popt
    fitted_parameters[:, 1]= perr
    return fitted_parameters
def gaus(x,area,center,width):
    return abs(area)*np.exp(-np.power(x - center, 2.) / (2 * np.power(width, 2.))) / (width * np.sqrt(2 * np.pi))

def SNR_analysis(aimg,ROI_FG):
    # A function to generate the signal-to-noise ratio (SNR) by fitting
    mask_FG,mask_BG = np.zeros_like(aimg),np.zeros_like(aimg)
    mask_FG[ROI_FG[0]:ROI_FG[1],ROI_FG[2]:ROI_FG[3]] = 1.
    plt.title('ROI-Cropped image after energy/I0 thresholding...')
    plt.imshow(aimg,cmap='jet',clim=[aimg.mean()+aimg.std()*0.5,aimg.mean()+aimg.std()*3.5])
    plt.colorbar(shrink=0.6)
    plt.imshow(mask_FG,alpha=0.3)
    plt.minorticks_on()
    plt.grid(True,'both',color='white')
    plt.show()
    projection = aimg[:,ROI_FG[2]:ROI_FG[3]].mean(axis=1)
    pixel_num = np.arange(len(projection))

    p = fit_LS(projection_fit_func,pixel_num,projection,[1.66e-4,1.53e1,10.32,5.9e-9,projection[-5:].mean()])
    a,c,w,k1,k0 = p[:,0] # fit parameters

    plt.title('Projected intensity between column {0:d} and {1:d}'.format(ROI_FG[2],ROI_FG[3]))
    plt.plot(pixel_num,projection,'ko')
    plt.plot(pixel_num,projection_fit_func(pixel_num,*p[:,0]),'r--',lw=2.4)
    plt.fill_between(pixel_num,projection_fit_func(pixel_num,*p[:,0]),k1*pixel_num+k0,color='r',alpha=0.2)
    plt.xlabel('Pixel')
    plt.ylabel('Intensity (arb. unit)')
    plt.minorticks_on()
    plt.grid(True,'both')
    plt.show()
    signal = gaus(c,a,c,w)
    noise = k1*c+k0
    SNR_a = signal/noise   # signal over background
    SNR_b = p[0,0]/p[0,1]  # confidence of fitting
    print('Signal over background ratio is: {:.3f}'.format(SNR_a))
    print('Signal over uncertainity ratio is: {:.3f}'.format(SNR_b))
    return SNR_a,SNR_b

def plot_ipm_pos(I0_x,I0_y,arg,IPM_pos_Filter,I0_x_mean,I0_y_mean):
    arg_I0_x = (I0_x<(I0_x_mean+IPM_pos_Filter[0]))&(I0_x>(I0_x_mean-IPM_pos_Filter[0]))
    arg_I0_y = (I0_y<(I0_y_mean+IPM_pos_Filter[1]))&(I0_y>(I0_y_mean-IPM_pos_Filter[1]))
    plt.title('Beam location on BPM...')
    plt.hist2d(I0_x[arg],I0_y[arg],bins=(100,100),norm=LogNorm(),cmap='jet')
    plt.xlabel('ipm2_posx (percentage)')
    plt.ylabel('ipm2_posy (percentage)')
    plt.axvline(I0_x_mean-IPM_pos_Filter[0],color='k')
    plt.axvline(I0_x_mean+IPM_pos_Filter[0],color='k')
    plt.axhline(I0_y_mean+IPM_pos_Filter[1],color='r')
    plt.axhline(I0_y_mean-IPM_pos_Filter[1],color='r')
    plt.colorbar()
    plt.minorticks_on()
    plt.show()

def plot_tt_amplidue(amplitude,threshold):
    plt.hist(amplitude,bins=100)
    plt.xlabel('TimeTool amplitude')
    plt.ylabel('Number of appearances')
    plt.axvline(threshold,color='k')
    plt.minorticks_on()
    plt.yscale('log')
    plt.show()

# def delay_bin(delay,delay_raw,Time_bin,arg_delay_nan):
#     delay_min,delay_max = delay_raw[arg_delay_nan==False].min(),delay_raw[arg_delay_nan==False].max()
#     num_delays = int((delay_max-delay_min)/Time_bin)
#     for i in range(num_delays+1):
#         idx = np.where((delay<=(delay_min+(i+1)*Time_bin))&(delay>=(delay_min+i*Time_bin)))[0]
#         delay[idx] = np.around(delay[idx].mean(),2)
#     print('Number of laser delays is: {0:d}, with an interval of {1:.2f} ps.'.format(num_delays,Time_bin))
#     return delay

def delay_bin(delay, delay_raw, Time_bin, arg_delay_nan):
    """
    """
    # TODO the bin values might be off by half a picosecond
    Time_bin = Time_bin * 1.0
    delay_min = np.floor(delay_raw[arg_delay_nan==False].min())
    delay_max = np.ceil(delay_raw[arg_delay_nan==False].max())
    bins = np.arange(delay_min, delay_max + Time_bin, Time_bin)
    binned_indices = np.digitize(delay, bins)
    binned_delays = np.array([bins[idx-1] if idx > 0 else bins[0] for idx in binned_indices])
#     print('Number of laser delays is: {0:d}, with an interval of {1:.2f} ps.'.format(num_delays,Time_bin))
    return binned_delays


def imgs_grouping(delay,imgs,I0,mask,arg_delay_nan,arg_I0,arg_I0_x,arg_I0_y,arg_laser_on,arg_laser_off,arg_tt_amplidude,TimeTool,ROI):

    delay_output = list(set(delay[arg_delay_nan==False]))
    delay_output = np.sort(np.array(delay_output))
    ims_group_on,ims_group_off,scan_motor = [],[],[] # laser on/off image groups
    for i in range(len(delay_output)):
        if TimeTool[0]==0:
            idx_on = np.where((arg_I0==True)&(delay==delay_output[i])&(arg_I0_x==True)&(arg_I0_y==True)&(arg_laser_on==True))[0]
            idx_off = np.where((arg_I0==True)&(delay==delay_output[i])&(arg_I0_x==True)&(arg_I0_y==True)&(arg_laser_off==True))[0]
        elif TimeTool[0]==1.:
            idx_on = np.where((arg_I0==True)&(delay==delay_output[i])&(arg_I0_x==True)&(arg_I0_y==True)&(arg_laser_on==True)&(arg_tt_amplidude==True))[0]
            idx_off = np.where((arg_I0==True)&(delay==delay_output[i])&(arg_I0_x==True)&(arg_I0_y==True)&(arg_laser_off==True))[0]
        if (len(idx_on)>20)&(len(idx_off)>20):
            print('Working on the data of delay {:.2f} ps...'.format(delay_output[i]))
            ims_group_on.append((imgs[idx_on].mean(axis=0)/I0[idx_on].mean(axis=0))*mask[ROI[0]:ROI[1],ROI[2]:ROI[3]])
            ims_group_off.append((imgs[idx_off].mean(axis=0)/I0[idx_off].mean(axis=0))*mask[ROI[0]:ROI[1],ROI[2]:ROI[3]])
            print('Number of laser on and off events after filtering are {0:d}/{1:d}.'.format(len(idx_on),len(idx_off)))
            scan_motor.append(delay_output[i])
    ims_group_on = np.array(ims_group_on)
    ims_group_off = np.array(ims_group_off)
    scan_motor = np.array(scan_motor)

    return scan_motor,ims_group_on,ims_group_off

def projection(imgs_on,imgs_off,direction,boundary):
    aimg = imgs_on.mean(axis=0)
    plt.imshow(aimg,cmap='jet',clim=[aimg[5:55,5:55].mean()-3*aimg[5:55,5:55].std(),aimg[5:55,5:55].mean()+5*aimg[5:55,5:55].std()])
    if direction==1:
        plt.axvline(boundary[0],color='white',lw=3.)
        plt.axvline(boundary[1],color='white',lw=3.)
        Intensity_on = imgs_on[:,:,boundary[0]:boundary[1]].mean(axis=direction+1)
        Intensity_off = imgs_off[:,:,boundary[0]:boundary[1]].mean(axis=direction+1)
    if direction==0:
        plt.axhline(boundary[0],color='white',lw=3.)
        plt.axhline(boundary[1],color='white',lw=3.)
        Intensity_on = imgs_on[:,boundary[0]:boundary[1],:].mean(axis=direction+1)
        Intensity_off = imgs_off[:,boundary[0]:boundary[1],:].mean(axis=direction+1)
    plt.minorticks_on()
    plt.show()

    return Intensity_on,Intensity_off
