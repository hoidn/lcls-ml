import numpy as np
import matplotlib.pyplot as plt
import tables

def SMD_Loader(Run_Number, exp, h5dir):
    # Load the Small Data
    fname = '{}_Run{:04d}.h5'.format(exp,Run_Number)
    fname = h5dir / fname
    rr = tables.open_file(fname).root # Small Data
    print(fname)
    return rr


def EnergyFilter(rr, Energy_Filter, ROI, filter_third_harmonic=False):
    # Thresholding the detector images
    E0,dE = Energy_Filter[0],Energy_Filter[1]
    thresh_1,thresh_2 = E0-dE,E0+dE
    thresh_3,thresh_4 = 2*E0-dE,2*E0+dE
    thresh_5,thresh_6 = 3*E0-dE,3*E0+dE

    imgs_temp = rr.jungfrau1M.ROI_0_area[:10000,ROI[0]:ROI[1],ROI[2]:ROI[3]].ravel()

    imgs_cleaned = rr.jungfrau1M.ROI_0_area[:,ROI[0]:ROI[1],ROI[2]:ROI[3]]
    # The filter_third_harmonic parameter controls whether energies above the third harmonic are filtered out.
    # By default, all energies above the third harmonic are passed through.
    if filter_third_harmonic:
        imgs_cleaned[(imgs_cleaned<thresh_1)
                     |((imgs_cleaned>thresh_2)&(imgs_cleaned<thresh_3))
                     |((imgs_cleaned>thresh_4)&(imgs_cleaned<thresh_5))
                     |(imgs_cleaned>thresh_6)] = 0
    else:
        imgs_cleaned[(imgs_cleaned<thresh_1)
                     |((imgs_cleaned>thresh_2)&(imgs_cleaned<thresh_3))
                     |((imgs_cleaned>thresh_4)&(imgs_cleaned<thresh_5))] = 0

    fig, axs = plt.subplots(1,2,figsize=[15,7])
    axs[0].set_title('Before Energy Thresholding')
    axs[0].hist(imgs_temp, bins=np.arange(-5,40,0.1))
    axs[0].set_xlabel('Pixel intensity (keV)')
    axs[0].set_ylabel('Counts')
    axs[0].set_yscale('log')
    axs[0].minorticks_on()
    axs[0].grid(True,'both')
    axs[0].set_xlim([-5,40])
    axs[0].axvline(thresh_1, color='green')
    axs[0].axvline(thresh_2, color='green')
    axs[0].axvline(thresh_3, color='green')
    axs[0].axvline(thresh_4, color='green')
    axs[0].axvline(thresh_5, color='green')
    axs[0].axvline(thresh_6, color='green')
    axs[1].set_title('After Energy Thresholding')
    axs[1].hist(imgs_cleaned[:10000].ravel(), bins=np.arange(-5,40,0.1))
    axs[1].set_xlabel('Pixel intensity (keV)')
    axs[1].set_ylabel('Counts')
    axs[1].set_yscale('log')
    axs[1].minorticks_on()
    axs[1].grid(True,'both')
    axs[1].set_xlim([-5,40])
    plt.show()
    return imgs_cleaned

    #return stacks_on, stacks_off, I0, binned_delays, arg_laser_on, arg_laser_off
