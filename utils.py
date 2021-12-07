import statsmodels.api as sm
import numpy as np


def find_summaries(mass, time, percentiles=np.linspace(0.1, 0.9, 9)):

    ''' compute the half mass and the half time of a galaxy 
          Input: the sfh
          Output: the half mass, the corresponding half time, and the index of the mass/time summary
    '''

    summary_masses = []
    summary_times = []
    summary_indices = []
    for percentile in percentiles:
        summary_mass = min(mass, key=lambda x: abs(x-mass[0]*percentile))  # find mass closest to the half mass
        summary_masses.append(summary_mass)
        summary_mass_indices = np.where(mass == summary_mass)[0]  # find the corresponding indices
        summary_mass_index = summary_mass_indices[0]  # chose the first index for the half mass
        summary_indices.append(summary_mass_index)
        summary_time = time[summary_mass_index]  # find the corresponding half time
        summary_times.append(summary_time)

    return summary_times, summary_masses, summary_indices


def plot_with_summaries(fig, axs, index, flux, wl, sfh):
    '''
    Plot the galaxy infos (SED, SFR, Mass) with the 0.5 and 0.9 mass summaries
    input: fig: the figure for the sublots with all indices
           axs:  the corresponding axes of the figure
           index: the ith index of the galaxies you want to plot
           flux: the flux of the galaxy
           wl: the wavelenghthes (x axis)
           sfh: the star fornation history
    output: nothing, but plot the figure
    '''

    axs[index, 0].scatter(np.array(wl)[np.array(wl)<10**3], np.log10(flux), s=10)
    axs[index, 0].set_xlabel("wavelength [$\mu m$]")
    axs[index, 0].set_xscale('log')
    axs[index, 0].set_ylabel("$\log(f)$ [Jy]")
    axs[index, 1].set_xlabel("Time")
    axs[index, 1].set_ylabel("SFR")
    axs[index, 1].plot(sfh.time, sfh.SFR_halfRad)
    axs[index, 2].plot(sfh.time, np.log10(sfh.Mstar_Half)+10)
    half_mass, half_time, half_mass_index = find_summaries(sfh, 0.5)
    ninth_mass, ninth_time, ninth_mass_index = find_summaries(sfh, 0.9)
    axs[index, 2].set_title(f'{half_mass:.2f}, {half_time:.2f}', fontsize=6)

    # axs[index, 2].set_title(half_mass, half_time)
    # print(np.nanmin(np.log10(sfh.Mstar_Half)+10), np.max(np.log10(sfh.Mstar_Half)+10))
    axs[index, 2].vlines(sfh.time[half_mass_index], 6.5, np.max(np.log10(sfh.Mstar_Half)+10)+0.5, label='0.5')
    axs[index, 2].vlines(sfh.time[ninth_mass_index], 6.5, np.max(np.log10(sfh.Mstar_Half)+10)+0.5, color='red', label='0.9')
    axs[index, 2].set_xlabel("Time")
    axs[index, 2].set_ylabel("Mstar")
    axs[index, 2].legend()


def smoothen(x, y):
    # smoothen data with frac = 0.2 to adjuct if needed
    lowess = sm.nonparametric.lowess(y, x, frac=0.2)
    return lowess
