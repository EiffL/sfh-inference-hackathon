
import numpy as np

def find_halftime(gal_sfh):

  ''' compute the half mass and the half time of a galaxy 
        Input: the sfh
        Output: the half mass, the corresponding half time, and the number of degenerecies
  '''

    half_mass = min(gal_sfh.Mstar_Half, key=lambda x:abs(x-np.max(gal_sfh.Mstar_Half)/2.)) # find mass closest to the half mass
    half_mass_indices = np.where(gal_sfh.Mstar_Half == half_mass)[0]  # find the corresponding indices
    try:
    nb_soluce = len(half_mass_indices) # find the number of indices
    except Exception:
    nb_soluce = 1
    half_mass_index = half_mass_indices[0]  # chose the first index for the half mass
    half_time = gal_sfh.time[half_mass_index]  # find the corresponding half time
    return half_mass, half_time, nb_soluce


  return half_mass, half_time, len(nb_soluce)

def smoothen(x,y):
  # smoothen data with frac = 0.2 to adjuct if needed
  import statsmodels.api as sm
  lowess = sm.nonparametric.lowess(y, x, frac=0.2)
  return lowess