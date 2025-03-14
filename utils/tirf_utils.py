import numpy as np
from coding_utils.coding_schemes import *


def get_coding_scheme(coding_id, n_tbins, k, h_irf):
    coding_obj = None
    if (coding_id == 'Greys'):
        coding_obj = GrayCoding(n_tbins=n_tbins, sigma=1,  n_bits=k, account_irf=True, h_irf=h_irf)
    elif (coding_id == 'Fourier'):
        coding_obj = FourierCoding(n_tbins=n_tbins, sigma=1,  n_codes=k, account_irf=True, h_irf=h_irf)

    elif (coding_id == 'TruncatedFourier'):
        coding_obj = TruncatedFourierCoding(n_tbins=n_tbins, sigma=1, n_codes=k, account_irf=True, h_irf=h_irf)

    elif (coding_id == 'GrayTruncatedFourier'):
        coding_obj = GrayTruncatedFourierCoding(n_tbins=n_tbins, sigma=1,  n_codes=k, account_irf=True, h_irf=h_irf)
    else:
        assert False, 'Coding ID WRONGGGG'
    return coding_obj.correlations

def get_irf(irf_filename, n_tbins=None):
    if irf_filename.endswith('.csv'):
        irf = np.genfromtxt(irf_filename, delimiter=',')
        irf = np.roll(irf, - np.argmax(irf))
    elif irf_filename.endswith('.npy'):
        irf = np.load(irf_filename)
    w = irf
    x = np.arange(w.size)
    if n_tbins is not None:
        new_length = n_tbins
    else:
        new_length = irf.shape[0]
    new_x = np.linspace(x.min(), x.max(), new_length)
    new_y = sp.interpolate.interp1d(x, w, kind='cubic')(new_x)
    new_y = new_y / np.sum(new_y)
    return new_y

