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