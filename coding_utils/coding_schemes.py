import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from felipe_utils import CodingFunctionsFelipe
from utils.torch_utils import zero_norm_t, norm_t
from felipe_utils.research_utils import signalproc_ops, np_utils
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse, smooth_codes

import scipy as sp


def poisson_noise_array(Signal, trials=1000):
    new_size = (trials,) + Signal.shape
    rng = np.random.default_rng()
    return rng.poisson(lam=Signal, size=new_size).astype(Signal.dtype)


class Coding(ABC):

    def __init__(self, n_tbins, total_laser_cycles=None, binomial=False, gated=False,
                 num_measures=None, t_domain=None, after=False, h_irf=None, account_irf=False,
                 win_duty=None):

        self.binomial = binomial
        self.after = after
        self.gated = gated
        self.n_tbins = n_tbins
        self.win_duty = win_duty
        self.update_irf(h_irf, t_domain, win_duty)
        self.account_irf = account_irf
        self.set_laser_cycles(total_laser_cycles)
        if self.correlations is None: self.set_coding_scheme(n_tbins)
        self.update_C_derived_params()
        self.set_num_measures(num_measures)

    def update_C_derived_params(self):
        # Store how many codes there are
        (self.n_tbins, self.n_functions) = (self.correlations.shape[-2], self.correlations.shape[-1])
        if (self.account_irf) and self.h_irf is not None:
            # self.decoding_C = signalproc_ops.circular_conv(self.C, self.h_irf[:, np.newaxis], axis=0)
            # self.decoding_C = signalproc_ops.circular_corr(self.C, self.h_irf[:, np.newaxis], axis=0)
            self.decode_corrfs = signalproc_ops.circular_corr(self.h_irf[:, np.newaxis], self.correlations, axis=0)
        else:
            self.decode_corrfs = self.correlations
        #self.zero_norm_corrfs = zero_norm_t(self.decode_corrfs)
        #self.norm_corrfs = norm_t(self.decode_corrfs)

    @abstractmethod
    def set_coding_scheme(self, n_tbins):
        pass

    @abstractmethod
    def encode(self, incident, trails):
        pass

    ''' Felipe's Code'''

    def zncc_reconstruction(self, intensities):
        norm_int = zero_norm_t(intensities, axis=-1)
        return np.matmul(self.zero_norm_corrfs, norm_int[..., np.newaxis]).squeeze(-1)

    def ncc_reconstruction(self, intensities):
        norm_int = norm_t(intensities, axis=-1)
        return np.matmul(self.norm_corrfs, norm_int[..., np.newaxis]).squeeze(-1)

    def update_irf(self, h_irf=None, t_domain=None, win_duty=None):
        # If nothing is given set to gaussian
        if (h_irf is None):
            print("hirf is NONE")
            if (t_domain is not None):
                self.h_irf = gaussian_pulse(t_domain, 0, self.sigma, circ_shifted=True)
            else:
                print('t_domain is None')
                self.h_irf = None
                return
        else:
            if h_irf.shape[0] == self.n_tbins:
                self.h_irf = h_irf.squeeze()
            else:
                w = h_irf
                x = np.arange(w.size)
                new_length = self.n_tbins
                new_x = np.linspace(x.min(), x.max(), new_length)
                new_y = sp.interpolate.interp1d(x, w, kind='cubic')(new_x)
                self.h_irf = new_y

        if self.win_duty is not None:
            dummy_var = np.zeros_like(np.expand_dims(self.h_irf, axis=-1))
            (self.h_irf, _) = np.squeeze(smooth_codes(np.expand_dims(self.h_irf, axis=-1), dummy_var, window_duty=win_duty))
        self.h_irf = self.h_irf / self.h_irf.sum()

    ''' Felipe's Code'''

    def get_rec_algo_func(self, rec_algo_id):
        rec_algo_func_name = '{}_reconstruction'.format(rec_algo_id)
        rec_algo_function = getattr(self, rec_algo_func_name, None)
        assert (
                rec_algo_function is not None), "Reconstruction algorithm {} is NOT available. Please choose from the following algos: {}".format(
            rec_algo_func_name, self.rec_algos_avail)
        return rec_algo_function

    ''' Felipe's Code'''

    def reconstruction(self, intensities, rec_algo_id='zncc', **kwargs):
        rec_algo_function = self.get_rec_algo_func(rec_algo_id)
        lookup = rec_algo_function(intensities, **kwargs)
        return lookup

    ''' Felipe's Code'''

    def max_peak_decoding(self, intensities, rec_algo_id='zncc', **kwargs):
        lookup = self.reconstruction(intensities, rec_algo_id, **kwargs)
        return np.argmax(lookup, axis=-1)

    ''' Felipe's Code'''

    def maxgauss_peak_decoding(self, intensities, gauss_sigma, rec_algo_id='zncc', **kwargs):
        lookup = self.reconstruction(intensities, rec_algo_id, **kwargs)
        return signalproc_ops.max_gaussian_center_of_mass_mle(lookup, sigma_tbins=gauss_sigma)

    def set_correlations(self, modfs, demodfs):
        self.correlations = np.fft.ifft(np.fft.fft(modfs, axis=0).conj() * np.fft.fft(demodfs, axis=0), axis=0).real

    def set_laser_cycles(self, input_cycles):
        if input_cycles is None:
            self.laser_cycles = None
            return
        assert self.binomial is True, 'To set laser cycles must be binomial poisson model'
        self.laser_cycles = input_cycles

    def set_num_measures(self, input_num):
        if input_num is None:
            self.num_measures = self.n_functions
            return
        self.num_measures = input_num

    def get_num_measures(self):
        return self.num_measures


class ContinuousWave(Coding):
    def __init__(self, split=False, **kwargs):
        self.split = split
        super().__init__(**kwargs)

    def encode(self, incident, trials):
        if self.binomial:
            return self.encode_binomial(incident, trials)
        else:
            return self.encode_poison(incident, trials)

    def encode_poison(self, incident, trials):
        a = np.copy(incident)
        b = np.copy(self.demodfs)
        if not self.after:
           # inc = incident[18, 0, :]
            if self.split == False:
                a = poisson_noise_array(a[:, 0, :], trials)
                #detected = a[100, 18, :]
                intent = np.einsum('mnp,pq->mnq', a, b)
            else:
                a = poisson_noise_array(a, trials)
                intent = np.einsum('mnqp,pq->mnq', a, b)
        else:
            if self.split == False:
                a = a[:, 0, :]
                intent = np.einsum('np,pq->nq', a, b)
            else:
                intent = np.einsum('nqp,pq->nq', a, b)
                #intent = np.einsum('ijj->i', result)
            intent = poisson_noise_array(intent, trials)

        return intent

    def encode_binomial(self, incident, trials):
        a = np.copy(incident)
        b = np.copy(self.demodfs)
        if self.split == False:
            a = a[:, 0, :]
            photons = np.einsum('np,pq->nq', a, b)
        else:
            photons = np.einsum('nqp,pq->nq', a, b)

        probabilities = 1 - np.exp(-photons)
        rng = np.random.default_rng()
        new_shape = (trials,) + probabilities.shape
        num_m = None
        if self.gated is True:
            num_m = self.num_measures
        else:
            num_m = 1
        photon_counts = rng.binomial(int(self.laser_cycles / num_m), probabilities, size=new_shape)
        return photon_counts


class ImpulseCoding(Coding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, incident, trials):
        if self.binomial:
            return self.encode_binomial(incident, trials)
        else:
            return self.encode_poison(incident, trials)

    def encode_poison(self, incident, trials):
        a = np.copy(np.squeeze(incident))
        b = np.copy(self.correlations)
        a = poisson_noise_array(a, trials)
        #detected = a[100, 18, :]

        intent = np.einsum('mnp,pq->mnq', a, b)
        return intent

    def encode_binomial(self, incident, trials):

        rng = np.random.default_rng()
        new_shape = (trials,) + incident.shape
        probabilities = 1 - np.exp(-incident)
        incident_noisy = rng.binomial(int(self.laser_cycles), probabilities, size=new_shape)

        c_vals = np.matmul(incident_noisy[..., np.newaxis, :], self.correlations).squeeze(-2)

        return c_vals


class KTapSinusoidCoding(ContinuousWave):

    def __init__(self, ktaps, **kwargs):
        if (ktaps is None): ktaps = 3
        self.n_functions = ktaps
        self.correlations = None
        self.sigma = 1
        super().__init__(**kwargs)

    def set_coding_scheme(self, n_tbins):
        (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetCosCos(N=n_tbins, K=self.n_functions)
        self.set_correlations(self.modfs, self.demodfs)


class HamiltonianCoding(ContinuousWave):
    def __init__(self, k, duty=None, win_duty=None, **kwargs):
        self.n_functions = k
        self.set_duty(duty)
        self.win_duty = win_duty
        self.correlations = None
        self.sigma = 1
        super().__init__(**kwargs)

    def set_duty(self, duty):
        if duty is None:
            if self.n_functions == 3:
                self.duty = 1. / 6.
            elif self.n_functions == 4:
                self.duty = 1. / 12.
            elif self.n_functions == 5:
                self.duty = 1. / 30.
            else:
                assert False
        else:
            self.duty = duty

    def set_num_measures(self, input_num):
        if input_num is None and self.gated:
            if self.n_functions == 3:
                self.num_measures = 4
            elif self.n_functions == 4:
                self.num_measures = 7
            elif self.n_functions == 5:
                self.num_measures = 16
            else:
                assert False, 'not implemented for k>5'
            return
        elif input_num is None:
            self.num_measures = self.n_functions
        else:
            self.num_measures = input_num

    def set_coding_scheme(self, n_tbins):
        k = self.n_functions
        if (k == 3):
            (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetHamK3(n_tbins, modDuty=self.duty)
        elif (k == 4):
            (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetHamK4(n_tbins, modDuty=self.duty)
        elif (k == 5):
            (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetHamK5(n_tbins, modDuty=self.duty)
        else:
            assert False

        if self.win_duty is not None:
            (self.modfs, self.demodfs) = signalproc_ops.smooth_codes(self.modfs, self.demodfs, window_duty=self.win_duty)

        # if self.account_irf:
        #     self.modfs = np.repeat(np.expand_dims(self.h_irf, axis=-1), k, axis=-1)

        self.set_correlations(self.modfs, self.demodfs)


class ModifiedHamiltonianCoding(ContinuousWave):
    def __init__(self, k, duty=None, win_duty=None, **kwargs):
        self.n_functions = k
        self.set_duty(duty)
        self.win_duty = win_duty
        self.correlations = None
        self.sigma = 1
        super().__init__(**kwargs)

    def set_duty(self, duty):
        if duty is None:
            if self.n_functions == 3:
                self.duty = 1. / 6.
            elif self.n_functions == 4:
                self.duty = 1. / 12.
            elif self.n_functions == 5:
                self.duty = 1. / 30.
            else:
                assert False
        else:
            self.duty = duty


    def set_coding_scheme(self, n_tbins):
        k = self.n_functions
        if (k == 3):
            (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetHamK3(n_tbins)
        elif (k == 4):
            (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetHamK4(n_tbins)
        elif (k == 5):
            (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetHamK5(n_tbins)
        else:
            assert False

        if self.win_duty is not None:
            (self.modfs, _) = signalproc_ops.smooth_codes(self.modfs, self.demodfs, window_duty=self.win_duty)

        # if self.account_irf:
        #     self.modfs = np.repeat(np.expand_dims(self.h_irf, axis=-1), k, axis=-1)

        self.set_correlations(self.modfs, self.demodfs)
        self.demodfs = np.exp(self.correlations)
        (self.modfs, _) = CodingFunctionsFelipe.GetHamK3(n_tbins, modDuty=self.duty)
        self.set_correlations(self.modfs, self.demodfs)



class GatedCoding(Coding):
    '''
        Gated coding class. Coding is applied like a gated camera or a coarse histogram in SPADs
        In the extreme case that we have a gate for every time bin then the C matrix is an (n_maxres x n_maxres) identity matrix
    '''

    def __init__(self, sigma=1, n_gates=None, **kwargs):
        self.n_gates = n_gates
        self.correlations = None
        self.sigma = sigma
        super().__init__(**kwargs)

    def encode(self, incident, trials):
        if self.binomial:
            return self.encode_binomial(incident, trials)
        else:
            return self.encode_poison(incident, trials)

    def set_coding_scheme(self, n_tbins):
        if self.n_gates == None:
            self.n_gates = n_tbins
        n_gates = self.n_gates
        self.gate_len = int(n_tbins / n_gates)
        self.correlations = np.zeros((n_tbins, n_gates))
        for i in range(n_gates):
            start_tbin = i * self.gate_len
            end_tbin = start_tbin + self.gate_len
            self.correlations[start_tbin:end_tbin, i] = 1.

    def encode_poison(self, transient_img, trials):
        '''
        Encode the transient image using the n_codes inside the self.C matrix
        For GatedCoding with many n_gates, encoding through matmul is quite slow, so we assign it differently
        '''
        assert (transient_img.shape[-1] == self.n_tbins), "Input c_vec does not have the correct dimensions"
        #inc = transient_img[18, :]
        transient_img = poisson_noise_array(transient_img, trials)
        #detected = np.squeeze(transient_img[100, 18, :])
        #hist = transient_img[100, 18, :]
        c_vals = np.array(transient_img[..., 0::self.gate_len])
        for i in range(self.gate_len - 1):
            start_idx = i + 1
            c_vals += transient_img[..., start_idx::self.gate_len]
        return c_vals


    def encode_binomial(self, transient_img, trials):
        assert (transient_img.shape[-1] == self.n_tbins), "Input c_vec does not have the correct dimensions"
        photons = np.array(transient_img[..., 0::self.gate_len])
        for i in range(self.gate_len - 1):
            start_idx = i + 1
            photons += transient_img[..., start_idx::self.gate_len]

        probabilities = 1 - np.exp(-photons)
        rng = np.random.default_rng()
        new_shape = (trials,) + probabilities.shape

        if not self.gated:
            num_measures = 1
        else:
            num_measures = self.n_gates

        photon_counts = rng.binomial(int(self.laser_cycles / num_measures), probabilities, size=new_shape)

        return photon_counts

    def matchfilt_reconstruction(self, c_vals):
        template = self.h_irf
        zn_template = zero_norm_t(template, axis=-1)
        zn_c_vals = zero_norm_t(c_vals, axis=-1)
        shifts = signalproc_ops.circular_matched_filter(zn_c_vals, zn_template)
        # vectorize tensors
        (c_vals, c_vals_shape) = np_utils.vectorize_tensor(c_vals, axis=-1)
        shifts = shifts.reshape((c_vals.shape[0],))
        h_rec = np.zeros(c_vals.shape, dtype=template.dtype)
        for i in range(shifts.size): h_rec[i, :] = np.roll(template, shift=shifts[i], axis=-1)
        c_vals = c_vals.reshape(c_vals_shape)
        return h_rec.reshape(c_vals_shape)

    def linear_reconstruction(self, c_vals):
        if (self.n_gates == self.n_tbins): return c_vals
        if (self.account_irf):
            print(
                "Warning: Linear Reconstruction in Gated does not account for IRF, so unless the IRF spreads across time bins, this will produce quantized depths")
        x_fullres = np.arange(0, self.n_tbins)
        # Create a circular x axis by concatenating the first element to the end and the last element to the begining
        circular_x_lres = np.arange((0.5 * self.gate_len) - 0.5 - self.gate_len, self.n_tbins + self.gate_len,
                                    self.gate_len)
        circular_c_vals = np.concatenate((c_vals[..., -1][..., np.newaxis], c_vals, c_vals[..., 0][..., np.newaxis]),
                                         axis=-1)
        f = interpolate.interp1d(circular_x_lres, circular_c_vals, axis=-1, kind='linear')
        return f(x_fullres)


class IdentityCoding(GatedCoding):
    '''
        Identity coding class. GatedCoding in the extreme case where n_maxres == n_gates
    '''

    def __init__(self, n_tbins, **kwargs):
        super().__init__(n_tbins=n_tbins, **kwargs)


class GrayCoding(ImpulseCoding):
    def __init__(self, n_tbins, sigma, n_bits, **kwargs):
        self.max_n_bits = int(np.floor(np.log2(n_tbins)))
        self.n_bits = np.min((n_bits, self.max_n_bits))
        if (n_bits > self.max_n_bits):
            print("not using n_bits={}, using n_bits={} which is the max_n_bits for {} bins".format(n_bits,
                                                                                                    self.max_n_bits,
                                                                                                    n_tbins))
        self.correlations = None
        self.sigma = sigma
        super().__init__(n_tbins=n_tbins, **kwargs)

    def np_gray_code(self, n_bits):
        return signalproc_ops.generate_gray_code(n_bits)

    def set_coding_scheme(self, n_tbins):
        self.correlations = np.zeros((self.n_tbins, self.n_bits))
        self.gray_codes = self.np_gray_code(self.n_bits)
        self.min_gray_code_length = self.gray_codes.shape[0]

        if ((n_tbins % self.min_gray_code_length) != 0):
            print(
                "WARNING: Gray codes where the n_maxres is not a multiple of the gray code length, may have some small ambiguous regions")
        self.x_fullres = np.arange(0, self.n_tbins) * (1. / self.n_tbins)
        self.x_lowres = np.arange(0, self.min_gray_code_length) * (1. / self.min_gray_code_length)
        ext_x_lowres = np.arange(-1, self.min_gray_code_length + 1) * (1. / self.min_gray_code_length)
        ext_gray_codes = np.concatenate(
            (self.gray_codes[-1, :][np.newaxis, :], self.gray_codes, self.gray_codes[0, :][np.newaxis, :]), axis=0)
        f = interpolate.interp1d(ext_x_lowres, ext_gray_codes, axis=0, kind='linear')
        self.correlations = f(self.x_fullres)
        self.correlations = (self.correlations * 2) - 1
        self.correlations = self.correlations - self.correlations.mean(axis=-2, keepdims=True)
        print(f'Gray coding K={self.correlations.shape[-1]}')



class FourierCoding(ImpulseCoding):
    def __init__(self, n_tbins, sigma, freq_idx=[0, 1], n_codes=None, **kwargs):
        self.n_codes = n_codes
        self.freq_idx = freq_idx
        self.sigma = sigma
        self.correlations = None
        super().__init__(n_tbins=n_tbins, **kwargs)
        self.lres_mode = False
        self.lres_factor = 10
        self.lres_n = int(np.floor(n_tbins / self.lres_factor))
        self.lres_n_freqs = self.lres_n // 2

    def init_coding_mat(self, n_tbins, freq_idx):
        self.n_maxfreqs = n_tbins // 2
        self.freq_idx = np_utils.to_nparray(freq_idx)
        self.n_freqs = self.freq_idx.size
        self.max_n_sinusoid_codes = self.k * self.n_freqs
        if (self.n_codes is None):
            self.n_sinusoid_codes = self.max_n_sinusoid_codes
        else:
            if (self.n_codes > self.max_n_sinusoid_codes): warnings.warn(
                "self.n_codes is larger than max_n_sinusoid_codes, truncating number of codes to max_n_sinusoid_codes")
            self.n_sinusoid_codes = np.min([self.max_n_sinusoid_codes, self.n_codes])
        # Check input args
        assert (self.freq_idx.ndim == 1), "Number of dimensions for freq_idx should be 1"
        assert (self.n_freqs <= (
                self.n_tbins // 2)), "Number of frequencies cannot exceed the number of points at the max resolution"
        # Initialize and populate the matrix with zero mean sinusoids
        self.correlations = np.zeros((self.n_tbins, self.n_sinusoid_codes))

    def set_coding_scheme(self, n_tbins):
        '''
        Initialize all frequencies
        '''
        self.k = 2
        self.init_coding_mat(n_tbins, self.freq_idx)
        domain = np.arange(0, n_tbins) * ((2 * np.pi) / n_tbins)
        fourier_mat = signalproc_ops.get_fourier_mat(n=n_tbins, freq_idx=self.freq_idx)
        for i in range(self.n_sinusoid_codes):
            if ((i % 2) == 0):
                self.correlations[:, i] = fourier_mat[:, i // 2].real
            else:
                self.correlations[:, i] = fourier_mat[:, i // 2].imag
        # self.C[:, 0::2] = fourier_mat.real
        # self.C[:, 1::2] = fourier_mat.imag
        print(f'Fourier coding K={self.correlations.shape[-1]}')

    def get_n_maxfreqs(self):
        if (self.lres_mode):
            return self.lres_n_freqs
        else:
            return self.n_maxfreqs

    def construct_phasor(self, incident):
        return incident[..., 0::2] - 1j * incident[..., 1::2]

    def construct_fft_transient(self, incident):
        fft_transient = np.zeros(incident.shape[0:-1] + (self.get_n_maxfreqs(),), dtype=np.complex64)
        # Set the correct frequencies to the correct value
        fft_transient[..., self.freq_idx] = self.construct_phasor(incident)
        return fft_transient

    def ifft_reconstruction(self, incident):
        '''
        Use ZNCC to approximately reconstruct the signal encoded by c_vec
        '''
        fft_transient = self.construct_fft_transient(incident)
        # Finally return the IFT
        return np.fft.irfft(fft_transient, axis=-1, n=self.n_tbins)


class TruncatedFourierCoding(FourierCoding):
    def __init__(self, n_tbins, sigma, n_freqs=1, n_codes=None, include_zeroth_harmonic=False, **kwargs):
        if (not (n_codes is None) and (n_codes > 1)): n_freqs = np.ceil(n_codes / 2)
        freq_idx = np.arange(0, n_freqs + 1)
        # Remove zeroth harmonic if needed.
        if (not include_zeroth_harmonic): freq_idx = freq_idx[1:]
        self.include_zeroth_harmonic = include_zeroth_harmonic
        super().__init__(n_tbins=n_tbins, sigma=sigma, freq_idx=freq_idx, n_codes=n_codes, **kwargs)

    def ifft_reconstruction(self, c_vec):
        phasors = self.construct_phasor(c_vec)
        # if not available append zeroth harmonic
        if (not self.include_zeroth_harmonic):
            phasors = np.concatenate((np.zeros(phasors.shape[0:-1] + (1,), dtype=phasors.dtype), phasors), axis=-1)
        # Finally return the IFT
        return np.fft.irfft(phasors, axis=-1, n=self.n_tbins)


class GrayTruncatedFourierCoding(ImpulseCoding):
    def __init__(self, n_tbins, sigma, n_codes, **kwargs):
        # Create Gray coding obj
        self.gray_coding_obj = GrayCoding(n_tbins, sigma, n_codes)
        self.n_gray_codes = self.gray_coding_obj.n_bits
        # Create Fourier Coding obj with remaining codes we want to use
        self.n_fourier_codes = n_codes - self.n_gray_codes
        self.n_freqs = int(np.ceil(float(self.n_fourier_codes) / 2.))
        if (self.n_freqs >= 1):
            self.truncfourier_coding_obj = TruncatedFourierCoding(n_tbins, n_freqs=self.n_freqs,
                                                                  sigma=sigma, include_zeroth_harmonic=False)
        else:
            self.truncfourier_coding_obj = None
        # Set coding mat by concat the matrices
        self.correlations = None
        self.sigma = sigma
        super().__init__(n_tbins=n_tbins, **kwargs)

    def set_coding_scheme(self, n_tbins):
        gray_C = self.gray_coding_obj.correlations
        if (self.n_fourier_codes >= 1):
            truncfourier_C = self.truncfourier_coding_obj.correlations[:, 0:self.n_fourier_codes]
            self.correlations = np.concatenate((gray_C, truncfourier_C), axis=-1)
        else:
            self.correlations = gray_C

