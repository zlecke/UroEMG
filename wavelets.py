import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d


def calculate_wavelets(size, J=10, sampling_rate=1000, q=1.45, r=1.959, scale=3/10):
    """

    Parameters
    ----------
    size : array_like
        Length of array.
    J : int, default=10
        Largest wavelet index to calculate.
    sampling_rate : int, default=10
        Sampling rate of input data.
    q : float, default=1.45
    r : float, default=1.959
    scale : float, default=0.3

    Returns
    -------
    F_psi_c : ndarray
        Wavelets in frequency space.
    cf : ndarray
        Center frequencies of each `F_psi_c` wavelet.
    time_res : ndarray
        Time resolution of each `F_psi_c` wavelet.
    bandwidth : ndarray
        Bandwidth of each `F_psi_c` wavelet.

    """
    frequencies = np.linspace(0, 500, size, dtype=complex)
    cf = np.empty(J+1, dtype=complex)
    F_psi = np.zeros((len(frequencies), J+1), dtype=complex)

    for j in range(J + 1):
        cf[j] = (1 / scale) * (j + q) ** r

        F_psi[:, j] = ((frequencies / cf[j]) ** (cf[j] * scale)) * np.exp(
                (((-1 * frequencies) / cf[j]) + 1) * (cf[j] * scale))

    ck = np.zeros(len(frequencies))

    for k in range(1, len(frequencies)):
        ck[k] = 1 / np.sqrt(np.sum(np.square(np.absolute(F_psi[k, :]))))

    d = [0.816, 0.912]
    d.extend([0.96] * (J - 1))

    F_psi_c = F_psi.copy()

    for j in range(J + 1):
        F_psi_c[:, j] *= ck * d[j]

    self_p_n = []
    time_res = []

    for i in range(J + 1):
        tmp_p_n, tmp_time_res = calculate_time_resolution(F_psi, J + 1, frequencies, cf, i, None)

        reiterate = True
        if np.isnan(tmp_time_res):
            reiterate = False

        while reiterate:
            last_time_res = tmp_time_res
            sigma = ((3 / 8) * tmp_time_res - 1) / 6
            tmp_p_n, tmp_time_res = calculate_time_resolution(F_psi, J + 1, frequencies, cf, i, sigma)

            if np.isclose(last_time_res, tmp_time_res) or np.isnan(tmp_time_res):
                reiterate = False

        self_p_n.append(tmp_p_n)
        time_res.append(tmp_time_res)
    time_res = np.asarray(time_res)
    power_spectrum = np.square(np.absolute(F_psi))
    bandwidth = []

    for j in range(J + 1):
        spline = UnivariateSpline(frequencies.real,
                                  power_spectrum[:, j] - power_spectrum[:, j].max() * np.exp(-1),
                                  s=0)
        r1, r2 = spline.roots()
        bandwidth.append(r2 - r1)

    return F_psi_c, cf, time_res, bandwidth


def calculate_time_resolution(wavelets, num_wavelets, frequencies, center_frequencies, ind, sigma):
    self_conv = [np.fft.ifftshift(np.fft.ifft(wavelets[:, ind] * wavelets[:, j],
                                              wavelets[:, ind].size * 2,
                                              norm='ortho')) for j in range(num_wavelets)]
    p_n = np.asarray([np.square(np.abs(self_conv[j])) for j in range(num_wavelets)])
    F_inv = np.asarray([np.pi
                        * 2j
                        * frequencies
                        * wavelets[:, ind]
                        * wavelets[:, j] for j in range(num_wavelets)])
    dvdt = np.asarray([np.fft.ifftshift(np.fft.ifft(F_inv[j],
                                                    wavelets[:, ind].size * 2,
                                                    norm='ortho')) for j in range(num_wavelets)])
    coef = np.asarray([(1 / (2 * np.pi * center_frequencies[j])) for j in range(num_wavelets)])
    p_n = np.asarray([p_n[j] + np.square(np.abs((coef[j] * dvdt[j]))) for j in range(num_wavelets)])

    if sigma is not None:
        p_n = gaussian_filter1d(p_n.sum(axis=0), sigma)
    else:
        p_n = p_n.sum(axis=0)

    spline = UnivariateSpline(np.arange(0, len(p_n)), p_n - p_n.max()*np.exp(-1), s=0)
    roots = spline.roots()
    if len(roots) == 2:
        r1, r2 = roots
    else:
        r1, r2 = np.nan, np.nan

    return p_n, r2 - r1


def calculate_wavelet_spectrum(data, wavelets, num_wavelets, frequencies, center_frequencies, time_res):
    p_n = np.zeros((num_wavelets, wavelets[:, 0].size * 2), dtype=complex)
    for j in range(num_wavelets):
        p_n[j] = np.pad(wavelets[:, j], ((0, wavelets[:, j].size), ))
        p_n[j] = p_n[j] * np.fft.fft(data, wavelets[:, 0].size * 2, norm='ortho')
        p_n[j] = np.fft.ifftshift(np.fft.ifft(p_n[j], wavelets[:, j].size * 2, norm='ortho'))
        p_n[j] = np.square(np.abs(p_n[j])) + np.square(np.abs(
                            (1 / (2 * np.pi * center_frequencies[j]))
                            * np.fft.ifftshift(np.fft.ifft(np.pi
                                                           * 2j
                                                           * frequencies
                                                           * np.fft.fft(data, norm='ortho')
                                                           * wavelets[:, j],
                                                           wavelets[:, j].size * 2,
                                                           norm='ortho'))))
        p_n[j] = gaussian_filter1d(p_n[j], ((3/8) * time_res[j] - 1)/6)

    return p_n[:-1, :wavelets[:, 0].size].T.real
