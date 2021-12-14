"""Functions to perform spectral analysis.

"""

import logging
from typing import Tuple, Union

import numpy as np
from scipy.signal.windows import dpss, hann
from sklearn.decomposition import non_negative_factorization
from tqdm import trange
from dynemo import array_ops
from dynemo.analysis import regression
from dynemo.analysis.time_series import get_mode_time_series
from dynemo.data import manipulation

_logger = logging.getLogger("DyNeMo")


def coherence_spectra(power_spectra: np.ndarray, print_message: bool = True):
    """Calculates coherences from (cross) power spectral densities.

    Parameters
    ----------
    power_spectra : np.ndarray
        Power spectra. Shape is (n_modes, n_channels, n_channels, n_f).
    print_message : bool
        Should we print a message to screen? Optional.

    Returns
    -------
    np.ndarray
        Coherence spectra for each mode.
        Shape is (n_modes, n_channels, n_channels, n_f).
    """
    n_modes, n_channels, n_channels, n_f = power_spectra.shape

    if print_message:
        print("Calculating coherences")

    coherences = np.empty([n_modes, n_channels, n_channels, n_f])
    for i in range(n_modes):
        for j in range(n_channels):
            for k in range(n_channels):
                coherences[i, j, k] = abs(power_spectra[i, j, k]) / np.sqrt(
                    power_spectra[i, j, j].real * power_spectra[i, k, k].real
                )

    return coherences


def decompose_spectra(
    coherences: np.ndarray,
    n_components: int,
    max_iter: int = 50000,
    random_state: int = None,
    verbose: int = 0,
) -> np.ndarray:
    """Performs spectral decomposition using coherences.

    Uses non-negative matrix factorization to decompose spectra.
    Follows the same procedure as the OSL function HMM-MAR/spectral/spectdecompose.m

    Parameters
    ----------
    coherences : np.ndarray
        Coherences spectra.
    n_components : int
        Number of spectral components to fit.
    max_iter : int
        Maximum number of iterations in sklearn's non_negative_factorization.
    random_state : int
        Seed for the random number generator.
    verbose : int
        Show verbose? (1) yes, (0) no.

    Returns
    -------
    components : np.ndarray
        Spectral components. Shape is (n_components, n_f).
    """
    print("Performing spectral decomposition")

    # Validation
    error_message = (
        "coherences must be a numpy array with shape "
        + "(n_channels, n_channels, n_f), "
        + "(n_modes, n_channels, n_channels, n_f) or "
        + "(n_subjects, n_modes, n_channels, n_channels, n_f)."
    )
    coherences = array_ops.validate(
        coherences,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    # Number of subjects, modes, channels and frequency bins
    n_subjects, n_modes, n_channels, n_channels, n_f = coherences.shape

    # Indices of the upper triangle of the [n_channels, n_channels, n_f] sub-array
    i, j = np.triu_indices(n_channels, 1)

    # Concatenate coherences for each subject and mode and only keep the upper triangle
    coherences = coherences[:, :, i, j].reshape(-1, n_f)

    # Perform non-negative matrix factorisation
    _, components, _ = non_negative_factorization(
        coherences,
        n_components=n_components,
        init=None,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )

    # Order the weights and components in ascending frequency
    order = np.argsort(components.argmax(axis=1))
    components = components[order]

    return components


def fourier_transform(
    data: np.ndarray,
    nfft: int,
    args_range: list = None,
    one_side: bool = False,
) -> np.ndarray:
    """Calculates a Fast Fourier Transform (FFT).

    Parameters
    ----------
    data : np.ndarray
        Data with shape (n_samples, n_channels) to FFT.
    nfft : int
        Number of points in the FFT.
    args_range : list
        Minimum and maximum indices of the FFT to keep. Optional.
    one_side : bool
        Should we return a one-sided FFT? Optional.

    Returns
    -------
    np.ndarray
        FFT data.
    """

    # Calculate the FFT
    X = np.fft.fft(data, nfft)

    # Only keep the postive frequency side
    if one_side:
        X = X[..., : X.shape[-1] // 2]

    # Only keep the desired frequency range
    if args_range is not None:
        X = X[..., args_range[0] : args_range[1]]

    return X


def get_frequency_args_range(frequencies: np.ndarray, frequency_range: list) -> list:
    """Get min/max indices of a range in a frequency axis.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis.
    frequency_range : list of len 2
        Min/max frequency.

    Returns
    -------
    list of len 2
        Min/max index.
    """
    f_min_arg = np.argwhere(frequencies >= frequency_range[0])[0, 0]
    f_max_arg = np.argwhere(frequencies <= frequency_range[1])[-1, 0]
    if f_max_arg <= f_min_arg:
        raise ValueError("Cannot select requested frequency range.")
    args_range = [f_min_arg, f_max_arg + 1]
    return args_range


def mar_spectra(
    coeffs: np.ndarray, covs: np.ndarray, sampling_frequency: float, n_f: int = 512
) -> np.ndarray:
    """Calculates cross power spectral densities from MAR model parameters.

    Parameters
    ----------
    coeffs : np.ndarray
        MAR coefficients.
    covs : np.ndarray
        MAR covariances.
    sampling_frequency : float
        Sampling frequency in Hertz.
    n_f : int
        Number of frequency bins in the cross power spectral density to calculate.
        Optional.

    Returns
    -------
    np.ndarray
        Cross power spectral densities.
        Shape is (n_f, n_modes, n_channels, n_channels) or
        (n_f, n_channels, n_channels).
    """
    # Validation
    if covs.shape[-1] != covs.shape[-2]:
        if covs.ndim == 2:
            covs = [np.diag(c) for c in covs]
        else:
            covs = np.diag(covs)
    error_message = (
        "covs must be a numpy array with shape "
        + "(n_modes, n_channels, n_channels), "
        + "(n_channels, n_channels) or (n_channels,)."
    )
    covs = array_ops.validate(
        covs,
        correct_dimensionality=3,
        allow_dimensions=[2],
        error_message=error_message,
    )
    error_message = (
        "coeffs must be a numpy array with shape "
        + "(n_modes, n_lags, n_channels, n_channels), "
        + "(n_lags, n_channels, n_channels)."
    )
    coeffs = array_ops.validate(
        coeffs,
        correct_dimensionality=4,
        allow_dimensions=[3],
        error_message=error_message,
    )

    n_modes = coeffs.shape[0]
    n_lags = coeffs.shape[1]
    n_channels = coeffs.shape[-1]

    # Frequencies to evaluate the PSD at
    f = np.arange(0, sampling_frequency / 2, sampling_frequency / (2 * n_f))

    # z-transform of the coefficients
    A = np.zeros([n_f, n_modes, n_channels, n_channels], dtype=np.complex_)
    for i in range(n_f):
        for l in range(n_lags):
            z = np.exp(-1j * (l + 1) * 2 * np.pi * f[i] / sampling_frequency)
            A[i] += coeffs[:, l] * z

    # Transfer function
    I = np.identity(n_channels)[np.newaxis, np.newaxis, ...]
    H = np.linalg.inv(I - A)

    # PSDs
    P = H @ covs[np.newaxis, ...] @ np.transpose(np.conj(H), axes=[0, 1, 3, 2])

    return f, np.squeeze(P)


def mode_covariance_spectra(
    autocorrelation_function: np.ndarray,
    sampling_frequency: float,
    nfft: int = 64,
    frequency_range: list = None,
):
    """Calculates spectra from the autocorrelation function.

    The power spectrum of each mode is calculated as the Fourier transform of
    the auto-correlation function. Coherences are calculated from the power spectra.

    Parameters
    ----------
    autocorrelation_function : np.ndarray
        Mode autocorrelation functions.
        Shape must be (n_modes, n_channels, n_channels, n_acf).
    sampling_frequency : float
        Frequency at which the data was sampled (Hz).
    nfft : int
        Number of data points in the FFT. The auto-correlation function will only
        have 2 * (n_embeddings + 2) - 1 data points. We pad the auto-correlation
        function with zeros to have nfft data points if the number of data points
        in the auto-correlation function is less than nfft. Default is 64.
    frequency_range : list
        Minimum and maximum frequency to keep (Hz).

    Returns
    -------
    frequencies : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_f,).
    power_spectra : np.ndarray
        Power (or cross) spectra calculated for each mode. Shape is (n_modes,
        n_channels, n_channels, n_f).
    coherences : np.ndarray
        Coherences calculated for each mode. Shape is (n_modes, n_channels,
        n_channels, n_f).
    """
    print("Calculating power spectra")

    # Validation
    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    # Number of data points in the autocorrelation function and FFT
    n_acf = autocorrelation_function.shape[-1]
    nfft = max(nfft, 2 ** nextpow2(n_acf))

    # Calculate the argments to keep for the given frequency range
    frequencies = np.arange(0, sampling_frequency / 2, sampling_frequency / nfft)
    args_range = get_frequency_args_range(frequencies, frequency_range)
    frequencies = frequencies[args_range[0] : args_range[1]]

    # Calculate cross power spectra as the Fourier transform of the
    # auto/cross-correlation function
    power_spectra = abs(fourier_transform(autocorrelation_function, nfft, args_range))

    # Normalise the power spectra
    power_spectra /= nfft ** 2

    # Coherences for each mode
    coherences = coherence_spectra(power_spectra)

    return frequencies, np.squeeze(power_spectra), np.squeeze(coherences)


def multitaper(
    data: np.ndarray,
    sampling_frequency: float,
    nfft: int = None,
    tapers: np.ndarray = None,
    time_half_bandwidth: float = None,
    n_tapers: int = None,
    args_range: list = None,
) -> np.ndarray:
    """Calculates a power (or cross) spectral density using the multitaper method.

    Parameters
    ----------
    data : np.ndarray
        Data with shape (n_samples, n_channels) to calculate a multitaper for.
    sampling_frequency : float
        Frequency used to sample the data (Hz).
    nfft : int
        Number of points in the FFT.
    tapers : np.ndarray
        Taper functions.
    time_half_bandwidth : float
        Parameter to control the resolution of the multitaper.
    n_tapers : int
        Number of tapers.
    args_range : list
        Minimum and maximum indices of the multitaper to keep.

    Returns
    -------
    np.ndarray
        Power (or cross) spectral density with shape (n_channels, n_channels, n_f).
    """

    # Transpose the data so that it is [n_channels, n_samples]
    data = np.transpose(data)

    # Number of channels and length of each signal
    n_channels, n_samples = data.shape

    # Number of FFT data points to calculate
    if nfft is None:
        nfft = max(256, 2 ** nextpow2(n_samples))

    # If tapers are not passed we generate them here
    if tapers is None:

        # Check the time half width bandwidth and number of tapers has been passed
        if time_half_bandwidth is None or n_tapers is None:
            raise ValueError("time_half_bandwidth and n_tapers must be passed.")

        # Calculate tapers
        tapers = dpss(n_samples, NW=time_half_bandwidth, Kmax=n_tapers)
        tapers *= np.sqrt(sampling_frequency)

    else:
        # Get number of tapers from the tapers passed
        n_tapers = len(tapers)

    # Multiply the data by the tapers
    data = data[np.newaxis, :, :] * tapers[:, np.newaxis, :]

    # Calculate the FFT, X, which has shape [n_tapers, n_channels, n_f]
    X = fourier_transform(data, nfft, args_range)
    X /= sampling_frequency

    # Number of frequency bins in the FFT
    n_f = X.shape[-1]

    # Calculate the periodogram with each taper
    P = np.zeros([n_channels, n_channels, n_f], dtype=np.complex_)
    for i in range(n_tapers):
        for j in range(n_channels):
            for k in range(j, n_channels):
                P[j, k] += np.conjugate(X[i, j]) * X[i, k]
                if i == n_tapers - 1 and k != j:
                    P[k, j] = np.conjugate(P[j, k])

    return P


def multitaper_spectra(
    data: Union[np.ndarray, list],
    alpha: Union[np.ndarray, list],
    sampling_frequency: float,
    time_half_bandwidth: float,
    n_tapers: int,
    segment_length: int = None,
    frequency_range: list = None,
    return_weights: bool = False,
    standardize: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Calculates spectra for inferred modes using a multitaper.

    This includes power and coherence spectra.
    Follows the same procedure as the OSL function HMM-MAR/spectral/hmmspectamt.m

    Parameters
    ----------
    data : np.ndarray or list
        Raw time series data. Must have shape (n_subjects, n_samples, n_channels)
        or (n_samples, n_channels).
    alpha : np.ndarray or list
        Inferred mode mixing factors. Must have shape (n_subjects, n_samples, n_modes)
        or (n_samples, n_modes).
    sampling_frequency : float
        Sampling frequency in Hz.
    time_half_bandwidth : float
        Parameter to control the resolution of the spectra.
    n_tapers : int
        Number of tapers to use when calculating the multitaper.
    segment_length : int
        Length of the data segement to use to calculate the multitaper.
    frequency_range : list
        Minimum and maximum frequency to keep.
    return_weights : bool
        Should we return the weights for subject-specific PSDs?
        Useful for calculating the group average PSD. Optional.
    standardize : bool
        Should we standardize the data before calculating the multitaper?
        Optional.

    Returns
    -------
    frequencies : np.ndarray
        Frequencies of the power spectra and coherences.
    power_spectra : np.ndarray
        Power spectra for each mode.
    coherences : np.ndarray
        Coherences for each mode.
    weights : np.ndarray
        Weight for each subject-specific PSD. Only returned if return_weights=True.
    """

    # Validation
    if (isinstance(data, list) != isinstance(alpha, list)) or (
        isinstance(data, np.ndarray) != isinstance(alpha, np.ndarray)
    ):
        raise ValueError(
            f"data is type {type(data)} and alpha is type "
            + f"{type(alpha)}. They must both be lists or numpy arrays."
        )

    if isinstance(data, list):
        # Check data and mode mixing factors for the same number of subjects
        # has been passed
        if len(data) != len(alpha):
            raise ValueError(
                "A different number of subjects has been passed for "
                + f"data and alpha: len(data)={len(data)}, "
                + f"len(alpha)={len(alpha)}."
            )

        # Check the number of samples in data and alpha
        for i in range(len(alpha)):
            if alpha[i].shape[0] != data[i].shape[0]:
                raise ValueError("items in data and alpha must have the same shape.")

    if isinstance(data, np.ndarray):
        if alpha.shape[0] != data.shape[0]:
            raise ValueError("data and alpha must have the same shape.")
        data = [data]
        alpha = [alpha]

    if segment_length is None:
        segment_length = 2 * sampling_frequency

    elif segment_length != 2 * sampling_frequency:
        _logger.warning("segment_length is recommended to be 2 * sampling_frequency.")

    segment_length = int(segment_length)

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    # Number of subjects
    n_subjects = len(data)

    # Number of FFT data points to calculate
    nfft = max(256, 2 ** nextpow2(segment_length))

    # Calculate the argments to keep for the given frequency range
    frequencies = np.arange(0, sampling_frequency / 2, sampling_frequency / nfft)
    args_range = get_frequency_args_range(frequencies, frequency_range)
    frequencies = frequencies[args_range[0] : args_range[1]]

    # Number of frequency bins
    n_f = args_range[1] - args_range[0]

    # Standardise before calculating the multitaper
    if standardize:
        data = [manipulation.standardize(d) for d in data]

    print("Calculating power spectra")
    power_spectra = []
    coherences = []
    for n in range(n_subjects):
        if n_subjects > 1:
            print(f"Subject {n}:")

        # Use the mode mixing factors to get a time series for each mode
        mode_time_series = get_mode_time_series(data[n], alpha[n])

        # Number of subjects, modes, samples and channels
        n_modes, n_samples, n_channels = mode_time_series.shape

        # Calculate tapers so we can estimate spectra with the multitaper method
        tapers = dpss(segment_length, NW=time_half_bandwidth, Kmax=n_tapers)
        tapers *= np.sqrt(sampling_frequency)

        # We will calculate the spectrum for several non-overlapping segments
        # of the time series and return the average over these segments.

        # Number of segments in the time series
        n_segments = round(n_samples / segment_length)

        # Power spectra for each mode
        p = np.zeros([n_modes, n_channels, n_channels, n_f], dtype=np.complex_)
        for i in range(n_modes):
            for j in trange(n_segments, desc=f"Mode {i}", ncols=98):

                # Time series for mode i and segment j
                time_series_segment = mode_time_series[
                    i, j * segment_length : (j + 1) * segment_length
                ]

                # If we're missing samples we pad with zeros either side of the data
                if time_series_segment.shape[0] != segment_length:
                    n_zeros = segment_length - time_series_segment.shape[0]
                    n_padding = n_zeros // 2
                    time_series_segment = np.pad(time_series_segment, n_padding)[
                        :, n_padding:-n_padding
                    ]

                # Calculate the power (and cross) spectrum using the multitaper method
                p[i] += multitaper(
                    time_series_segment,
                    sampling_frequency,
                    nfft=nfft,
                    tapers=tapers,
                    args_range=args_range,
                )

        # Normalise the power spectra
        # TODO: We should be normalising using sum alpha instead of sum alpha^2,
        # but this makes a small difference, so left like this for consistency
        # with HMM-MAR
        sum_alpha = np.sum(alpha[n] ** 2, axis=0)[
            ..., np.newaxis, np.newaxis, np.newaxis
        ]
        p *= n_samples / (sum_alpha * n_tapers * n_segments)

        # Coherences for each mode
        c = coherence_spectra(p, print_message=False)

        # Add spectra to lists
        power_spectra.append(p)
        coherences.append(c)

    # Weights for calculating the group average PSD
    n_samples = [d.shape[0] for d in data]
    weights = np.array(n_samples) / np.sum(n_samples)

    if return_weights:
        return frequencies, np.squeeze(power_spectra), np.squeeze(coherences), weights
    else:
        return frequencies, np.squeeze(power_spectra), np.squeeze(coherences)


def nextpow2(x: int) -> int:
    """Next power of 2.

    Parameters
    ----------
    x : int
        Any integer.

    Returns
    -------
    int
        The smallest power of two that is greater than or equal to the absolute
        value of x.
    """
    res = np.ceil(np.log2(x))
    return res.astype("int")


def regression_spectra(
    data: Union[np.ndarray, list],
    alpha: Union[np.ndarray, list],
    window_length: int,
    sampling_frequency: float = 1.0,
    frequency_range: list = None,
    n_embeddings: int = None,
    psd_only: bool = False,
    step_size: int = 1,
    n_sub_windows: int = 1,
    return_weights: bool = False,
    return_coef_int: bool = False,
    standardize: bool = False,
):
    """Calculates the PSD of each mode by regressing a time-varying PSD with alpha.

    Parameters
    ----------
    data : np.ndarray or list
        Data to calculate a time-varying PSD for. Shape must be (n_samples, n_channels).
    alpha : np.ndarray
        Inferred mode mixing factors. Shape must be (n_samples, n_modes).
    window_length : int
        Number samples to use in the window to calculate a PSD.
    sampling_frequency : float
        Sampling_frequency in Hz. Optional.
    frequency_range : list
        Minimum and maximum frequency to keep.
    n_embeddings : int
        Number of time embeddings applied when inferring alpha. Optional.
    psd_only : bool
        Should we only calculate the PSD? Optional.
    step_size : int
        Step size for shifting the window. Optional.
    n_sub_windows : int
        Should we split the window into a set of sub-windows and average the
        spectra for each sub-window. Optional.
    return_weights : bool
        Should we return the weights for subject-specific PSDs?
        Useful for calculating the group average PSD. Optional.
    return_coef_int : bool
        Should we return the regression coefficients and intercept
        separately for the PSDs? Optional.
    standardize : bool
        Should we standardize the data before calculating the spectrogram?
        Optional.

    Returns
    -------
    f : np.ndarray
        Frequency axis.
    psd : np.ndarray
        Mode PSDs. A numpy array with shape (n_subjects, 2, n_modes, n_channels, n_f)
        where the first axis is the coefficients/intercept if return_coef_int=True,
        otherwise shape is (n_subjects, n_modes, n_channels, n_f).
    coh : np.ndarray
        Mode coherences.
    w : np.ndarray
        Weight for each subject-specific PSD. Only returned if return_weights=True.
    """

    # Validation
    if isinstance(data, list):
        if not isinstance(alpha, list):
            raise ValueError(
                "data and alpha must both be lists or both be numpy arrays."
            )

    if isinstance(data, np.ndarray):
        if not isinstance(alpha, np.ndarray):
            raise ValueError(
                "data and alpha must both be lists or both be numpy arrays."
            )
        data = [data]
        alpha = [alpha]

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    # Do we calculate cross spectral densities?
    calc_cpsd = not psd_only

    # Number of subjects
    n_subjects = len(data)

    # Standardise before calculating the spectrogram
    if standardize:
        data = [manipulation.standardize(d) for d in data]

    # Remove data points not in alpha due to time embedding the training data
    if n_embeddings is not None:
        data = [d[n_embeddings // 2 : -(n_embeddings // 2)] for d in data]

    # Remove the data points lost due to separating into sequences
    data = [d[: a.shape[0]] for d, a in zip(data, alpha)]

    # Calculate a time-varying PSD
    Pt = []
    at = []
    for i in range(n_subjects):
        if n_subjects > 1:
            print(f"Subject {i}:")
        t, f, p, a = spectrogram(
            data[i],
            window_length,
            sampling_frequency,
            frequency_range,
            calc_cpsd=calc_cpsd,
            step_size=step_size,
            alpha=alpha[i],
            n_sub_windows=n_sub_windows,
        )
        Pt.append(p)
        at.append(a)

    # Regress the time-varying PSD with alpha to get the mode PSDs
    Pj = []
    for i in trange(n_subjects, desc="Fitting linear regression", ncols=98):
        Pt[i] = abs(Pt[i])  # sklearn LinearRegression doesn't accept complex numbers
        coefs, intercept = regression.linear(
            at[i], Pt[i], fit_intercept=True, normalize=True
        )
        if return_coef_int:
            Pj.append([coefs, intercept])
        else:
            Pj.append([coefs + intercept[np.newaxis, ...]])
    Pj = np.array(Pj, dtype=object)

    # Weights for calculating the group average PSD
    n_samples = [d.shape[0] for d in data]
    weights = np.array(n_samples) / np.sum(n_samples)

    if psd_only:
        if return_weights:
            return f, np.squeeze(Pj), weights
        else:
            return f, np.squeeze(Pj)

    # Number of parcels and freqency bins
    n_parcels = data[0].shape[1]
    n_modes = alpha[0].shape[1]
    n_f = Pj[0][0].shape[-1]

    # Indices of the upper triangle of an n_parcels by n_parcels array
    m, n = np.triu_indices(n_parcels)

    # Create a n_parcels by n_parcels array
    P = []
    for i in range(n_subjects):
        P.append([])
        for j in range(len(Pj[i])):
            p = np.empty([n_modes, n_parcels, n_parcels, n_f])
            p[:, m, n] = Pj[i][j]
            p[:, n, m] = Pj[i][j]
            P[i].append(p)
    P = np.array(P)

    # PSDs and coherences for each mode
    psd = []
    coh = []
    for i in range(n_subjects):
        p = P[i, :, :, range(n_parcels), range(n_parcels)]
        p = np.rollaxis(p, 0, 3)
        c = np.sum(P[i], axis=0)  # sum coefs and intercept
        c = coherence_spectra(c, print_message=False)
        psd.append(p)
        coh.append(c)

    if return_weights:
        return f, np.squeeze(psd), np.squeeze(coh), weights
    else:
        return f, np.squeeze(psd), np.squeeze(coh)


def spectrogram(
    data: np.ndarray,
    window_length: int,
    sampling_frequency: float = 1.0,
    frequency_range: list = None,
    calc_cpsd: bool = True,
    step_size: int = 1,
    alpha: np.ndarray = None,
    n_sub_windows: int = 1,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Calculates a spectogram.

    The data is segmented into overlapping windows which are then used to calculate
    a periodogram.

    Parameters
    ----------
    data : np.ndarray
        Data to calculate the spectrogram for. Shape must be (n_samples, n_channels).
    window_length : int
        Number of data points to use when calculating the periodogram.
    sampling_frequency : float
        Sampling frequency in Hz. Optional.
    calc_cpsd : bool
        Should we calculate cross spectra? Optional.
    step_size : int
        Step size for shifting the window. Optional.
    alpha : np.ndarray
        Alpha fitted to the data. Optional. Useful to pass if you want to regress
        the spectrogram with alpha.
    n_sub_windows : int
        Should we split the window into a set of sub-windows and average the
        spectra for each sub-window. Optional.

    Returns
    -------
    t : np.ndarray
        Time axis.
    f : np.ndarray
        Frequency axis.
    P : np.ndarray
        Spectrogram.
    a : np.ndarray
        Alpha values corresponding to each periodogram.
        Returned if alpha is passed.
    """

    # Number of samples, channels and modes
    n_samples = data.shape[0]
    n_channels = data.shape[1]
    if alpha is not None:
        n_modes = alpha.shape[1]

    # First pad the data so we have enough data points to estimate the periodogram
    # for time points at the start/end of the data
    data = np.pad(data, window_length // 2)[
        :, window_length // 2 : window_length // 2 + n_channels
    ]
    if alpha is not None:
        alpha = np.pad(alpha, window_length // 2)[
            :, window_length // 2 : window_length // 2 + n_modes
        ]

    # Window to apply to the data before calculating the Fourier transform
    window = hann(window_length // n_sub_windows)

    # Number of data points in the FFT
    nfft = max(256, 2 ** nextpow2(window_length // n_sub_windows))

    # Time and frequency axis
    t = np.arange(n_samples) / sampling_frequency
    f = np.arange(nfft // 2) * sampling_frequency / nfft

    # Only keep a particular frequency range
    args_range = get_frequency_args_range(f, frequency_range)
    f = f[args_range[0] : args_range[1]]

    # Number of frequency bins
    n_f = args_range[1] - args_range[0]

    # Indices of an upper triangle of an n_channels by n_channels array
    m, n = np.triu_indices(n_channels)

    # Indices of time points to calculate a periodogram for
    time_indices = range(0, n_samples, step_size)
    n_psds = n_samples // step_size

    if alpha is not None:
        # Array to hold mean of alpha multiplied by the windowing function
        a = np.empty([n_psds, n_modes])

    if calc_cpsd:
        # Calculate cross periodograms for each segment of the data
        P = np.empty(
            [n_psds, n_channels * (n_channels + 1) // 2, n_f], dtype=np.complex_
        )
        XY_sub_window = np.empty(
            [n_sub_windows, n_channels * (n_channels + 1) // 2, n_f], dtype=np.complex_
        )
        if alpha is not None:
            a_sub_window = np.empty([n_sub_windows, n_modes], dtype=np.float32)
        for i in trange(n_psds, desc="Calculating spectrogram", ncols=98):
            j = time_indices[i]

            # Data and alpha in the window
            x_window = data[j : j + window_length].T
            a_window = alpha[j : j + window_length]

            # Loop through sub-windows
            for k in range(n_sub_windows):

                # Data in the sub-window with the windowing function applied
                x_sub_window = (
                    x_window[
                        :,
                        k
                        * window_length
                        // n_sub_windows : (k + 1)
                        * window_length
                        // n_sub_windows,
                    ]
                    * window[np.newaxis, ...]
                )

                # Calculate cross spectra for the sub-window
                X = fourier_transform(x_sub_window, nfft, args_range)
                XY = X[:, np.newaxis, :] * np.conj(X)[np.newaxis, :, :]
                XY_sub_window[k] = XY[m, n]

                if alpha is not None:
                    # Calculate alpha for the sub-window by taking the mean
                    # over time after applying the windowing function
                    a_sub_window[k] = np.mean(
                        a_window[
                            k
                            * window_length
                            // n_sub_windows : (k + 1)
                            * window_length
                            // n_sub_windows
                        ]
                        * window[..., np.newaxis],
                        axis=0,
                    )

            # Average the cross spectra and alpha for each sub-window
            P[i] = np.mean(XY_sub_window, axis=0)
            a[i] = np.mean(a_sub_window, axis=0)

    else:
        # Calculate the periodogram for each segment of the data
        P = np.empty([n_psds, n_channels, n_f], dtype=np.float32)
        XX_sub_window = np.empty([n_sub_windows, n_channels, n_f], dtype=np.float32)
        if alpha is not None:
            a_sub_window = np.empty([n_sub_windows, n_modes], dtype=np.float32)
        for i in trange(n_psds, desc="Calculating spectrogram", ncols=98):
            j = time_indices[i]

            # Data and alpha in the window
            x_window = data[j : j + window_length].T
            if alpha is not None:
                a_window = alpha[j : j + window_length]

            # Loop through sub-windows
            for k in range(n_sub_windows):

                # Data in the sub-window with the windowing function applied
                x_sub_window = (
                    x_window[
                        :,
                        k
                        * window_length
                        // n_sub_windows : (k + 1)
                        * window_length
                        // n_sub_windows,
                    ]
                    * window[np.newaxis, ...]
                )

                # Calculate PSD for the sub-window
                X = fourier_transform(x_sub_window, nfft, args_range)
                XX_sub_window[k] = np.real(X * np.conj(X))

                if alpha is not None:
                    # Calculate alpha for the sub-window by taking the mean
                    # over time after applying the windowing function
                    a_sub_window[k] = np.mean(
                        a_window[
                            k
                            * window_length
                            // n_sub_windows : (k + 1)
                            * window_length
                            // n_sub_windows
                        ]
                        * window[..., np.newaxis],
                        axis=0,
                    )

            # Average the cross spectra and alpha for each sub-window
            P[i] = np.mean(XX_sub_window, axis=0)
            if alpha is not None:
                a[i] = np.mean(a_sub_window, axis=0)

    # Scaling for the periodograms
    P /= sampling_frequency * np.sum(window ** 2)

    if alpha is not None:
        return t, f, P, a
    else:
        return t, f, P
