import math
from collections import namedtuple
from math import pi, atan, sqrt, acos, sin, cos

import numpy as np
from numpy.fft import fft2, ifftshift, ifft2


def _log_gabor(freqs: np.ndarray, fo: float, sigma_on_f: float) -> np.ndarray:
    """
    The logarithmic Gabor function in the frequency domain.
    sigma_on_f = 0.75 gives a filter bandwidth of about 1 octave.
    sigma_on_f = 0.55 gives a filter bandwidth of about 2 octaves.
    Args:
        freqs: Array of frequencies to evaluate the function at.
        fo: Centre frequency of filter.
        sigma_on_f: Ratio of the standard deviation of the Gaussian
                    describing the log Gabor filter's transfer function
                    in the frequency domain to the filter center frequency.

    Returns: Array of outputs of the logarithmic Gabor function.

    """
    out = np.zeros_like(freqs)
    nonzero = freqs >= np.finfo(np.float64).eps  # Everything below eps stays at default array value (zero)
    out[nonzero] = np.exp((-(np.log(freqs[nonzero] / fo)) ** 2) / (2 * np.log(sigma_on_f) ** 2))
    return out


def _grid_angles(freqs: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate arrays of filter grid angles.
    Args:
        freqs: As returned by _filter_grids.
        fx: As returned by _filter_grids.
        fy: As returned by _filter_grids.

    Returns:
        sin_thetas: Array of the sines of the angles in the filtergrid.
        cos_thetas: Array of the cosines of the angles in the filtergrid.

    """
    freqs[0, 0] = 1  # Avoid divide by 0
    sintheta = fx / freqs  # sine and cosine of filter grid angles
    costheta = fy / freqs
    freqs[0, 0] = 0  # Restore 0 DC
    return sintheta, costheta


def _spaced_frequencies(n: int) -> list[float]:
    """
    1D array of equally spaced frequencies, ranging from -0.5 to +0.5.
    Adjusts things appropriately for odd and even values of n so that
    the 0 frequency point is placed appropriately.
    n=2 -> [-0.5, 0.0]
    n=3 -> [-0.5, 0.0, 0.5]
    n=4 -> [-0.5, -0.25, 0.0, 0.25]
    n=5 -> [-0.5, -0.25, 0.0, 0.25, 0.5]
    Args:
        n: Number of frequencies used to cover the space. Must be at least 2.

    Returns: A list of frequencies covering -0.5 to +0.5, but including 0.

    """
    if n % 2 == 1:  # is odd
        return [i / (n - 1) for i in range(-(n - 1) // 2, (n - 1) // 2 + 1)]
    else:
        return [i / n for i in range(-n // 2, n // 2)]


def _filter_grids(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate frequency-shifted grids for constructing frequency domain filters.
    This differs slightly from Kovesi's implementation in Julia in that the extreme
    frequencies for odd-length rows (or cols) are the +ve and -ve Nyquist frequencies of 0.5.

    rows=4, cols=3 gives:
        f: array([[0.        , 0.5       , 0.5       ],
                  [0.25      , 0.55901699, 0.55901699],
                  [0.5       , 0.70710678, 0.70710678],
                  [0.25      , 0.55901699, 0.55901699]])
        fx: array([[ 0. ,  0.5, -0.5],
                   [ 0. ,  0.5, -0.5],
                   [ 0. ,  0.5, -0.5],
                   [ 0. ,  0.5, -0.5]])
        fy: array([[ 0.  ,  0.  ,  0.  ],
                   [ 0.25,  0.25,  0.25],
                   [-0.5 , -0.5 , -0.5 ],
                   [-0.25, -0.25, -0.25]])

    Args:
        rows: Number of rows in the image / filter.
        cols: Number of columns in the image / filter.

    Returns:
        f - Grid of size (rows, cols) containing frequency
            values from 0 to 0.5, where f = sqrt(fx^2 + fy^2).
            The grid is quadrant shifted so that 0 frequency is at f[0,0].

        fx, fy - Grids containing normalised frequency values
                 ranging from -0.5 to 0.5 in x and y directions
                 respectively. fx and fy are quadrant shifted.

    """

    # Set up X and Y spatial frequency matrices, fx and fy, with ranges
    # normalised to +/- 0.5
    fxrange = _spaced_frequencies(cols)
    fyrange = _spaced_frequencies(rows)

    fx = [[c for c in fxrange] for _ in fyrange]
    fy = [[r for _ in fxrange] for r in fyrange]  # reversed so +ve y comes first (plotted at the top)

    # Quadrant shift so that filters are constructed with 0 frequency at
    # the corners
    fx = ifftshift(fx)
    fy = ifftshift(fy)

    # Construct spatial frequency values in terms of normalised radius from
    # centre.
    f = np.sqrt(fx ** 2 + fy ** 2)
    return f, fx, fy


def _gaussian_angular_filter(
        angle: float,
        theta_sigma: float,
        sin_thetas: np.ndarray,
        cos_thetas: np.ndarray
) -> np.ndarray:
    """
    Orientation selective frequency domain filter with Gaussian windowing function.

    Args:
        angle: Orientation of the filter (radians)
        theta_sigma: Standard deviation of angular Gaussian window function.
        sin_thetas: As returned by _grid_angles
        cos_thetas: As returned by _grid_angles

    Returns: Filter described above.

    """
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)

    # For each point in the filter matrix calculate the angular
    # distance from the specified filter orientation.  To overcome
    # the angular wrap-around problem sine difference and cosine
    # difference values are first computed and then the atan2
    # function is used to determine angular distance.
    d_sins = sin_thetas * cos_angle - cos_thetas * sin_angle  # Difference in sine: sin(theta - alpha)
    d_coss = cos_thetas * cos_angle + sin_thetas * sin_angle  # Difference in cosine: cos(theta - alpha)
    d_thetas = np.arctan2(d_sins, d_coss)  # Angular distance.
    return np.exp(-d_thetas ** 2 / (2 * theta_sigma ** 2))


def _gaussian_angular_filter_3d(
        polar: float,
        azimuth: float,
        theta_sigma: float,
        sin_polars: np.ndarray,
        cos_polars: np.ndarray,
        sin_azimuths: np.ndarray,
        cos_azimuths: np.ndarray
) -> np.ndarray:
    """
    Orientation selective frequency domain filter with Gaussian windowing function.

    Args:
        polar: Polar orientation of the filter (radians) [theta]
        azimuth: Azimuthal orientation of the filter (radians) [phi]
        theta_sigma: Standard deviation of angular Gaussian window function.
        sin_polars: As returned by _grid_angles
        cos_polars: As returned by _grid_angles
        sin_azimuths: As returned by _grid_angles
        cos_azimuths: As returned by _grid_angles

    Returns: Filter described above.
    """
    # For each point in the filter matrix calculate the angular distance from the
    # specified filter orientation. The angle between the filter matrix element
    # and the filter orientation is obtained via a dot product in spherical coordinates:
    #
    #   cos(d_theta) = filter_orientation . grid_orientation
    #
    #                  | fx |   | gx |        || | fx | ||   || | fx | ||
    #                = | fy | . | gy |  where || | fy | || = || | fy | || = 1
    #                  | fz |   | gz |        || | fz | ||   || | fz | ||
    #
    #                  | Sin(f_polar) Cos(f_azi)    |     | Sin(g_polar) Cos(g_azi)    |
    #                = | Sin(f_polar) Sin(f_polar)) |  .  | Sin(g_polar) Sin(g_polar)) |
    #                  | Cos(f_polar)               |     | Cos(g_polar)               |

    d_thetas = np.arccos(
        sin(polar) * sin_polars * (cos(azimuth) * cos_azimuths + sin(azimuth) * sin_azimuths)
        + cos(polar) * cos_polars
    )
    return np.exp(-d_thetas ** 2 / (2 * theta_sigma ** 2))


def ppdenoise(
        img: np.ndarray,
        nscale: int = 5,
        norient: int = 6,
        mult: float = 2.5,
        minwavelength: float = 2.0,
        sigmaonf: float = 0.55,
        dthetaonsigma: float = 1.0,
        k: float = 3.0,
        softness: float = 1.0,
) -> np.ndarray:
    """
    Port of Kovesi's phase preserving denoising algorithm (Julia implementation).

    The convolutions are done via the FFT.  Many of the parameters relate
    to the specification of the filters in the frequency plane.  Most
    arguments do not need to be changed from the defaults and are mostly
    not that critical.  The main parameter that you may wish to play with
    is `k`, the number of standard deviations of noise to reject.

    Usage: cleanimage = ppdenoise(img,  nscale = 5, norient = 6,
                                  mult = 2.5, minwavelength = 2, sigmaonf = 0.55,
                                  dthetaonsigma = 1.0, k = 3, softness = 1.0)

    Original code:
        https://github.com/peterkovesi/ImagePhaseCongruency.jl/blob/master/src/phasecongruency.jl.

    Reference:
        Peter Kovesi, "Phase Preserving Denoising of Images".
        The Australian Pattern Recognition Society Conference: DICTA'99.
        December 1999. Perth WA. pp 212-217
        http://www.peterkovesi.com/papers/denoise.pdf

    Args:
        img: Image to be processed (greyscale)
        nscale: No of filter scales to use (5-7) - the more scales used the more low frequencies are covered.
        norient: No of orientations to use (6)
        mult: Multiplying factor between successive scales  (2.5-3)
        minwavelength: Wavelength of smallest scale filter (2)
        sigmaonf:  Ratio of the standard deviation of the Gaussian
                   describing the log Gabor filter's transfer function
                   in the frequency domain to the filter center
                   frequency (0.55)
        dthetaonsigma: Ratio of angular interval between filter
                       orientations and the standard deviation of
                       the angular Gaussian (1) function used to
                       construct filters in the freq. plane.
        k: No of standard deviations of noise to reject 2-3
        softness: Degree of soft thresholding (0-hard to 1-soft)

    Returns: Cleaned image of the same dimensions the input image. Absolute intensity magnitudes are not preserved
             as per the original algorithm.
    """

    # Reference:
    # Peter Kovesi, "Phase Preserving Denoising of Images".
    # The Australian Pattern Recognition Society Conference: DICTA'99.
    # December 1999. Perth WA. pp 212-217
    # http://www.peterkovesi.com/papers/denoise.pdf
    epsilon = 1e-5  # Used to prevent division by zero.
    # Calculate the standard deviation of the angular Gaussian function used to construct filters in the freq. plane.
    thetaSigma = pi / norient / dthetaonsigma
    rows, cols = img.shape
    IMG = fft2(img)
    # Generate grid data for constructing filters in the frequency domain
    freq, fx, fy = _filter_grids(rows, cols)
    sintheta, costheta = _grid_angles(freq, fx, -fy)  # Negative fy to give +ve anti-clockwise angles
    totalEnergy = np.zeros((rows, cols), np.cdouble)  # response at each orientation.
    RayMean = 0.0
    RayVar = 0.0
    for o in range(1, norient + 1):  # For each orientation.
        angle_rad = (o - 1) * pi / norient  # Calculate filter angle.
        # Generate angular filter
        angle_filter = _gaussian_angular_filter(angle_rad, thetaSigma, sintheta, costheta)

        wavelength = minwavelength  # Initialize filter wavelength.

        for s in range(1, nscale + 1):
            # Construct the filter = logGabor filter * angular filter
            fo = 1.0 / wavelength
            scale_filter = _log_gabor(freq, fo, sigmaonf)
            final_filter = scale_filter * angle_filter

            # Convolve image with even an odd filters returning the result in EO
            EO = ifft2(IMG * final_filter)
            aEO = np.abs(EO)

            if s == 1:
                # Estimate the mean and variance in the amplitude
                # response of the smallest scale filter pair at this
                # orientation.  If the noise is Gaussian the amplitude
                # response will have a Rayleigh distribution.  We
                # calculate the median amplitude response as this is a
                # robust statistic.  From this we estimate the mean
                # and variance of the Rayleigh distribution
                RayMean = np.median(aEO) * 0.5 * math.sqrt(-pi / math.log(0.5))
                RayVar = (4 - pi) * (RayMean ** 2) / pi

            # Compute soft threshold noting that the effect of noise
            # is inversely proportional to the filter bandwidth/centre
            # frequency. (If the noise has a uniform spectrum)
            T = (RayMean + k * math.sqrt(RayVar)) / (mult ** (s - 1))

            above_thresh = aEO > T  # aEO is less than T outside of this mask so makes no contribution to totalEnergy
            # Complex noise vector to subtract = T * normalize(EO) times degree of 'softness'
            V = softness * T * EO[above_thresh] / (aEO[above_thresh] + epsilon)
            EO[above_thresh] -= V  # Subtract noise vector.
            totalEnergy[above_thresh] += EO[above_thresh]

            wavelength *= mult  # Wavelength of next filter
    return np.real(totalEnergy)


def _grid_angles_3d(
        freqs: np.ndarray,
        fx: np.ndarray,
        fy: np.ndarray,
        fz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate arrays of filter grid angles.
    Args:
        freqs: As returned by _filter_grids.
        fx: As returned by _filter_grids.
        fy: As returned by _filter_grids.
        fz: As returned by _filter_grids.

    Returns:
        sin_polar: Array of the sines of the polar angles in the filtergrid.
        cos_polar: Array of the cosines of the polar angles in the filtergrid.
        sin_azimuth: Array of the sines of the azimuthal angles in the filtergrid.
        cos_azimuth: Array of the cosines of the azimuthal angles in the filtergrid.

    """
    fxy = np.sqrt(fx ** 2 + fy ** 2)
    fxy[:, 0, 0] = 1  # Avoid divide by 0
    freqs[0, 0, 0] = 1  # Avoid divide by 0
    sin_polar = fy / fxy
    cos_polar = fx / fxy
    sin_azimuth = fxy / freqs
    cos_azimuth = fz / freqs
    freqs[0, 0, 0] = 0  # Restore 0 DC
    return sin_polar, cos_polar, sin_azimuth, cos_azimuth


def _filter_grids_3d(plns: int, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate frequency-shifted grids for constructing frequency domain filters.
    This differs slightly from Kovesi's implementation in Julia in that the extreme
    frequencies for odd-length rows (or cols) are the +ve and -ve Nyquist frequencies of 0.5.

    Args:
        pln: Number of planes in the volume / 3D filter.
        rows: Number of rows in the volume / 3D filter.
        cols: Number of columns in the volume / 3D filter.
        pages: Number of pages (depths) in the volume / 3D filter.

    Returns:
        f - Grid of size (rows, cols) containing frequency
            values from 0 to 0.5, where f = sqrt(fx^2 + fy^2 + z^2).
            The grid is quadrant shifted so that 0 frequency is at f[0,0].

        fx, fy, fz - Grids containing normalised frequency values
                     ranging from -0.5 to 0.5 in x, y, and z directions
                     respectively. fx, fy, and fz are quadrant shifted.

    """

    # Set up X and Y spatial frequency matrices, fx and fy, with ranges
    # normalised to +/- 0.5
    fxrange = _spaced_frequencies(cols)
    fyrange = _spaced_frequencies(rows)
    fzrange = _spaced_frequencies(plns)

    # Array structure maintains convention: shape is (plns, rows, cols)
    fx = [[[c for c in fxrange] for _ in fyrange] for _ in fzrange]
    fy = [[[r for _ in fxrange] for r in fyrange] for _ in fzrange]
    fz = [[[p for _ in fxrange] for _ in fyrange] for p in fzrange]

    # Quadrant shift so that filters are constructed with 0 frequency at
    # the corners
    fx = ifftshift(fx)
    fy = ifftshift(fy)
    fz = ifftshift(fz)

    # Construct spatial frequency values in terms of normalised radius from
    # centre.
    f = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    return f, fx, fy, fz


def ppdenoise3d(
        img: np.ndarray,
        nscale: int = 5,
        mult: float = 2.5,
        minwavelength: float = 2.0,
        sigmaonf: float = 0.55,
        dthetaonsigma: float = 1.0,
        k: float = 3.0,
        softness: float = 1.0,
) -> np.ndarray:
    epsilon = 1e-5  # Used to prevent division by zero.

    # Calculate the standard deviation of the angular Gaussian function used to construct filters in the freq. plane.
    # Use the angle between face-centres of icosahedron (or vertices of dodecahedron)
    theta_sigma = acos(sqrt(5) / 3)  # about 42 degrees

    plns, rows, cols = img.shape
    freq_img = fft2(img)
    # Generate grid data for constructing filters in the frequency domain
    freq, fx, fy, fz = _filter_grids_3d(plns, rows, cols)
    sin_polar, cos_polar, sin_azimuth, cos_azimuth = _grid_angles_3d(freq, fx, fy, fz)
    total_energy = np.zeros((plns, rows, cols), np.cdouble)  # response at each orientation.
    ray_mean = 0.0
    ray_var = 0.0

    # Face normals of the top hemisphere of a (regular) icosahedron
    Ori3D = namedtuple('Ori3D', ['polar', 'azimuth'])
    orientations = [Ori3D(polar=pi - atan(3 - sqrt(5)), azimuth=-(2 * pi) / 5),
                    Ori3D(polar=pi - atan(3 - sqrt(5)), azimuth=0),
                    Ori3D(polar=pi - atan(3 - sqrt(5)), azimuth=(2 * pi) / 5),
                    Ori3D(polar=pi - atan(3 - sqrt(5)), azimuth=(4 * pi) / 5),
                    Ori3D(polar=pi - atan(3 - sqrt(5)), azimuth=-(4 * pi) / 5),
                    Ori3D(polar=pi - atan(3 + sqrt(5)), azimuth=-(2 * pi) / 5),
                    Ori3D(polar=pi - atan(3 + sqrt(5)), azimuth=0),
                    Ori3D(polar=pi - atan(3 + sqrt(5)), azimuth=(2 * pi) / 5),
                    Ori3D(polar=pi - atan(3 + sqrt(5)), azimuth=(4 * pi) / 5),
                    Ori3D(polar=pi - atan(3 + sqrt(5)), azimuth=-(4 * pi) / 5)]

    for o in orientations:  # For each of ten orientations.
        # Generate angular filter
        angle_filter = _gaussian_angular_filter_3d(
            polar=o.polar,
            azimuth=o.azimuth,
            theta_sigma=theta_sigma,
            sin_polars=sin_polar,
            cos_polars=cos_polar,
            sin_azimuths=sin_azimuth,
            cos_azimuths=cos_azimuth
        )

        wavelength = minwavelength  # Initialize filter wavelength.

        for s in range(1, nscale + 1):
            # Construct the filter = logGabor filter * angular filter
            fo = 1.0 / wavelength
            scale_filter = _log_gabor(freq, fo, sigmaonf)
            final_filter = scale_filter * angle_filter

            # Convolve image with even an odd filters returning the result in EO
            EO = ifft2(freq_img * final_filter)
            aEO = np.abs(EO)

            if s == 1:
                # Estimate the mean and variance in the amplitude
                # response of the smallest scale filter pair at this
                # orientation.  If the noise is Gaussian the amplitude
                # response will have a Rayleigh distribution.  We
                # calculate the median amplitude response as this is a
                # robust statistic.  From this we estimate the mean
                # and variance of the Rayleigh distribution
                ray_mean = np.median(aEO) * 0.5 * math.sqrt(-pi / math.log(0.5))
                ray_var = (4 - pi) * (ray_mean ** 2) / pi

            # Compute soft threshold noting that the effect of noise
            # is inversely proportional to the filter bandwidth/centre
            # frequency. (If the noise has a uniform spectrum)
            T = (ray_mean + k * math.sqrt(ray_var)) / (mult ** (s - 1))

            above_thresh = aEO > T  # aEO is less than T outside of this mask so makes no contribution to totalEnergy
            # Complex noise vector to subtract = T * normalize(EO) times degree of 'softness'
            V = softness * T * EO[above_thresh] / (aEO[above_thresh] + epsilon)
            EO[above_thresh] -= V  # Subtract noise vector.
            total_energy[above_thresh] += EO[above_thresh]

            wavelength *= mult  # Wavelength of next filter
    return np.real(total_energy)
