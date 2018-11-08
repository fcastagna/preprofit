import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes
import corner

import matplotlib as mp
mp.rcParams['axes.linewidth'] = .5
mp.rcParams['grid.linewidth'] = .5
mp.rcParams['lines.linewidth'] = .5
mp.rcParams['patch.linewidth'] = .5
mp.rcParams['axes.labelsize'] = 8
mp.rcParams['xtick.labelsize'] = 5
mp.rcParams['ytick.labelsize'] = 5
mp.rcParams['legend.fontsize'] = 8

def pressure_prof(r, P0, a, b, c, r500):
    '''
    Compute the pressure profile
    ----------------------------
    r = radius (kpc)
    P0 = normalizing constant
    a = slope at intermediate radii
    b = slope at large radii
    c = slope at small radii
    r500 = characteristic radius
    '''
    cdelta = 3.2
    rp = r500 / cdelta
    return P0 / ((r / rp)**c * (1 + (r / rp)**a)**((b - c) / a))

def abel_intg(r, y, P0, a, b, c, r500):
    '''
    Function to be integrated along the line of sight
    -------------------------------------------------------
    r = radius(kpc)
    y = orthogonal radius (kpc)
    '''
    return 2 * pressure_prof(r, P0, a, b, c, r500) * r / np.sqrt(r**2 - y**2)

def prof_intg(step, kpa, P0, a, b, c, r500):
    '''
    Computes the Abel integrated pressure profile
    ---------------------------------------------
    step = radius step (arcsec)
    kpa = number of kpc per arcsec
    '''
    r = np.arange(0, np.ceil(r500 * 5 / kpa), step) * kpa # radius [0, r500 * 5]
    # Simps always requires an even number of intervals
    if r.size % 2 == 0:
        r = r[:-1] # if size is even, it becomes odd
    r = np.append(r, r[-1] + r[1]) # from odd to even
    # we need to use a shift, otherwise the integral returns an error when r = y
    shift = (r[1] - r[0]) / 16
    prof_integral = np.array([simps(abel_intg(r[j:(-1 - j % 2)] + shift, r[j], P0, a, b, c, r500), r[j:(-1 - j % 2)] + shift)
                              for j in range(r.size - 1)])
    return prof_integral[:-2]

def read_data(name):
    '''
    Read the beam data
    ------------------
    '''
    file = fits.open(name) # load and read the file
    data = file[""].data
    radius = data[0][0]; beam_prof = data[0][1]
    ## Delete negative values
    neg = np.where(beam_prof[np.where(~np.isnan(beam_prof))] < 0)[0][0]
    radius = radius[0:neg]
    beam_prof = beam_prof[0:neg]
    return radius, beam_prof

def myread(r, filename, regularize = False, norm = 1):
    '''
    Read the beam data, optionally set a regular step, optionally normalize the 1D or 2D distribution
    -------------------------------------------------------------------------------------------------
    r = radius
    regularize = T / F
    norm = '1d' / '2d'
    ------------------
    RETURN: the normalized beam
    '''
    r2, b = read_data(filename)
    f = interp1d(np.append(-r2, r2), np.append(b, b), kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')
    if regularize == True:
        step = r[1] - r[0]
        b = np.where(abs(r) > r2[-1], 0, f(r))
        if norm == '1d':
            norm = np.sum(b) * step
        elif norm == '2d':
            sep = r.size // 2
            norm = simps(r[sep:] * b[sep:], r[sep:]) * 2 * np.pi
    else:
        if norm == '1d':
            norm = 2 * np.trapz(np.append(f(0), b), np.append(0, r2))
        elif norm == '2d':
            norm = simps(r2 * b, r2) * 2 * np.pi
        z = np.zeros(int((r.size - 2 * r2.size - 1) / 2))
        b = np.hstack((z, b[::-1], f(0), b, z))
    return b / norm

def centeredDistanceMatrix(n, num, offset = 0):
    '''
    Create a matrix of distances from the central element
    -----------------------------------------------------
    n = maximum distance
    num = number of rows and columns (number of pixels)
    offset = basic value for all the distances in the matrix (default is 0)
    -----------------------------------------------------------------------
    RETURN: the (num x num) matrix
    '''
    r = np.linspace(0, n, num) # Array of radius values
    m = int((num - 1) / 2)
    x, y = np.meshgrid(r, r)
    val = x[m][m] # Median value at which the center is to be shifted 
    return np.sqrt((x - val)**2 + (y - val)**2) + offset

def interpolate(dist, x, y):
    '''
    Interpolate the (x, y) values at x = dist
    -----------------------------------------
    dist = matrix of distances
    x, y = vector of coordinates of the distribution to interpolate
    ---------------------------------------------------------------
    RETURN: the matrix of the interpolated y-values for the x-values in dist
    '''
    f = interp1d(x, y, kind = 'cubic', bounds_error = False, fill_value = (0, 0))
    return f(dist.flat).reshape(dist.shape)  # interpolate to get value at radius

def ima_interpolate(y, r, odd):
    '''
    2D image interpolation (1 pixel = 1 step)
    ---------------------------------------------------------------------
    y = y-axis values to interpolate
    r = x-axis values to interpolate
    odd = odd number of pixel for one side of the (odd x odd)-dimensional image
    -------------------------------------------------------------------------------
    RETURN: matrix values of the interpolated image
    '''
    mat = centeredDistanceMatrix(odd - 1, odd)
    return interpolate(mat * (r[1] - r[0]), r, y)

def dist(naxis):
    '''
    Returns a symmetric matrix in which the value of each element is proportional to its frequency 
    (https://www.harrisgeospatial.com/docs/DIST.html)
    ----------------------------------------------------------------------------------------------
    naxis = number of elements per row and per column
    -------------------------------------------------
    RETURN: the (naxis x naxis) matrix
    '''
    axis = np.linspace(- naxis // 2 + 1, naxis // 2, naxis)
    result = np.sqrt(axis**2 + axis[:,np.newaxis]**2)
    return np.roll(result, naxis // 2 + 1, axis = (0, 1))

def log_posterior(theta, step, fit_par, par, par_val, kpa, phys_const, radius, pix_comp, beam_2d, filtering, tf_len, sep, flux_data):
    '''
    Computes the log-posterior probability for the parameters in theta
    ------------------------------------------------------------------
    theta = array of free parameters
    step = radius[1] - radius[0]
    fit_par = parameters to fit
    par = fixed parameters
    par_val = values for the fixed parameters
    kpa = number of kpc per arcsec
    phys_const = physical constants
    radius = radius (arcsec)
    pix_comp = number of pixels for the Compton parameter image
    beam_2d = PSF image
    filtering = tranfer function
    tf_len = number of tf measurements
    sep = index of radius 0
    flux data:
        y_data = flux density
        r_sec = x-axis values for y_data
        err = statistical errors of the flux density
    --------------------------------------------------------------
    RETURN: log-posterior probability or -inf whether theta is out of the parameter space
    '''
    for j in range(len(fit_par)): globals()[fit_par[j]] = theta[j]
    for j in range(len(par)): globals()[par[j]] = par_val[j]
    if (P0 < 0):
        return -np.inf
    else:
        prof_integral = prof_intg(step, kpa, P0, a, b, c, r500)[:radius[sep:].size]
        y = phys_const[2] * phys_const[1] / phys_const[0] * prof_integral
        y = np.append(y[:0:-1], y)
        y_2d = ima_interpolate(y, radius, pix_comp)
        conv_2d = fftconvolve(y_2d, beam_2d, 'same')[tf_len - 1:-(tf_len - 1), tf_len - 1:-(tf_len - 1)]
        FT_map_in = fft2(conv_2d)
        map_out = np.real(ifft2(FT_map_in * filtering))
        map_prof = map_out[conv_2d.shape[0] // 2, conv_2d.shape[0] // 2:]
        g = interp1d(radius[sep:sep + map_prof.size], map_prof, fill_value = 'extrapolate')
        log_lik = -np.sum(((flux_data[1] - g(flux_data[0])) / flux_data[2])**2)/2
        log_post = log_lik
        return log_post

def traceplot(mysamples, param_names, clusdir = './'):
    '''
    Traceplot of the MCMC
    ---------------------
    mysamples = array of sampled values
    param_names = parameters' labels
    clusdir = directory where to place the plot
    '''
    plt.clf()
    pdf = PdfPages(clusdir + 'traceplot.pdf')
    plt.figure().suptitle('Traceplot', fontsize = 12)
    for i in np.arange(mysamples.shape[1]):
        plt.subplot(mysamples.shape[1], 1, i + 1)
        plt.plot(np.arange(mysamples.shape[0]), mysamples[:,i])
        plt.xlabel('Iteration number')
        plt.ylabel('%s' % param_names[i])
        if (abs((i + 1) % mysamples.shape[1]) < 0.01):
            pdf.savefig()
            plt.clf()
    pdf.savefig()                 
    pdf.close()

def triangle(samples2, param_names, clusdir = './'):
    '''
    Univariate and multivariate distribution of the parameters in the MCMC
    ----------------------------------------------------------------------
    samples2 = array of sampled values
    param_names = parameters' labels
    clusdir = directory where to place the plot
    '''
    plt.clf()
    pdf = PdfPages(clusdir + 'cornerplot.pdf')
    fig = corner.corner(samples2, labels = param_names)
    pdf.savefig()
    pdf.close()

def fit_best(theta, fit_par, par, par_val, step, kpa, phys_const, radius, pix_comp, 
             beam_2d, filtering, tf_len, sep, flux_data, out = 'comp'):
    '''
    Computes alternatively the filtered Compton parameter profile or the integrated pressure profile for the optimized parameter values
    -----------------------------------------------------------------------------------------------------------------------------------
    see (log_posterior)
    out = 'comp' or 'pp' depending on the desired output
    '''
    for j in range(len(fit_par)): globals()[fit_par[j]] = theta[j]
    for j in range(len(par)): globals()[par[j]] = par_val[j]
    prof_integral = prof_intg(step, kpa, P0, a, b, c, r500)[:radius[sep:].size]
    if out == 'pp': return prof_integral
    elif out == 'comp':
        y = phys_const[2] * phys_const[1] / phys_const[0] * prof_integral
        y = np.append(y[:0:-1], y)
        y_2d = ima_interpolate(y, radius, pix_comp)
        conv_2d = fftconvolve(y_2d, beam_2d, 'same')[tf_len - 1:-(tf_len - 1), tf_len - 1:-(tf_len - 1)]
        FT_map_in = fft2(conv_2d)
        map_out = np.real(ifft2(FT_map_in * filtering))
        map_prof = map_out[conv_2d.shape[0] // 2, conv_2d.shape[0] // 2:]
        return map_prof
    else: raise('Error: incorrect out parameter')

def plot_best(theta, fit_par, mp_mean, mp_lb, mp_ub, radius, sep, flux_data, clusdir = './'):
    '''
    Plot of the Compton parameter profile compared to the flux density
    ------------------------------------------------------------------
    mp_mean = Compton profile for the optimized parameter values
    mp_lb, mp_ub = 95% IC of the Compton profile
    clusdir = directory where to place the plot
    '''
    for j in range(len(fit_par)): globals()[fit_par[j]] = theta[j]
    r_sec, y_data, err = flux_data
    plt.clf()
    pdf = PdfPages(clusdir + 'best_fit.pdf')
    plt.plot(radius[sep:sep + mp_mean.size], 10**4 * mp_mean, color = 'blue')
    plt.fill_between(radius[sep:sep + mp_mean.size], 10**4 * mp_lb, 10**4 * mp_ub)
    plt.plot(r_sec, 10**4 * y_data, '.', color = 'red', markersize = .7)
    plt.vlines(r_sec, 10**4 * (y_data - err), 10**4 * (y_data + err), color = 'red')
    plt.legend(('Filtered profile', 'Flux density', '95% CI'))
    plt.axhline(y = 0, color = 'black', linestyle = ':')
    plt.xlabel('Radius (arcsec)')
    plt.ylabel('y x 10$^{4}$')
    plt.title('Compton parameter profile with ' + str(fit_par) + ' = ' + str(list(map(lambda x: round(float(x), 3), theta))),
              fontsize = 9)
    pdf.savefig()
    pdf.close()

def pp_best(theta, fit_par, par, par_val, r, clusdir):
    '''
    Plot of the pressure profile
    ----------------------------
    '''
    for j in range(len(fit_par)): globals()[fit_par[j]] = theta[j]
    for j in range(len(par)): globals()[par[j]] = par_val[j]
    pp = pressure_prof(r, P0, a, b, c, r500)
    pdf = PdfPages(clusdir + 'pp_best.pdf')
    plt.clf()
    plt.plot(r, pp)
    plt.plot(200, .08, '.', 1150, .001, '.')
    plt.xlim(100, 1300)
    plt.yscale('log')
    plt.ylim(.0001, 1)
    plt.ylabel('Pressure (keV / cm$^{3}$)')
    plt.xlabel('Radius (kpc)')
    plt.title('Pressure profile with ' + str(fit_par) + ' = ' + str(list(map(lambda x: round(float(x), 3), theta))), fontsize = 9)
    pdf.savefig()
    pdf.close()

def abel_best(theta, fit_par, pp, rad_kpc, sep, clusdir = './'):
    '''
    Plot of the integrated pressure profile
    ---------------------------------------
    '''
    for j in range(len(fit_par)): globals()[fit_par[j]] = theta[j]
    plt.clf()
    pdf = PdfPages(clusdir + 'abel_best.pdf')
    plt.plot(rad_kpc[:pp.size] / r500, pp, color = 'blue')
    plt.xlabel('Radius / r500')
    plt.ylabel('Pressure')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.09, 1.2)
    plt.title('Integrated pressure profile with ' + str(fit_par) + ' = ' + str(list(map(lambda x: round(float(x), 3), theta))), 
              fontsize = 9)
    pdf.savefig()
    pdf.close()
