import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from abel.direct import direct_transform
from scipy.integrate import simps
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import corner

font = {'size': 8}
plt.rc('font', **font)
plt.style.use('classic')

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

def read_beam(name):
    '''
    Read the beam data up to the first negative or nan value
    --------------------------------------------------------
    '''
    file = fits.open(name) # load and read the file
    data = file[''].data
    radius, beam_prof = data[0][:2]
    if np.isnan(beam_prof).sum() > 0:
        first_nan = np.where(np.isnan(beam_prof))[0][0]
        radius = radius[:first_nan]
        beam_prof = beam_prof[:first_nan]
    if np.min(beam_prof) < 0:
        first_neg = np.where(beam_prof < 0)[0][0]
        radius = radius[:first_neg]
        beam_prof = beam_prof[:first_neg]
    return radius, beam_prof

def mybeam(filename, r, sep, regularize = True, norm = 1):
    '''
    Read the beam data, optionally set a regular step, optionally normalize the 1D or 2D distribution
    -------------------------------------------------------------------------------------------------
    r = radius
    regularize = T / F
    norm = '1d' / '2d'
    ------------------
    RETURN: the normalized beam
    '''
    r2, b = read_beam(filename)
    f = interp1d(np.append(-r2, r2), np.append(b, b), kind = 'cubic',
                 bounds_error = False, fill_value = (0, 0))
    if regularize == True:
        b = f(r)
        if norm == '1d':
            norm = np.trapz(b, r)
        elif norm == '2d':
            step = r[1] - r[0]
            norm = simps(r[sep:] * b[sep:], r[sep:]) * 2 * np.pi / step**2
    else:
        if norm == '1d':
            norm = 2 * np.trapz(np.append(f(0), b), np.append(0, r2))
        elif norm == '2d':
            norm = simps(r2 * b, r2) * 2 * np.pi
        z = np.zeros(int((r.size - 2 * r2.size - 1) / 2))
        b = np.hstack((z, b[::-1], f(0), b, z))
    return b / normdef myread(r, filename, regularize = False, norm = 1):

def centdistmat(num, offset = 0):
    '''
    Create a matrix of distances from the central element
    -----------------------------------------------------
    num = number of rows and columns (number of pixels)
    offset = basic value for all the distances in the matrix (default is 0)
    -----------------------------------------------------------------------
    RETURN: the (num x num) matrix
    '''
    r = np.arange(num) # Array of radius values
    x, y = np.meshgrid(r, r)
    return np.sqrt((x - num // 2)**2 + (y - num // 2)**2) + offset

def ima_interpolate(dist, x, y):
    '''
    Interpolate the (x, y) values at x = dist
    -----------------------------------------
    dist = matrix of distances
    x, y = vector of coordinates of the distribution to interpolate
    ---------------------------------------------------------------
    RETURN: the matrix of the interpolated y-values for the x-values in dist
    '''
    f = interp1d(x, y, 'cubic', bounds_error = False, fill_value = (0, 0))
    return f(dist.flat).reshape(dist.shape)  # interpolate to get value at radius

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

def log_posterior(theta, fit_par, par, par_val, step, kpa, phys_const, radius,
                  y_mat, beam_2d, filtering, tf_len, sep, flux_data, conv):
    '''
    Computes the log-posterior probability for the parameters in theta
    ------------------------------------------------------------------
    theta = array of free parameters
    fit_par = parameters to fit
    par = fixed parameters
    par_val = values for the fixed parameters
    step = radius[1] - radius[0]
    kpa = number of kpc per arcsec
    phys_const = physical constants
    radius = radius (arcsec)
    y_mat = matrix of distances for the Compton parameter
    beam_2d = PSF image
    filtering = tranfer function
    tf_len = number of tf measurements
    sep = index of radius 0
    flux data:
        y_data = flux density
        r_sec = x-axis values for y_data
        err = statistical errors of the flux density
    conv = conversion rate from Jy to beam
    --------------------------------------------------------------
    RETURN: log-posterior probability or -inf whether theta is out of the parameter space
    '''
    for j in range(len(fit_par)): globals()[fit_par[j]] = theta[j]
    for j in range(len(par)): globals()[par[j]] = par_val[j]
    if (P0 < 0) or (r500 < 500) or (a < 0):
        return -np.inf
    else:
        r = np.arange(step * kpa, r500 * 5 + step * kpa, step * kpa)
        pp = pressure_prof(r, P0, a, b, c, r500)
        ab = direct_transform(pp, r = r, direction = "forward", 
                              backend = 'Python')[:sep] # Check Cython!
        y = phys_const[2] * phys_const[1] / phys_const[0] * ab
        f = interp1d(np.append(-r[:sep], r[:sep]), np.append(y, y), 'cubic')
        y = np.concatenate((y[::-1], f(0), y), axis = None)
        y_2d = ima_interpolate(y_mat * step, radius, y)
        conv_2d = fftconvolve(y_2d, beam_2d, 'same')[
                sep - tf_len + 1:sep + tf_len, sep - tf_len + 1:sep + tf_len]
        FT_map_in = fft2(conv_2d)
        map_out = np.real(ifft2(FT_map_in * filtering))
        map_prof = map_out[conv_2d.shape[0] // 2, conv_2d.shape[0] // 2:]
        g = interp1d(radius[sep:sep + map_prof.size], map_prof * conv,
                     fill_value = 'extrapolate')
        log_lik = -np.sum(((flux_data[1] - g(flux_data[0])) / flux_data[2])**2)/2
        log_post = log_lik
        return log_post

def traceplot(mysamples, param_names, nsteps, nw, plotdir = './'):
    '''
    Traceplot of the MCMC
    ---------------------
    mysamples = array of sampled values
    param_names = parameters' labels
    nsteps = number of steps in the MCMC (after burn-in) 
    nw = number of random walkers
    plotdir = directory where to place the plot
    '''
    nw_step = nw // 20
    pdf = PdfPages(plotdir + 'traceplot.pdf')
    plt.figure().suptitle('Traceplot')
    for i in np.arange(mysamples.shape[1]):
        plt.subplot(2, 1, i % 2 + 1)
        for j in range(nw)[::nw_step]:
            plt.plot(np.arange(nsteps), mysamples[j::nw,i], linewidth = .2)
        plt.xlabel('Iteration number')
        plt.ylabel('%s' % param_names[i])
        if (abs((i + 1) % 2) < 0.01):
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

def fit_best(theta, fit_par, par, par_val, step, kpa, phys_const, radius,
             y_mat, beam_2d, filtering, tf_len, sep, flux_data, conv,
             out = 'comp'):
    '''
    Computes alternatively the filtered Compton parameter profile or the
    integrated pressure profile for the optimized parameter values
    --------------------------------------------------------------------
    v(log_posterior)
    out = 'comp' or 'pp' depending on the desired output
    '''
    for j in range(len(fit_par)): globals()[fit_par[j]] = theta[j]
    for j in range(len(par)): globals()[par[j]] = par_val[j]
    r = np.arange(step * kpa, r500 * 5 + step * kpa, step * kpa)
    pp = pressure_prof(r, P0, a, b, c, r500)
    if out == 'pp': return pp[:sep]
    elif out == 'comp':
        ab = direct_transform(pp, r = r, direction = "forward", 
                              backend = 'Python')[:sep] # Check Cython!
        y = phys_const[2] * phys_const[1] / phys_const[0] * ab
        f = interp1d(np.append(-r[:sep], r[:sep]), np.append(y, y), 
                     kind = 'cubic')
        y = np.concatenate((y[::-1], f(0), y), axis = None)
        y_2d = ima_interpolate(y_mat * step, radius, y)
        conv_2d = fftconvolve(y_2d, beam_2d, 'same')[
                sep - tf_len + 1:sep + tf_len, sep - tf_len + 1:sep + tf_len]
        FT_map_in = fft2(conv_2d)
        map_out = np.real(ifft2(FT_map_in * filtering))
        map_prof = map_out[conv_2d.shape[0] // 2, conv_2d.shape[0] // 2:]
        return map_prof * conv
    else: raise('Error: incorrect out parameter')

def plot_best(theta, fit_par, mp_mean, mp_lb, mp_ub, radius, sep, flux_data,
              clusdir = './'):
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
    plt.plot(radius[sep:sep + mp_mean.size], mp_mean)
    plt.plot(radius[sep:sep + mp_mean.size], mp_lb, ':', color = 'b')
    plt.plot(r_sec, y_data, '.', color = 'red')
    plt.plot(radius[sep:sep + mp_mean.size], mp_ub, ':', color = 'b')
    plt.vlines(r_sec, y_data - err, y_data + err, color = 'red')
    plt.legend(('Filtered profile', '95% CI', 'Flux density'), loc = 
               'lower right')
    plt.axhline(y = 0, color = 'black', linestyle = ':')
    plt.xlabel('Radius (arcsec)')
    plt.ylabel('y x 10$^{4}$')
    plt.title('Compton parameter profile - best fit with 95% CI')
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
    plt.title('Pressure profile with ' + str(fit_par) + ' = ' +
              str(list(map(lambda x: round(float(x), 3), theta))))
    pdf.savefig()
    pdf.close()
