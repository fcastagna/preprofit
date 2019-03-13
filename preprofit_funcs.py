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

class Param:
    '''
    Class for parameters
    --------------------
    val = value of the parameter
    minval, maxval = minimum and maximum allowed values
    frozen = whether the parameter is allowed to vary (True/False)
    '''
    def __init__(self, val, minval = -1e99, maxval = 1e99, frozen = False):
        self.val = float(val)
        self.minval = minval       
        self.maxval = maxval
        self.frozen = frozen

    def __repr__(self):
        return '<Param: val=%.3g, minval=%.3g, maxval=%.3g, frozen=%s>' % (
            self.val, self.minval, self.maxval, self.frozen)


class Pressure:
    '''
    Class to parametrize the pressure profile
    -----------------------------------------    
    '''
    def __init__(self):
        pass
        
    def defPars(self):
        '''
        Default parameter values
        ------------------------
        P0 = normalizing constant
        a = slope at intermediate radii
        b = slope at large radii
        c = slope at small radii
        r500 = characteristic radius
        '''
        pars = {
            'P0': Param(0.4, minval=0, maxval = 0.8),
            'a': Param(1.33, minval = 0.3, maxval = 10),
            'b': Param(4.13, minval = 0.5, maxval = 15),
            'c': Param(0.014, minval = 0, maxval = 3),
            'r500': Param(930, minval = 500, maxval = 1500)
            }
        return pars

    def update_vals(self, pars, fit_pars, pars_val):
        '''
        Update the parameter values
        ---------------------------
        pars = set of pressure parameters
        fit_pars = name of the parameters to update
        pars_val = new parameter values
        '''
        for name, i in zip(fit_pars, range(len(fit_pars))):
            pars[name].val = pars_val[i] 

    def press_fun(self, pars, r_kpc):
        '''
        Compute the pressure profile
        ----------------------------
        pars = set of pressure parameters
        r_kpc = radius (kpc)
        '''
        P0 = pars['P0'].val
        a = pars['a'].val
        b = pars['b'].val
        c = pars['c'].val
        r500 = pars['r500'].val
        rp = r500 / 3.2 # c_delta = 3.2
        return P0 / ((r_kpc / rp)**c * (1 + (r_kpc / rp)**a)**((b - c) / a))    

    
 

def read_beam(filename):
    '''
    Read the beam data up to the first negative or nan value
    --------------------------------------------------------
    '''
    file = fits.open(filename) # load and read the file
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

def mybeam(filename, r_reg, regularize = True):
    '''
    Read the beam data, optionally set a regular step, normalize the 2D distribution and return the beam profile
    ------------------------------------------------------------------------------------------------------------
    filename = name of the file including the beam data
    r_reg = radius with regular step (arcsec)
    regularize = whether to regularize the step (True/False)
    ---------------------------
    RETURN: the normalized beam
    '''
    r_irreg, b = read_beam(filename)
    f = interp1d(np.append(-r_irreg, r_irreg), np.append(b, b), kind = 'cubic',
                 bounds_error = False, fill_value = (0, 0))
    if regularize == True:
        b = f(r_reg)
        sep = r_reg.size // 2
        step = r_reg[1] - r_reg[0]
        norm = simps(r_reg[sep:] * b[sep:], r_reg[sep:]) * 2 * np.pi / step**2
    else:
        step = np.mean(np.diff(r_irreg))
        norm = simps(r_irreg * b, r_irreg) * 2 * np.pi / step**2
        z = np.zeros(int((r_reg.size - 2 * r_irreg.size - 1) / 2))
        b = np.hstack((z, b[::-1], f(0), b, z))
    return b / norm

def centdistmat(num_odd, offset = 0):
    '''
    Create a matrix of distances from the central element
    -----------------------------------------------------
    num_odd = odd number of rows and columns
    offset = basic value for all the distances in the matrix (default is 0)
    -----------------------------------------------------------------------
    RETURN: the (num_odd x num_odd) matrix
    '''
    r = np.arange(num_odd)
    x, y = np.meshgrid(r, r)
    return np.sqrt((x - num_odd // 2)**2 + (y - num_odd // 2)**2) + offset

def ima_interpolate(dist_mat, x, y):
    '''
    Interpolate the (x, y) values at the distance values in dist_mat
    ----------------------------------------------------------------
    dist_mat = matrix of distances
    x, y = vector of coordinates of the distribution to interpolate
    ---------------------------------------------------------------
    RETURN: the matrix of the interpolated y-values for the x-values in dist_mat
    '''
    f = interp1d(x, y, 'cubic', bounds_error = False, fill_value = (0, 0))
    return f(dist_mat.flat).reshape(dist_mat.shape)

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

def log_lik(pars_val, press, pars, fit_pars, step, kpa, phys_const, radius, 
            y_mat, beam_2d, filtering, tf_len, sep, flux_data, conv):
    '''
    Computes the log-likelihood for the current pressure parameters
    ---------------------------------------------------------------
    pars_val = array of free parameters
    press = pressure object of the class Pressure
    pars = set of pressure parameters
    fit_pars = name of the parameters to fit
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
    # update pars
    press.update_vals(pars, fit_pars, pars_val)
    if all([pars[i].minval < pars[i].val < pars[i].maxval for i in pars]):
        r = np.arange(step * kpa, pars['r500'].val * 5 + step * kpa, step * kpa)
        # pressure profile
        pp = press.press_fun(pars, r)
        ub = min(pp.size, sep)
        # abel transform
        ab = direct_transform(pp, r = r, direction = "forward", backend = 'Python')[:ub]
        # Compton parameter
        y = phys_const[2] * phys_const[1] / phys_const[0] * ab
        f = interp1d(np.append(-r[:ub], r[:ub]), np.append(y, y), 'cubic')
        y = np.concatenate((y[::-1], f(0), y), axis = None)
        # Compton parameter 2D image
        y_2d = ima_interpolate(y_mat * step, radius[sep - ub:sep + ub + 1], y)
        # Convolution with the PSF
        conv_2d = fftconvolve(y_2d, beam_2d, 'same')[
                y_mat.shape[0] // 2 - tf_len + 1:y_mat.shape[0] // 2 + tf_len, 
                y_mat.shape[0] // 2 - tf_len + 1:y_mat.shape[0] // 2 + tf_len]
        # Convolution with the transfer function
        FT_map_in = fft2(conv_2d)
        map_out = np.real(ifft2(FT_map_in * filtering))
        map_prof = map_out[conv_2d.shape[0] // 2, conv_2d.shape[0] // 2:]
        g = interp1d(radius[sep:sep + map_prof.size], map_prof * conv,
                     fill_value = 'extrapolate')
        # Log-likelihood calculation
        log_lik = -np.sum(((flux_data[1] - g(flux_data[0])) / flux_data[2])**2) / 2
        return log_lik
    else:
        return -np.inf

def traceplot(mysamples, param_names, nsteps, nw, plotw = 20, plotdir = './'):
    '''
    Traceplot of the MCMC
    ---------------------
    mysamples = array of sampled values in the chain
    param_names = names of the parameters
    nsteps = number of steps in the chain (after burn-in) 
    nw = number of random walkers
    plotw = number of random walkers that we wanna plot (default is 20)
    plotdir = directory where to place the plot
    '''
    nw_step = int(np.ceil(nw / plotw))
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

def triangle(mysamples, param_names, plotdir = './'):
    '''
    Univariate and multivariate distribution of the parameters in the MCMC
    ----------------------------------------------------------------------
    mysamples = array of sampled values in the chain
    param_names = names of the parameters
    plotdir = directory where to place the plot
    '''
    plt.clf()
    pdf = PdfPages(plotdir + 'cornerplot.pdf')
    corner.corner(mysamples, labels = param_names)
    pdf.savefig()
    pdf.close()

def fit(pars_val, press, pars, fit_pars, step, kpa, phys_const, radius,
        y_mat, beam_2d, filtering, tf_len, sep, flux_data, conv):
    '''
    Computes the filtered Compton parameter profile for the values in pars_val
    --------------------------------------------------------------------------
    see log_lik
    '''
    press.update_vals(pars, fit_pars, pars_val)
    r = np.arange(step * kpa, pars['r500'].val * 5 + step * kpa, step * kpa)
    pp = press.press_fun(pars, r)
    ub = min(pp.size, sep)
    ab = direct_transform(pp, r = r, direction = "forward", backend = 'Python')[:ub]
    y = phys_const[2] * phys_const[1] / phys_const[0] * ab
    f = interp1d(np.append(-r[:ub], r[:ub]), np.append(y, y), 'cubic')
    y = np.concatenate((y[::-1], f(0), y), axis = None)
    y_2d = ima_interpolate(y_mat * step, radius[sep - ub:sep + ub + 1], y)
    conv_2d = fftconvolve(y_2d, beam_2d, 'same')[
            y_mat.shape[0] // 2 - tf_len + 1:y_mat.shape[0] // 2 + tf_len, 
            y_mat.shape[0] // 2 - tf_len + 1:y_mat.shape[0] // 2 + tf_len]
    FT_map_in = fft2(conv_2d)
    map_out = np.real(ifft2(FT_map_in * filtering))
    map_prof = map_out[conv_2d.shape[0] // 2, conv_2d.shape[0] // 2:]
    return map_prof * conv
    
def plot_best(theta, fit_pars, mp_med, mp_lb, mp_ub, radius, sep, flux_data, ci, plotdir = './'):
    '''
    Plot of the Compton parameter profile compared to the flux density data
    -----------------------------------------------------------------------
    mp_med = Compton profile for the median parameter values
    mp_lb, mp_ub = CI of the Compton profile
    plotdir = directory where to place the plot
    '''
    r_sec, y_data, err = flux_data
    plt.clf()
    pdf = PdfPages(plotdir + 'best_fit.pdf')
    plt.plot(radius[sep:sep + mp_med.size], mp_med)
    plt.plot(radius[sep:sep + mp_med.size], mp_lb, ':', color = 'b')
    plt.plot(r_sec, y_data, '.', color = 'red')
    plt.plot(radius[sep:sep + mp_med.size], mp_ub, ':', color = 'b')
    plt.vlines(r_sec, y_data - err, y_data + err, color = 'red')
    plt.legend(('Filtered profile', '%s% CI' % ci, 'Flux density'), loc = 'lower right')
    plt.axhline(y = 0, color = 'black', linestyle = ':')
    plt.xlabel('Radius (arcsec)')
    plt.ylabel('y x 10$^{4}$')
    plt.title('Compton parameter profile - best fit with %s% CI' % ci)
    pdf.savefig()
    pdf.close()
