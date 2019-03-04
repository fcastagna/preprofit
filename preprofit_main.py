from conv_funcs import (mybeam, centdistmat, ima_interpolate, dist, 
                        log_posterior, traceplot, triangle, fit_best, 
                        plot_best, pp_best)
import numpy as np
import mbproj2 as mb
from astropy.io import fits
from scipy.interpolate import interp1d
import emcee
import six.moves.cPickle as pickle

plotdir = './' # plot directory


### Global-global variables

# Physical constants
m_e = 0.5109989 * 10**3 # electron rest mass (keV)
sigma_T = 6.6524587158 * 10**(-25) # Thomson cross section (cm^2)
kpc_cm = mb.physconstants.kpc_cm # cm in 1 kpc
phys_const = [m_e, sigma_T, kpc_cm]

### Global variables

# Fittable parameters
par = ['P0', 'a', 'b', 'c', 'r500']

# Parameters that we want to fit
fit_par = ['P0', 'r500']

# Parameters for the gNFW pressure profile calculation
# ----------------------------------------------------
# P0 = 0.4 --- normalizing constant
# a = 1.33 --- slope at intermediate radii
# b = 4.13 --- slope at large radii
# c = 0.014 -- slope at small radii
# r500 = 930 - characteristic radius

# Values for the fixed parameters
par_val = [0.4, 1.33, 4.13, 0.014, 930]

# Sampling step
mystep = 2 # (arcsec)

# Number of pixels
pix_beam = 61 # number of pixels for one side of the PSF image
pix_comp = 301 # number of pixels for one side of the Compton parameter image

# MCMC parameters
ndim = len(fit_par)
nwalkers = 200
nthreads = 8
nburn = 5000
nsteps = 2000
np.random.seed(0)
ci = 95 # confidence interval level


### Local variables

redshift = 0.888
compt_param_mJy = -10.9 * 10**3 # Jy/beam to Compton parameter

# File names
beam_filename = 'data/Beam150GHz.fits'
tf_filename = 'data/TransferFunction150GHz_CLJ1227.fits'
flux_filename = 'data/press_data_cl1226_flagsource.dat'

# Cosmological parameters
cosmology = mb.Cosmology(redshift)
cosmology.H0 = 67.11 # Hubble's constant (km/s/Mpc)
cosmology.WM = 0.3175 # matter density
cosmology.WV = 0.6825 # vacuum density
kpc_per_arcsec = cosmology.kpc_per_arcsec # number of kpc per arcsec

# -------------------------------------------------------------------------------------------------------------------------------
# Code 
# -------------------------------------------------------------------------------------------------------------------------------

# Parameter definition
start_val = np.zeros(len(fit_par))
for j in range(len(par)):
    if par[j] not in fit_par: globals()[par[j]] = par_val[j] # fixed parameters
    else: start_val[np.where(start_val == 0)[0][0]] = par_val[j] # starting values for the parameters to fit
for j in range(len(fit_par)):
    par.remove(fit_par[j])
par_val = list(map(lambda x: globals()[x], par))

# Radius definition
tf_len = fits.open(tf_filename)[1].data[0][1].size # number of tf measurements
tf_mat_len = tf_len * 2 - 1 # one side length of the tf image
mymaxr = np.arange(0, pix_comp + 1, 2).size * 2 # max radius needed
radius = np.arange(0, mymaxr, mystep) # arcsec
rad_kpc = radius * kpc_per_arcsec # from arcsec to kpc
radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
sep = radius.size // 2 # index of radius 0

# PSF read, regularize and image creation
beam = mybeam(beam_filename, radius, sep, regularize = True, norm = '2d')
beam_mat = centdistmat(pix_beam)
beam_2d = ima_interpolate(beam_mat * mystep, radius, beam)

# Y 2D matrix
y_mat = centdistmat(pix_comp)

# Transfer function
kmax = 1 / mystep
karr = dist(tf_mat_len)
karr = karr / np.max(karr) * kmax
tf_data = fits.open(tf_filename)
tf = tf_data[1].data[0][1]
wn_as = tf_data[1].data[0][0] # wave number in arcsec^(-1)
f = interp1d(wn_as, tf, fill_value = 'extrapolate') # tf interpolation
filtering = f(karr)

# Flux density data
data = np.loadtxt(flux_filename, skiprows = 1, unpack = True)
r_sec = data[0]
y_data = data[1]
err = data[2] # TBI statistical error
flux_data = [r_sec, y_data, err]

# Jy/beam to Compton parameter
convert = compt_param_mJy

# Bayesian fit
starting_guess = start_val
starting_var = np.array(np.repeat(.1, ndim))
starting_guesses = np.random.random((nwalkers, ndim)) * starting_var + starting_guess
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args = [
    fit_par, par, par_val, mystep, kpc_per_arcsec, phys_const, radius, y_mat, 
    beam_2d, filtering, tf_len, sep, flux_data, convert], threads = nthreads)
sampler.run_mcmc(starting_guesses, nburn + nsteps)
print('Acceptance fraction: %s' % np.mean(sampler.acceptance_fraction))
mysamples = sampler.chain[:,nburn:,:].reshape(-1, ndim, order = 'F')

## Save the chain
file = open('mychain', 'wb') # create file
res = list([sampler.chain, sampler.lnprobability])
pickle.dump(res, file) # write
file.close()

# Posterior distribution's parameters
param_mean = np.empty(ndim)
param_std = np.empty(ndim)
for ii in np.arange(ndim):
    param_mean[ii] = np.mean(mysamples[:,ii])
    param_std[ii] = np.std(mysamples[:,ii])
    print('Mean(%s): %s; Sd(%s): %s' % (fit_par[ii], param_mean[ii], fit_par[ii], param_std[ii]))


### Plots
## Traceplot
traceplot(mysamples, fit_par, nsteps, nwalkers, plotdir)

## Corner plot
triangle(mysamples, fit_par, plotdir)

# Random samples of at most 1000 profiles
out_prof = np.array([fit_best(
    mysamples[j], fit_par, par, par_val, mystep, kpc_per_arcsec, phys_const,
    radius, y_mat, beam_2d, filtering, tf_len, sep, flux_data, convert,
    out = 'comp') for j in np.random.choice(mysamples.shape[0], 
                size = min(1000, mysamples.shape[0]), replace = False)])
quant = np.percentile(out_prof, [50, 50 - ci / 2, 50 + ci / 2], axis = 0)
plot_best(param_mean, fit_par, quant[0], quant[1], quant[2], radius, sep,
          flux_data, clusdir = plotdir)

## Pressure profile for the best fitting parameters
best_pp = fit_best(param_mean, fit_par, par, par_val, mystep, kpc_per_arcsec,
                   phys_const, radius, y_mat, beam_2d, filtering, tf_len, sep,
                   flux_data, convert, out = 'pp')
pp_best(param_mean, fit_par, par, par_val, rad_kpc[1:], plotdir)
