from preprofit_funcs import (myread, interpolate, ima_interpolate, dist, log_posterior, traceplot, triangle, fit_best, plot_best, 
                             pp_best, abel_best, test_abel_integ)
import numpy as np
import mbproj2 as mb
from astropy.io import fits
from scipy.interpolate import interp1d
import emcee
import six.moves.cPickle as pickle
import time

time0 = time.time()
plotdir = './bayes_plot/' # plot directory


### Global-global variables

# Physical constants
m_e = 0.5109989 * 10**3 # electron rest mass (keV)
sigma_T = 6.6524587158 * 10**(-25) # Thomson cross section (cm^2)
kpc_cm = mb.physconstants.kpc_cm # cm in 1 kpc
phys_const = [m_e, sigma_T, kpc_cm]

### Global variables

# Fittable parameters
par = ['P0', 'a', 'b', 'c', 'r500']

# Parameters to fit
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
nwalkers = 28
nthreads = 14
nburn = 2000
nsteps = 5000
np.random.seed(0)


### Local variables

redshift = 0.888
compt_param_mJy = -10.9 * 10**3 # Compton parameter to Jy/beam

# File names
beam_filename = 'Beam150GHz.fits'
tf_filename = 'TransferFunction150GHz_CLJ1227.fits'
flux_filename = 'CLJ1227_data.txt'

# Cosmology parameters
cosmology = mb.Cosmology(redshift)
cosmology.H0 = 67.11 # Hubble's constant (km/s/Mpc)
cosmology.WM = 0.3175 # matter density
cosmology.WV = 0.6825 # vacuum density
kpc_per_arcsec = cosmology.kpc_per_arcsec # number of kpc per arcsec

# ------------------------------------------------------------------
# Code
# ------------------------------------------------------------------

# Parameter definition
start_val = np.zeros(len(fit_par))
for j in range(len(par)):
    if par[j] not in fit_par: globals()[par[j]] = par_val[j]
    else: start_val[np.where(start_val == 0)[0][0]] = par_val[j]
for j in range(len(fit_par)):
    par.remove(fit_par[j])
par_val = list(map(lambda x: globals()[x], par))

# Radius definition
tf_len = fits.open(tf_filename)[1].data[0][1].size # number of tf measurements
tf_mat_len = tf_len * 2 - 1 # one side length of the tf image
mymaxr = np.ceil(tf_mat_len // 2 * np.sqrt(2) * mystep) # max radius needed
radius = np.arange(0, mymaxr, mystep) # arcsec
rad_kpc = radius * kpc_per_arcsec # from arcsec to kpc
radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
sep = radius.size // 2 # index of radius 0

# PSF read, regularize and image creation
beam = myread(radius, beam_filename, regularize = True, norm = '2d') * mystep**2
beam_2d = ima_interpolate(beam, radius, pix_beam)

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
r_sec = np.linspace(0.5, max(data[0]), max(data[0])) * 0.0017425 * 3600
# 0.0017425 * 3600 is the step in r [arcsec]
y_data = data[1] / data[2] / compt_param_mJy # Jy / beam to Compton parameter
err = .00001 + y_data - y_data # TBI statistical error
flux_data = [r_sec, y_data, err]

# Bayesian fit
starting_guess = start_val
starting_var = np.array(np.repeat(.1, ndim))
starting_guesses = np.random.random((nwalkers, ndim)) * starting_var + starting_guess
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args = [
    mystep, fit_par, par, par_val, kpc_per_arcsec, phys_const, radius, pix_comp,
    beam_2d, filtering, tf_len, sep, flux_data], threads = nthreads)
sampler.run_mcmc(starting_guesses, nburn)
intermediate = sampler.chain[:,-1,:]
sampler.reset()
sampler.run_mcmc(intermediate, nsteps)
mysamples = sampler.chain.reshape(-1, sampler.chain.shape[2])

## Save the chain
#file = open('catena', 'wb') # create file
#pickle.dump(sampler.chain, file) # write
#file.close()

## Read a saved chain
#chain = pickle.load(open('catena', 'rb' ))
#mysamples = chain.reshape(-1, chain.shape[2])

# Posterior distribution's parameters
param_mean = np.empty(ndim)
param_std = np.empty(ndim)
for ii in np.arange(ndim):
    param_mean[ii] = np.mean(mysamples[:,ii])
    param_std[ii] = np.std(mysamples[:,ii])
    print('Mean(' + fit_par[ii] + '): ' + str(param_mean[ii]) +
          '; Sd(' + fit_par[ii] + '): ' + str(param_std[ii]))
time1 = time.time()
print('Execution time: ' + str(time1 - time0))


### Plots
## Traceplot
traceplot(mysamples, fit_par, plotdir)

## Corner plot
triangle(mysamples, fit_par, plotdir)

## Compton parameter profile for the best fitting parameters (with CI)
best_comp = fit_best(param_mean, fit_par, par, par_val, mystep, kpc_per_arcsec, phys_const, radius, pix_comp, beam_2d, filtering, 
                     tf_len, sep, flux_data, out = 'comp')
# Subset of at most 1000 profiles
out_prof = np.array([fit_best(mysamples[j], fit_par, par, par_val, mystep, kpc_per_arcsec, phys_const, radius, pix_comp, beam_2d, 
                              filtering, tf_len, sep, flux_data, out = 'comp')
                     for j in np.unique(np.linspace(0, mysamples.shape[0] - 1, 1000).astype(int))])
ci = 95 # confidence interval level
quant = np.percentile(out_prof, [50 - ci / 2, 50 + ci / 2], axis = 0)
plot_best(param_mean, fit_par, best_comp, quant[0], quant[1], radius, sep, flux_data, clusdir = plotdir)

## Pressure profile for the best fitting parameters
best_pp = fit_best(param_mean, fit_par, par, par_val, mystep, kpc_per_arcsec, phys_const, radius, pix_comp, beam_2d, filtering, 
                   tf_len, sep, flux_data, out = 'pp')
pp_best(param_mean, fit_par, par, par_val, rad_kpc[1:], plotdir)

## Integrated pressure profile for the best fitting parameters
abel_best(param_mean, fit_par, best_pp, rad_kpc, sep, plotdir)

## Test on the integrated Compton parameter
test_abel_integ(param_mean, fit_par, par, par_val, 900, mystep, kpc_per_arcsec, phys_const)
