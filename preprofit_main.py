from preprofit_funcs import Pressure, mybeam, centdistmat, ima_interpolate, dist, log_lik, traceplot, triangle, fit, plot_best
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

# Pressure parameters
press = Pressure()
pars = press.defPars()
name_pars = list(pars.keys())

# Parameters that we want to fit (among P0, r500, a, b, c)
fit_pars = ['P0', 'r500', 'a']

# Sampling step
mystep = 2 # constant step in arcsec (values higher than 1/3 * FWHM of the PSF are not recommended)

# MCMC parameters
ndim = len(fit_pars)
nwalkers = 100
nthreads = 8
nburn = 1000
nsteps = 1000
np.random.seed(0) # optionally, we set a random seed
ci = 95 # confidence interval level


### Local variables

redshift = 0.888
compt_param_mJy = -10.9 * 10**3 # Compton parameter to Jy/beam conversion factor

# File names
files_dir = 'data'
flux_filename = '%s/press_data_Adam.dat' % files_dir
beam_filename = '%s/Beam150GHz.fits' % files_dir
tf_filename = '%s/TransferFunction150GHz_CLJ1227.fits' % files_dir

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
for i in name_pars:
    if i not in fit_pars:
        pars[i].frozen = True

# Flux density data
flux_data = np.loadtxt(flux_filename, skiprows = 1, unpack = True) # radius (arcsec), flux density, statistical error

# Radius definition
mymaxr = 60 * (np.ceil(flux_data[0][-1] / 60)) # max radius needed (arcsec)
# here we set it to the integer unit of arcmin that follows the highest x-value in the data
radius = np.arange(0, mymaxr + mystep, mystep) # array of radii in arcsec
rad_kpc = radius * kpc_per_arcsec # radius in kpc
radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
sep = radius.size // 2 # index of radius 0

# Basic matrix to represent the 2D image of the integrated Compton parameter
y_mat = centdistmat(mystep, max_dist = mymaxr)

# PSF computation and creation of the 2D image
beam, fwhm_beam = mybeam(beam_filename, radius, regularize = True)
beam_mat = centdistmat(mystep, max_dist = 3 * fwhm_beam) 
# here we use 3 times the FWHM of the PSF as the maximum distance for half-side of the image
beam_2d = ima_interpolate(beam_mat, radius, beam)

# Transfer function
tf_mat_side = y_mat.shape[0] # one side length of the tf image
reso = fwhm_beam # resolution of the map in arcsec (= FWHM of the PSF)
kmax = 1 / reso
karr = dist(tf_mat_side)
karr = karr / np.max(karr) * kmax
tf_data = fits.open(tf_filename)[1].data[0] # transfer function data
wn_as = tf_data[0] # wave number in arcsec^(-1)
tf = tf_data[1] # transmission
f = interp1d(wn_as, tf, fill_value = 'extrapolate') # tf interpolation
filtering = f(karr)

# Compton parameter to Jy/beam conversion
convert = compt_param_mJy
# (...)

# Bayesian fit
starting_guess = [pars[i].val for i in fit_pars]
starting_var = np.array(np.repeat(.1, ndim))
starting_guesses = np.random.random((nwalkers, ndim)) * starting_var + starting_guess
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_lik, args = [
    press, pars, fit_pars, mystep, kpc_per_arcsec, phys_const, radius, y_mat, 
    beam_2d, filtering, sep, flux_data, convert], threads = nthreads)
print('Starting burn-in')
for i, result in enumerate(sampler.sample(starting_guesses, iterations = nburn,
                                          storechain = False)):
    if i % 10 == 0:
        print(' Burn %i / %i (%.1f%%)' % (i, nburn, i * 100 / nburn))
    val = result[0]
print('Finished burn-in \nStarting sampling')
for i, result in enumerate(sampler.sample(val, iterations = nsteps)):
    if i % 10 == 0:
        print(' Sampling %i / %i (%.1f%%)' % (i, nsteps, i * 100 / nsteps))
print('Finished sampling')
print('Acceptance fraction: %s' % np.mean(sampler.acceptance_fraction))
mysamples = sampler.chain.reshape(-1, ndim, order = 'F')

## Save the chain
file = open('mychain', 'wb') # create file
res = list([sampler.chain, sampler.lnprobability])
pickle.dump(res, file) # write
file.close()

# Posterior distribution's parameters
param_med = np.empty(ndim)
param_std = np.empty(ndim)
for i in np.arange(ndim):
    param_med[i] = np.median(mysamples[:,i])
    param_std[i] = np.std(mysamples[:,i])
    print('{:>13}'.format('Median(%s):' % fit_pars[i]) + '%9s' % format(param_med[i], '.3f') + 
          ';{:>12}'.format('Sd(%s):' % fit_pars[i]) + '%9s' % format(param_std[i], '.3f'))


### Plots
## Traceplot
traceplot(mysamples, fit_pars, nsteps, nwalkers, plotdir = plotdir)

## Corner plot
triangle(mysamples, fit_pars, plotdir)

# Random samples of at most 1000 profiles
out_prof = np.array([fit(mysamples[j], press, pars, fit_pars, mystep, kpc_per_arcsec, phys_const, radius, y_mat, beam_2d, 
                         filtering, sep, flux_data, convert) for j in 
                     np.random.choice(mysamples.shape[0], size = min(1000, mysamples.shape[0]), replace = False)])
quant = np.percentile(out_prof, [50, 50 - ci / 2, 50 + ci / 2], axis = 0)

# Best fit
plot_best(param_med, fit_pars, quant[0], quant[1], quant[2], radius, sep, flux_data, ci, plotdir)
