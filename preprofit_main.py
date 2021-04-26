import preprofit_funcs as pfuncs
import preprofit_plots as pplots
import numpy as np
try:
    from astropy.cosmology import Planck18_arXiv_v2 as cosmology
except:
    from astropy.cosmology import Planck15 as cosmology
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.fftpack import fft2
import emcee
from itertools import chain


### Global variables

# Cluster cosmology
z = 0.888
kpc_as = cosmology.kpc_proper_per_arcmin(z).to('kpc arcsec-1') # number of kpc per arcsec

### Pressure modelization
slope_prior = True # prior on the outer slope?
r_out = 1e3*u.kpc # outer radius
max_slopeout = -2. # outer slope constrain
## Parametric model
# Generalized Navarro Frenk and White
press = pfuncs.Press_gNFW(slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)
## Non parametric
# Cubic spline
#press = pfuncs.Press_cubspline(slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)
#press.knots = [5, 15, 30, 60]*u.arcsec*kpc_as
# Power law interpolation
#press = pfuncs.Press_nonparam_plaw(slope_prior=slope_prior, max_slopeout=max_slopeout)
#press.bins = [5, 15, 30, 60]*u.arcsec*kpc_as

# Parameters that we want to fit
name_pars = list(press.pars)
# To see the default parameter space extent, use: print(press.pars)
# For each parameter, use the following to change the bounds of the prior distribution:
#press.pars['P_0'].minval = 0.1
#press.pars['P_0'].maxval = 10.
# To exclude a parameter from the fit:
#press.pars['P_0'].frozen = True
press.pars['c'].frozen = True

# name for outputs
name = 'preprofit'
plotdir = './' # directory for the plots
savedir = './' # directory for saved files

# Uncertainty level
ci = 95

# MCMC parameters
nburn = 2000 # number of burn-in iterations
nlength = 5000 # number of chain iterations (after burn-in)
nwalkers = 30 # number of random walkers
nthreads = 8 # number of processes/threads
nthin = 50 # thinning
seed = None # random seed


### Local variables

# Sampling step
mystep = 2.*u.arcsec # constant step in arcsec (values higher than (1/7)*FWHM of the beam are not recommended)

R_b = 5000*u.kpc # Radial cluster extent (kpc), serves as upper bound for Compton y parameter integration
t_const = 12*u.keV # constant value of temperature of the cluster (keV), serves for Compton y to mJy/beam conversion

# File names (FITS and ASCII formats are accepted)
files_dir = './data' # files directory
beam_filename = '%s/Beam150GHz.fits' %files_dir
tf_filename = '%s/TransferFunction150GHz_CLJ1227.fits' %files_dir
flux_filename = '%s/press_clj1226_flagsource.dat' %files_dir
convert_filename = '%s/Compton_to_Jy_per_beam.dat' %files_dir # conversion Compton -> Jy/beam

# Units
beam_units = [u.arcsec, u.beam]
flux_units = [u.arcsec, u.Unit('mJy beam-1'), u.Unit('mJy beam-1')]
tf_units = [1/u.arcsec, u.Unit('')]
conv_units = [u.keV, u.Jy/u.beam]

# Beam and transfer function. From raw data or Gaussian approximation?
beam_approx = False
tf_approx = False
fwhm_beam = None # fwhm of the normal distribution for the beam approximation
loc, scale, c = None, None, None # location, scale and normalization parameters of the normal cdf for the tf approximation

# Transfer function provenance (not the instrument, but the team who derived it)
tf_source_team = 'NIKA' # alternatively, 'MUSTANG' or 'SPT'

# Beam file already includes transfer function?
beam_and_tf = False

# Integrated Compton parameter option
calc_integ = False # apply or do not apply?
integ_mu = .94/1e3 # from Planck 
integ_sig = .36/1e3 # from Planck

# -------------------------------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------------------------------

def main():

    # Parameter definition
    press.fit_pars =  [x for x in press.pars if not press.pars[x].frozen]
    ndim = len(press.fit_pars)

    # Flux density data
    flux_data = pfuncs.read_data(flux_filename, ncol=3, units=flux_units) # radius, flux density, statistical error
    maxr_data = flux_data[0][-1] # highest radius in the data
    press.pars['pedestal'].unit = str(flux_data[1].unit) # automatically update pedestal parameter unit

    # PSF computation and creation of the 2D image
    beam_2d, fwhm = pfuncs.mybeam(mystep, maxr_data, approx=beam_approx, filename=beam_filename, units=beam_units, normalize=True, fwhm_beam=fwhm_beam)

    # The following depends on whether the beam image already includes the transfer function
    if beam_and_tf:
        mymaxr = beam_2d.shape[0]//2*mystep
        filtering = beam_2d.copy()
    else:
        mymaxr = (maxr_data+3*fwhm)//mystep*mystep # max radius needed
        # Transfer function
        wn_as, tf = pfuncs.read_tf(tf_filename, tf_units=tf_units, approx=tf_approx, loc=loc, scale=scale, c=c) # wave number, transmission
        filt_tf = pfuncs.filt_image(wn_as, tf, tf_source_team, beam_2d.shape[0], mystep) # transfer function matrix
        filtering = fft2(beam_2d)*filt_tf # filtering matrix including both PSF and transfer function

    # Radius definition
    radius = np.arange(0., (mymaxr+mystep).value, mystep.value)*mystep.unit # array of radii
    radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
    sep = radius.size//2 # index of radius 0
    r_pp = np.arange((mystep*kpc_as).value, (R_b+mystep*kpc_as).value, (mystep*kpc_as).value)*u.kpc # radius in kpc used to compute the pressure profile

    # Matrix of distances in kpc centered on 0 with step=mystep
    d_mat = pfuncs.centdistmat(radius*kpc_as)
    
    # If required, temperature-dependent conversion factor from Compton to surface brightness data unit
    if not flux_units[1] == '':
        temp_data, conv_data = pfuncs.read_data(convert_filename, 2, conv_units)
        conv_fun = interp1d(temp_data, conv_data, 'linear', fill_value='extrapolate')
        conv_temp_sb = conv_fun(t_const)*conv_units[1]
    else:
        conv_temp_sb = 1*u.Unit('')

    # Set of SZ data required for the analysis
    sz = pfuncs.SZ_data(mystep, kpc_as, conv_temp_sb, flux_data, beam_2d, radius, sep, r_pp, d_mat, filtering, calc_integ, integ_mu, integ_sig)

    # Bayesian fit
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pfuncs.log_lik, args=[press, sz], threads=nthreads, blobs_dtype=[('bright', list)])
    except:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pfuncs.log_lik, args=[press, sz], threads=nthreads)
    # Preliminary fit to increase likelihood
    pfuncs.prelim_fit(sampler, press.pars, press.fit_pars)
    # construct MCMC object and do burn in
    mcmc = pfuncs.MCMC(sampler, press.pars, press.fit_pars, seed=seed, initspread=0.1)
    chainfilename = '%s%s_chain.hdf5' % (savedir, name)
    # run mcmc proper and save the chain
    mcmc.mcmc_run(nburn, nlength, nthin)
    mcmc.save(chainfilename)
    cube_chain = mcmc.sampler.chain # (nwalkers x niter x nparams)
    flat_chain = cube_chain.reshape(-1, cube_chain.shape[2], order='F') # ((nwalkers x niter) x nparams)
    cube_surbr = np.array([list(chain.from_iterable(x)) for x in zip(*mcmc.sampler.blobs)]).reshape(nwalkers, cube_chain.shape[1], sep+1)
    flat_surbr = cube_surbr.reshape(-1, cube_surbr.shape[2], order='F')

    # Posterior distribution parameters
    param_med = np.median(flat_chain, axis=0)
    param_std = np.std(flat_chain, axis=0)
    pfuncs.print_summary(press, param_med, param_std, sz)

    ### Plots
    # Bayesian diagnostics
    pplots.traceplot(cube_chain, press.fit_pars, seed=None, plotdir=plotdir)
    pplots.triangle(flat_chain, press.fit_pars, show_lines=True, col_lines='r', ci=ci, plotdir=plotdir)

    # Best fitting profile on SZ surface brightness
    perc_sz = pplots.get_equal_tailed(flat_surbr, ci=ci)
    pplots.fitwithmod(sz, perc_sz, ci=ci, plotdir=plotdir)

    # Radial pressure profile
    p_prof = pplots.press_prof(cube_chain, press, r_pp, ci=ci)
    pplots.plot_press(r_pp, p_prof, ci=ci, plotdir=plotdir)

if __name__ == '__main__':
    main()
