import preprofit_funcs as pfuncs
import preprofit_plots as pplots
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.fftpack import fft2
import emcee
import h5py
from types import MethodType
emcee.moves.move.Move.update = MethodType(pfuncs.update_new, emcee.moves.move.Move.update)


### Global and local variables

## Cluster cosmology
H0 = 70 # Hubble constant at z=0
Om0 = 0.3 # Omega matter
z = 0.888 # redshift
cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)
kpc_as = cosmology.kpc_proper_per_arcmin(z).to('kpc arcsec-1') # number of kpc per arcsec

## Beam and transfer function
# Beam file already includes transfer function?
beam_and_tf = False

# Beam and transfer function. From input data or Gaussian approximation?
beam_approx = False
tf_approx = False
fwhm_beam = None # fwhm of the normal distribution for the beam approximation
loc, scale, c = None, None, None # location, scale and normalization parameters of the normal cdf for the transfer function approximation

# Transfer function provenance (not the instrument, but the team who derived it)
tf_source_team = 'NIKA' # alternatively, 'MUSTANG' or 'SPT'

## File names (FITS and ASCII formats are accepted)
# NOTE: if some of the files are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
# NOTE: if you have beam + transfer function in the same file, assign the name of the file to beam_filename and ignore tf_filename
files_dir = './data' # files directory
beam_filename = '%s/Beam150GHz.fits' %files_dir # beam
tf_filename = '%s/TransferFunction150GHz_CLJ1227.fits' %files_dir # transfer function
flux_filename = '%s/press_clj1226_flagsource.dat' %files_dir # observed data
convert_filename = '%s/Compton_to_Jy_per_beam.dat' %files_dir # conversion Compton -> observed data

# Temperature used for the conversion factor above
t_const = 12*u.keV # if conversion is not required, preprofit ignores it

# Units (here users have to specify units of measurements for the input data, either a list of units for multiple columns or a single unit for a single measure in the file)
# NOTE: if some of the units are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
beam_units = [u.arcsec, u.beam] # beam units
tf_units = [1/u.arcsec, u.Unit('')] # transfer function units
flux_units = [u.arcsec, u.Unit('mJy beam-1'), u.Unit('mJy beam-1')] # observed data units
conv_units = [u.keV, u.Jy/u.beam] # conversion units

# Adopt a cropped version of the beam / beam + transfer function image? Be careful while using this option
crop_image = False # adopt or do not adopt?
cropped_side = 501 # side of the cropped image (automatically set to odd value)

# Maximum radius for line-of-sight Abel integration
R_b = 5000*u.kpc

# Name for outputs
name = 'preprofit'
plotdir = './' # directory for the plots
savedir = './' # directory for saved files

## Prior on the Integrated Compton parameter?
calc_integ = False # apply or do not apply?
integ_mu = .94/1e3 # from Planck 
integ_sig = .36/1e3 # from Planck

## Prior on the pressure slope at large radii?
slope_prior = True # apply or do not apply?
r_out = 1e3*u.kpc # large radius for the slope prior
max_slopeout = -2. # maximum value for the slope at r_out

## Pressure modelization
# 3 models available: 1 parametric (Generalized Navarro Frenk and White), 2 non parametric (cubic spline / power law interpolation)

# Generalized Navarro Frenk and White
press = pfuncs.Press_gNFW(slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

# Cubic spline
#knots = [5, 15, 30, 60]*u.kpc
#press_knots = [1e-1, 2e-2, 5e-3, 1e-4]*u.Unit('keV/cm3')
#press = pfuncs.Press_cubspline(knots=knots, pr_knots=press_knots, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

# Power law interpolation
#rbins = [5, 15, 30, 60]*u.kpc
#pbins = [1e-1, 2e-2, 5e-3, 1e-3]*u.Unit('keV/cm3')
#press = pfuncs.Press_nonparam_plaw(rbins=rbins, pbins=pbins, slope_prior=slope_prior, max_slopeout=max_slopeout)

## Parameters setup
name_pars = list(press.pars) # all parameters
# To see the default parameter space extent, use: print(press.pars)

# To exclude a parameter from the fit:
#press.pars['P_0'].frozen = True
press.pars['c'].frozen = True

# To start the MCMC not too far from the peak of the posterior, you can guess a value of r500 and 
# JoXSZ will automatically apply the parameters of the universal pressure profile defined in Arnaud et al. 2010
# NOTE: this option is available for both parametric and non parametric pressure models
press.set_universal_params(r500=600*u.kpc, cosmo=cosmology, z=z)

# Otherwise, you can customize your set of parameters
# For each parameter, use the following to change the values of the prior distribution, either altogether...
#press.pars['P_0'] = pfuncs.Param(val=1.5, minval=0.1, maxval=10., frozen=False, unit=u.Unit('keV cm-3'))
# ... or separately
#press.pars['P_0'].val = 1.5
#press.pars['P_0'].minval = 0.1
#press.pars['P_0'].maxval = 10.

# To adopt a Gaussian prior:
#press.pars['r_p'] = pfuncs.ParamGaussian(400., prior_mu=300., prior_sigma=50, minval=0.1, unit=u.kpc)

# Sampling step
mystep = 2.*u.arcsec # constant step (values higher than (1/7)*FWHM of the beam are not recommended)

# MCMC parameters
nburn = 2000 # number of burn-in iterations
nlength = 5000 # number of chain iterations (after burn-in)
nwalkers = 30 # number of random walkers
nthreads = 8 # number of processes/threads
nthin = 50 # thinning
seed = None # random seed

# Uncertainty level
ci = 68

# -------------------------------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------------------------------

def main():

    # Parameter definition
    press.fit_pars =  [x for x in press.pars if not press.pars[x].frozen]
    press.max_val = [press.pars[name].maxval for name in press.fit_pars]
    press.min_val = [press.pars[name].minval for name in press.fit_pars]
    ndim = len(press.fit_pars)
    press.indexes = {'ind_'+x: np.array(press.fit_pars)==x if x in press.fit_pars else press.pars[x].val for x in name_pars}

    # Flux density data
    flux_data = pfuncs.read_data(flux_filename, ncol=3, units=flux_units) # radius, flux density, statistical error
    maxr_data = flux_data[0][-1] # highest radius in the data
    press.pars['pedestal'].unit = flux_data[1].unit # automatically update pedestal parameter unit

    # PSF computation and creation of the 2D image
    beam_2d, fwhm = pfuncs.mybeam(mystep, maxr_data, approx=beam_approx, filename=beam_filename, units=beam_units, crop_image=crop_image, cropped_side=cropped_side, 
                                  normalize=True, fwhm_beam=fwhm_beam)

    # The following depends on whether the beam image already includes the transfer function
    if beam_and_tf:
        mymaxr = beam_2d.shape[0]//2*mystep
        filtering = beam_2d.copy()
    else:
        mymaxr = (maxr_data+3*fwhm)//mystep*mystep # max radius needed
        # Transfer function
        wn_as, tf = pfuncs.read_tf(tf_filename, tf_units=tf_units, approx=tf_approx, loc=loc, scale=scale, c=c) # wave number, transmission
        filt_tf = pfuncs.filt_image(wn_as, tf, tf_source_team, beam_2d.shape[0], mystep) # transfer function matrix
        filtering = np.abs(fft2(beam_2d))*filt_tf # filtering matrix including both PSF and transfer function

    # Radius definition
    radius = np.arange(0., (mymaxr+mystep).value, mystep.value)*mystep.unit # array of radii
    radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
    sep = radius.size//2 # index of radius 0
    r_pp = np.arange(mystep.to(u.kpc, equivalencies=eq_kpc_as).value, (R_b.to(u.kpc, equivalencies=eq_kpc_as)+mystep.to(u.kpc, equivalencies=eq_kpc_as)).value, 
                     mystep.to(u.kpc, equivalencies=eq_kpc_as).value)*u.kpc # radius in kpc used to compute the pressure profile (radius 0 excluded)
    r_am = np.arange(0., (mystep*(1+r_pp.size)).to(u.arcmin, equivalencies=eq_kpc_as).value, 
                     mystep.to(u.arcmin, equivalencies=eq_kpc_as).value)*u.arcmin # radius in arcmin (radius 0 included)

    # Matrix of distances in kpc centered on 0 with step=mystep
    d_mat = pfuncs.centdistmat(radius.to(u.kpc, equivalencies=eq_kpc_as))
    
    # If required, temperature-dependent conversion factor from Compton to surface brightness data unit
    if not flux_units[1] == '':
        temp_data, conv_data = pfuncs.read_data(convert_filename, 2, conv_units)
        conv_fun = interp1d(temp_data, conv_data, 'linear', fill_value='extrapolate')
        conv_temp_sb = conv_fun(t_const)*conv_units[1]
    else:
        conv_temp_sb = 1*u.Unit('')

    # Collection of data required for Abel transform calculation
    abel_data = pfuncs.abel_data(r_pp.value)
    
    # Set of SZ data required for the analysis
    sz = pfuncs.SZ_data(step=mystep, conv_temp_sb=conv_temp_sb, flux_data=flux_data, radius=radius, sep=sep, r_pp=r_pp, r_am=r_am, d_mat=d_mat, filtering=filtering, 
                        abel_data=abel_data, calc_integ=calc_integ, integ_mu=integ_mu, integ_sig=integ_sig)

    # Modeled profile resulting from starting parameters VS observed data (useful to adjust parameters if they are way off the target
    if not np.isfinite(pfuncs.log_lik([press.pars[x].val for x in press.fit_pars], press, sz)[0][0]):
        raise Warning('The starting parameters are not in accordance with the prior distributions. Better change them!')
    else:
        start_prof = pfuncs.log_lik([press.pars[x].val for x in press.fit_pars], press, sz, output='bright')
        pplots.plot_guess(start_prof, sz, plotdir=plotdir)
    
    # Bayesian fit
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pfuncs.log_lik, args=[press, sz], threads=nthreads, blobs_dtype=[('bright', list)], vectorize=True)
    except:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pfuncs.log_lik, args=[press, sz], threads=nthreads, vectorize=True)
    # Preliminary fit to increase likelihood
    pfuncs.prelim_fit(sampler, press.pars, press.fit_pars)
    # Construct MCMC object and do burn in
    mcmc = pfuncs.MCMC(sampler, press.pars, press.fit_pars, seed=seed, initspread=0.1)
    chainfilename = '%s%s_chain.hdf5' % (savedir, name)
    # Run mcmc proper and save the chain
    mcmc.mcmc_run(nburn, nlength, nthin, autorefit=False)
    mcmc.save(chainfilename)

    # Extract chain of parameters
    cube_chain = np.array(h5py.File(chainfilename, 'r')['chain']) # (nwalkers x niter x nparams)
    flat_chain = cube_chain.reshape(-1, cube_chain.shape[2], order='F') # ((nwalkers x niter) x nparams)
    # Extract surface brightness profiles
    cube_surbr = np.array(h5py.File(chainfilename, 'r')['bright'])
    flat_surbr = cube_surbr.reshape(-1, cube_surbr.shape[2], order='F')

    # Posterior distribution parameters
    param_med = np.median(flat_chain, axis=0)
    param_std = np.std(flat_chain, axis=0)
    pfuncs.print_summary(press, param_med, param_std, sz)
    pfuncs.save_summary(name, press, param_med, param_std, ci=ci)

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
    
    # Outer slope posterior distribution
    slopes = pfuncs.get_outer_slope(flat_chain, press, r_out)
    pplots.hist_slopes(slopes, ci=ci, plotdir=plotdir)
    
if __name__ == '__main__':
    main()
