from preprofit_funcs import Press_gNFW, Press_cubspline, read_xy_err, mybeam, centdistmat, read_tf, filt_image, SZ_data, log_lik, prelim_fit, MCMC
from preprofit_plots import traceplot, triangle, best_fit_prof, fitwithmod, press_prof, plot_press
import numpy as np
from astropy.cosmology import Planck18_arXiv_v2 as cosmology
from astropy import units as u
from scipy.interpolate import interp1d
import emcee


### Global variables

# Pressure parameters
press = Press_cubspline()
pars = press.pars
name_pars = list(pars)

# name for outputs
name = 'preprofit'
plotdir = './plots/' # directory for the plots
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

# Cluster cosmology
z = 0.888
cosmology.kpc_per_arcsec = cosmology.kpc_proper_per_arcmin(z).to('kpc arcsec-1')
kpc_as = cosmology.kpc_per_arcsec # number of kpc per arcsec

# Parameters that we want to fit
press.fit_pars = ['pedestal', 'P_0', 'P_1', 'P_2', 'P_3']
press.update_knots([5, 15, 30, 60]*u.arcsec*kpc_as)
# To see the default parameter space extent, use: print(pars)
# For each parameter, use the following to change the bounds of the prior distribution:
#pars['P_0'].minval = 0.1
#pars['P_0'].maxval = 10.

# Sampling step
mystep = 2.*u.arcsec # constant step in arcsec (values higher than (1/3)*FWHM of the beam are not recommended)

R_b = 5000*u.kpc # Radial cluster extent (kpc), serves as upper bound for Compton y parameter integration
t_const = 12*u.keV # constant value of temperature of the cluster (keV), serves for Compton y to mJy/beam conversion

# File names (FITS and ASCII formats are accepted)
files_dir = './data' # files directory
beam_filename = '%s/Beam150GHz.fits' %files_dir 
# The first two columns must be [radius (arcsec), beam]
tf_filename = '%s/TransferFunction150GHz_CLJ1227.fits' %files_dir
flux_filename = '%s/press_clj1226_flagsource.dat' %files_dir
convert_filename = '%s/Compton_to_Jy_per_beam.dat' %files_dir # conversion Compton -> Jy/beam

# Units
flux_units = [u.arcsec, u.Unit('mJy beam-1'), u.Unit('mJy beam-1')]
tf_units = [1/u.arcsec, u.Unit('')]
conv_units = [u.keV, u.Jy/u.beam]

# Beam and transfer function. From raw data or Gaussian approximation?
beam_approx = False
tf_approx = False
fwhm_beam = None # fwhm of the normal distribution for the beam approximation
loc, scale, c = None, None, None # location, scale and normalization parameters of the normal cdf for the tf approximation

# Integrated Compton parameter option
calc_integ = False # apply or do not apply?
integ_mu = .94/1e3 # from Planck 
integ_sig = .36/1e3 # from Planck

# -------------------------------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------------------------------

def main():
    # Parameter definition
    for i in name_pars:
        if i not in press.fit_pars:
            pars[i].frozen = True
    ndim = len(press.fit_pars)

    # Flux density data
    flux_data = read_xy_err(flux_filename, ncol=3, units=flux_units) # radius, flux density, statistical error
    maxr_data = flux_data[0][-1] # highest radius in the data

    # PSF computation and creation of the 2D image
    beam_2d, fwhm = mybeam(mystep, maxr_data, approx=beam_approx, filename=beam_filename, normalize=True, fwhm_beam=fwhm_beam)

    # Radius definition
    mymaxr = (maxr_data+3*fwhm)//mystep*mystep # max radius needed
    radius = np.arange(0., (mymaxr+mystep).value, mystep.value)*mystep.unit # array of radii
    radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
    sep = radius.size//2 # index of radius 0
    r_pp = np.arange((mystep*kpc_as).value, (R_b+mystep*kpc_as).value, (mystep*kpc_as).value)*u.kpc # radius in kpc used to compute the pressure profile

    # Matrix of distances in kpc centered on 0 with step=mystep
    d_mat = centdistmat(radius*kpc_as)

    # Transfer function
    wn_as, tf = read_tf(tf_filename, tf_units=tf_units, approx=tf_approx, loc=loc, scale=scale, c=c) # wave number, transmission
    filtering = filt_image(wn_as, tf, d_mat.shape[0], mystep) # transfer function matrix

    # Compton parameter to mJy/beam conversion
    t_keV, compt_Jy_beam = np.loadtxt(convert_filename, skiprows=1, unpack=True)
    convert = interp1d(t_keV, compt_Jy_beam*1e3, 'linear', fill_value='extrapolate')
    compt_mJy_beam = convert(t_const) # we assume a constant value of temperature

    conv_data = np.loadtxt(convert_filename, skiprows=1, unpack=True) # Temp-dependent conversion Compton to Jy
    t_keV, conv_mJy_beam = map(lambda x, y, z: (x*y).to(z), conv_data, conv_units, ['keV', 'mJy beam-1'])
    convert = interp1d(t_keV, conv_mJy_beam, 'linear', fill_value='extrapolate')
    convert.unit = conv_units
    
    # Set of SZ data required for the analysis
    sz = SZ_data(mystep, kpc_as, compt_mJy_beam, flux_data, beam_2d, radius, sep, r_pp, d_mat, filtering, calc_integ, integ_mu, integ_sig)

    # Bayesian fit
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_lik, args=[pars, press, sz], threads=nthreads)
    # Preliminary fit to increase likelihood
    prelim_fit(sampler, pars, press.fit_pars)
    # construct MCMC object and do burn in
    mcmc = MCMC(sampler, pars, press.fit_pars, seed=seed, initspread=0.1)
    chainfilename = '%s%s_chain.hdf5' % (savedir, name)
    # run mcmc proper and save the chain
    mcmc.mcmc_run(nburn, nlength, nthin)
    mcmc.save(chainfilename)
    cube_chain = mcmc.sampler.chain # (nwalkers x niter x nparams)
    flat_chain = cube_chain.reshape(-1, cube_chain.shape[2], order='F') # ((nwalkers x niter) x nparams)

    # Posterior distribution parameters
    param_med = np.median(flat_chain, axis=0)
    param_std = np.std(flat_chain, axis=0)
    print('{:>6}'.format('|')+'%11s' % 'Median |'+'%11s' % 'Sd |'+'%13s' % 'Unit\n'+'-'*40)
    for i in range(ndim):
        print('{:>6}'.format('%s |' %press.fit_pars[i])+'%9s |' %format(param_med[i], '.3f')+
              '%9s |' %format(param_std[i], '.3f')+'%12s' % [pars[n].unit for n in press.fit_pars][i])
    print('-'*40+'\nChi2 = %s with %s df' % ('{:.4f}'.format(log_lik(param_med, pars, press, sz, output='chisq')), flux_data[1][~np.isnan(flux_data[1])].size-ndim))

    ### Plots
    # Bayesian diagnostics
    traceplot(cube_chain, press.fit_pars, seed=None, plotdir=plotdir)
    triangle(flat_chain, press.fit_pars, show_lines=True, col_lines='r', ci=ci, plotdir=plotdir)

    # Best fitting profile on SZ surface brightness
    perc_sz = best_fit_prof(cube_chain, log_lik, press, sz, ci=ci)
    fitwithmod(sz, perc_sz, ci=ci, plotdir=plotdir)

    # Radial pressure profile
    p_prof = press_prof(cube_chain, press, r_pp, ci=ci)
    plot_press(r_pp, p_prof, ci=ci, plotdir=plotdir)

if __name__ == '__main__':
    main()
