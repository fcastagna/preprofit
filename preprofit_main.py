import preprofit_plots as pplots
import preprofit_funcs as pfuncs
import preprofit_likfuncs as lfuncs
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.interpolate import interp1d
import cloudpickle
import pymc as pm
import pytensor.tensor as pt

### Global and local variables

## Cluster cosmology
H0 = 70 # Hubble constant at z=0
Om0 = 0.3 # Omega matter
cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)	

# Cluster
clus = 'SPT-CLJ0500-5116'
z = 0.11 # redshift
# Overdensity measures (set them for defining the starting point for the MCMC)
r500 = 1099*u.kpc 
M500 = (4/3*np.pi*cosmology.critical_density(z).to(u.g/u.kpc**3)*500*r500**3).to(u.Msun)

## Beam and transfer function
# Beam file already includes transfer function?
beam_and_tf = True

# Beam and transfer function. From input data or Gaussian approximation?
beam_approx = True
tf_approx = False
fwhm_beam = [75]*u.arcsec # fwhm of the normal distribution for the beam approximation
loc, scale, k = None, None, None # location, scale and normalization parameters of the normal cdf for the transfer function approximation

# Transfer function provenance (not the instrument, but the team who derived it)
tf_source_team = 'SPT' # choose among 'NIKA', 'MUSTANG' or 'SPT'

## File names (FITS and ASCII formats are accepted)
# NOTE: if some of the files are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
# NOTE: if you have beam + transfer function in the same file, assign the name of the file to beam_filename and ignore tf_filename
files_dir = './data' # files directory
beam_filename = '%s/min_variance_flat_sky_xfer_1p25_arcmin.fits' %files_dir # beam
tf_filename = None # transfer function
flux_filename = '%s/press_data_%s.dat' % (files_dir, clus) # observed data
convert_filename = None # conversion Compton -> observed data
# Temperature used for the conversion factor above
t_const = 8*u.keV # if conversion is not required, preprofit ignores it

# Units (here users have to specify units of measurements for the input data, either a list of units for multiple columns or a single unit for a single 
# measure in the file)
# NOTE: if some of the units are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
# NOTE: base unit is u.Unit(''), e.g. used for Compton y measurements
beam_units = u.Unit('') # beam units
flux_units = [u.arcsec, u.Unit(''), u.Unit('')] # observed data units
# tf_units = [1/u.arcsec, u.Unit('')] # transfer function units
# conv_units = [u.keV, u.Jy/u.beam] # conversion units

# Adopt a cropped version of the beam / beam + transfer function image? Be careful while using this option
crop_image = False # adopt or do not adopt?
cropped_side = 200 # side of the cropped image (automatically set to odd value)

# Maximum radius for line-of-sight Abel integration
R_b = 5000*u.kpc
# Maximum radius for radial profile computation
maxr_data = 1200*u.arcsec

# Name for outputs
name = 'preprofit'
plotdir = './' # directory for the plots
savedir = './' # directory for saved files

## Prior constraint on the Integrated Compton parameter?
calc_integ = False # apply or do not apply?
integ_mu = .94/1e3
integ_sig = .36/1e3

## Prior constraint on the pressure slope at large radii?
slope_prior = True # apply or do not apply?
r_out = (r500.to(u.kpc).value)*1.4 # large radius for the slope prior
max_slopeout = 0. # maximum value for the slope at r_out

## Pressure modelization
# 3 models available: 1 parametric (Generalized Navarro Frenk and White), 2 non parametric (restricted cubic spline / power law interpolation)
# Select your model and please comment the remaining ones

# 1. Generalized Navarro Frenk and White
# press = pfuncs.Press_gNFW(z=z, cosmology=cosmology, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

# 2. Restricted cubic spline
knots = np.outer([.1, .4, .7, 1, 1.3], r500.to(u.kpc).value).T
press = pfuncs.Press_rcs(z=z, cosmology=cosmology, knots=knots, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

# 3. Power law interpolation
# press = pfuncs.Press_nonparam_plaw(z=z, cosmology=cosmology, knots=knots, slope_prior=slope_prior, max_slopeout=max_slopeout)

## Get parameters from the universal pressure profile to be used in the model when setting the prior distributions
logunivpars = press.get_universal_params(M500=M500)[0]

# Sampling step
mystep = 15.*u.arcsec # constant step (values larger than (1/7)*FWHM of the beam are not recommended)
# NOTE: when tf_source_team = 'SPT', be careful to adopt the same sampling step used for the transfer function

# Uncertainty level
ci = 68

# -------------------------------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------------------------------

def main():

    # Flux density data
    flux_data = [pfuncs.read_data(flux_filename, ncol=3, units=flux_units)] # radius, flux density, statistical error

    # Transfer function
    wn_as, tf = [None, None] if beam_and_tf else pfuncs.read_tf(tf_filename, tf_units=tf_units, approx=tf_approx, loc=loc, scale=scale, k=k) # wave number, transmission

    # PSF+tf filtering
    freq, fb, filtering = pfuncs.filtering(mystep, press.eq_kpc_as, maxr_data=maxr_data, approx=beam_approx, filename=beam_filename, beam_and_tf=beam_and_tf, 
                                           crop_image=crop_image, cropped_side=cropped_side, fwhm_beam=fwhm_beam, step_data=15*u.arcsec, w_tf_1d=wn_as, tf_1d=tf)

    # Radius definition
    radius = np.arange(filtering.shape[0]//2+1)*mystep
    radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
    sep = radius.size//2 # index of radius 0
    # radius in kpc used to compute the pressure profile (radius 0 excluded)
    r_pp = [np.arange(1, R_b/mystep.to(u.kpc, equivalencies=press.eq_kpc_as)+1)]*mystep.to(u.kpc, equivalencies=press.eq_kpc_as)
    r_am = np.arange(1+min([len(r) for r in r_pp]))*mystep.to(u.arcmin, equivalencies=press.eq_kpc_as) # radius in arcmin (radius 0 included)

    # If required, temperature-dependent conversion factor from Compton to surface brightness data unit
    if not flux_units[1] == '':
        temp_data, conv_data = pfuncs.read_data(convert_filename, 2, conv_units)
        conv_fun = interp1d(temp_data, conv_data, 'linear', fill_value='extrapolate')
        conv_temp_sb = conv_fun(t_const)*conv_units[1]
    else:
        conv_temp_sb = 1*u.Unit('')

    # Set of SZ data required for the analysis
    sz = pfuncs.SZ_data(clus=clus, step=mystep, eq_kpc_as=press.eq_kpc_as, conv_temp_sb=conv_temp_sb, flux_data=flux_data, radius=radius, sep=sep, 
                        r_pp=r_pp, r_am=r_am, filtering=filtering, calc_integ=calc_integ, integ_mu=integ_mu, integ_sig=integ_sig)

    # Compute P500
    press.P500 = [pfuncs.get_P500((sz.r_pp/r500).value, cosmology, z, M500=M500).value]
    press.r500 = np.atleast_1d(r500)

    # Other indexes
    if type(press) == pfuncs.Press_nonparam_plaw:
        press.ind_low = [np.maximum(0, np.digitize(sz.r_pp[0], press.knots[0])-1)] # lower bins indexes
        press.r_low = [p[i] for p, i in zip(press.knots, press.ind_low)] # lower radial bins
        press.alpha_ind = [np.minimum(i, len(press.knots[0])-2) for i in press.ind_low] # alpha indexes
    if type(press) == pfuncs.Press_rcs:
        press.kn = [pt.log10(k/r) for k, r in zip(press.knots, press.r500)]
        press.sv = [[(kn > kn[_])*(kn-kn[_])**3-(kn > kn[-2])*(kn-kn[_])*(kn-kn[-2])**2
                     for _ in range(press.N[i])] for i, kn in enumerate(press.kn)]
        press.X = [pt.concatenate((pt.atleast_2d(pt.ones(len(press.knots[i]))), pt.atleast_2d(kn), pt.as_tensor(sv))).T
                   for i, (kn, sv) in enumerate(zip(press.kn, press.sv))]

    # Save objects
    with open('%s/press_obj.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(press, f, -1)
    with open('%s/szdata_obj.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(sz, f, -1)

    ## Model definition
    with pm.Model() as model:
        # Customize the prior distribution of the parameters using pymc distributions
        if type(press) == pfuncs.Press_gNFW:
            fitted = ['Ps', 'a', 'b', 'c'] # parameters that we aim to fit
            nps = len(fitted)
            pm.Normal('Ps', mu=logunivpars[0], sigma=.3) if 'Ps' in fitted else None
            pm.Normal('a', mu=logunivpars[1], sigma=.1) if 'a' in fitted else None
            pm.Normal('b', mu=logunivpars[2], sigma=.1) if 'b' in fitted else None
            pm.Normal('c', mu=logunivpars[3], sigma=.1) if 'c' in fitted else None
            c500=1.177
            logr_p = np.log10(r500.value/c500)
        else:
            nk = len(logunivpars)
            pm.Normal('lgP_k', mu=logunivpars, sigma=.5, initval=logunivpars)
        # Add pedestal component to the model
        pm.Normal("ped", 0., 1e-6, initval=0.)        
        
        # Likelihood function
        lprof, pprof, mprof, slopes = lfuncs.whole_lik(
            model, model['lgP_k'], model['ped'], press, sz.r_pp[0].value, sz.r_red[0].value, sz.abel_data[0], sz.filtering.value, sz.conv_temp_sb.value, 
            sz.dist.labels[0], sz.sep, sz.dist.d_mat[0], sz.radius[sz.sep:].value, [s.value for s in sz.flux_data[0]], 'll')
        pm.Normal('like', mu=lprof, sigma=sz.flux_data[0][2], observed=sz.flux_data[0][1], shape=len(sz.flux_data[0][1]))

        # Save useful measures
        pm.Deterministic('bright', mprof)
        pm.Deterministic('press', pprof)
        if slope_prior:
            pm.Deterministic('slope', slopes)

        ## Sampling
        start_guess = model['bright'].eval({str(p): model.rvs_to_initial_values[model.named_vars[str(p)]] for p in model.free_RVs})
        pplots.plot_guess(start_guess, sz, press, fact=1e4, plotdir=plotdir)
        # Fit
        trace = pm.sample(draws=1000, tune=1000, chains=8, initvals=model.rvs_to_initial_values)

    # Save model and trace
    with open('%s/model.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(model, f, -1)
    trace.to_netcdf("%s/trace.nc" % savedir)

    # Extract chain of parameters ((nwalkers x niter) x nparams)
    prs = [str(_) for _ in model.free_RVs]
    samples = []
    for (i, par) in enumerate(prs):
        res = trace.posterior[par].data.reshape(np.prod(trace.posterior[prs[0]].shape[:2]), -1)
        for j in range(res.shape[1]):
            samples.append(res[:,j])
    samples = np.array(samples).T
    prs_ext = ['%s%s' % (prs[0][:-1], i) for i in range(nk)]+[prs[-1]]

    # Extract surface brightness profiles
    flat_surbr = np.array([trace.posterior['bright']]).reshape(1, samples.shape[0], -1)
    # Median surface brightness profile + CI
    perc_sz = np.array([pplots.get_equal_tailed(f, ci=ci) for f in flat_surbr])

    # Posterior distributions summary
    pm.summary(trace, var_names=prs)

    # Traceplot
    pplots.traceplot(trace, prs, prs_ext, fact_ped=1e4, compact=False, plotdir=savedir)

    # Best fitting profile on SZ surface brightness
    pplots.fitwithmod(sz, perc_sz, press.eq_kpc_as, rbins=None if type(press)==pfuncs.Press_gNFW else (
        press.knots*u.kpc).to(u.arcsec, equivalencies=press.eq_kpc_as).value, peds=[trace.posterior['ped'].data.mean()], fact=1e5, ci=ci, plotdir=plotdir)

    # Cornerplots
    pplots.triangle(samples, prs_ext, model, fact_ped=1e4, plot_prior=True, show_lines=True, col_lines='r', ci=ci, plotdir=plotdir)

    # Radial pressure profile
    p_prof = [trace.posterior['press'].data.reshape(samples.shape[0], -1)]
    p_quant = [pplots.get_equal_tailed(pp, ci=ci) for pp in p_prof]
    pplots.plot_press(sz.r_pp, p_quant, clus=sz.clus, ci=ci, plotdir=plotdir, rbins=None if type(press)==pfuncs.Press_gNFW else press.knots)

    # Outer slope posterior distribution
    if press.slope_prior:
        slopes = np.array([trace.posterior['slope'].data.flatten()]).flatten()
        pplots.hist_slopes(slopes.flatten(), ci=ci, plotdir=plotdir)

if __name__ == '__main__':
    main()
