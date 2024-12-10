import preprofit_funcs as pfuncs
import preprofit_plots as pplots
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.interpolate import interp1d
import cloudpickle
import pymc as pm
from pytensor import shared
import pytensor.tensor as pt
from pymc.sampling.mcmc import assign_step_methods

### Global and local variables

## Cluster cosmology
H0 = 70 # Hubble constant at z=0
Om0 = 0.3 # Omega matter
cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)	

# Cluster
clus = ['SPT-CLJ0500-5116', 'SPT-CLJ0637-4829', 'SPT-CLJ2055-5456']
nc = len(clus)
print('%s Clusters: %s' % (nc, clus))
z = [.11, .2026, .139]
r500 = [ 943.85207035, 1290.31531693, 1022.3744362 ]*u.kpc
M500 = (4/3*np.pi*cosmology.critical_density(z).to(u.g/u.kpc**3)*500*r500**3).to(u.Msun)

kpc_as = cosmology.kpc_proper_per_arcmin(z).to('kpc arcsec-1') # number of kpc per arcsec
eq_kpc_as = [(u.arcsec, u.kpc, lambda x: x*kpc_as.value, lambda x: x/kpc_as.value)] # equation for switching between kpc and arcsec

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
flux_filename = ['%s/press_data_' %files_dir+cl+'.dat' for cl in clus] # observed data
convert_filename = None # conversion Compton -> observed data

# Temperature used for the conversion factor above
t_const = 8*u.keV # if conversion is not required, preprofit ignores it

# Units (here users have to specify units of measurements for the input data, either a list of units for multiple columns or a single unit for a single 
# measure in the file)
# NOTE: if some of the units are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
# NOTE: base unit is u.Unit(''), e.g. used for Compton y measurements
beam_units = u.beam # beam units
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
integ_mu = .94/1e3 # from Planck
integ_sig = .36/1e3 # from Planck

## Prior constraint on the pressure slope at large radii?
slope_prior = True # apply or do not apply?
r_out = r500*1.4 # large radius for the slope prior
max_slopeout = 0. # maximum value for the slope at r_out

## Pressure modelization
# 3 models available: 1 parametric (Generalized Navarro Frenk and White), 2 non parametric (restricted cubic spline / power law interpolation)
# Select your model and please comment the remaining ones

# 1. Generalized Navarro Frenk and White
# press = pfuncs.Press_gNFW(eq_kpc_as=eq_kpc_as, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

# 2. Restricted cubic spline
knots = np.outer([.1, .4, .7, 1, 1.3], r500).T
press = pfuncs.Press_rcs(knots=knots, eq_kpc_as=eq_kpc_as, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

# 3. Power law interpolation
# press = pfuncs.Press_nonparam_plaw(knots=knots, eq_kpc_as=eq_kpc_as, slope_prior=slope_prior, max_slopeout=max_slopeout)

## Get parameters from the universal pressure profile to determine starting point
logunivpars = press.get_universal_params(cosmology, z, M500=M500)
press_knots = np.mean(logunivpars, axis=0)
nk = len(press_knots)

## Model definition
with pm.Model() as model:
    # Customize the prior distribution of the parameters using pymc distributions
    pm.Uniform('sigmas', 0, 1, shape=nk)
    pm.Normal('log(P_k)', mu=press_knots, sigma=.5, initval=press_knots, shape=nk)
    [pm.Normal('log(P_{%s,i})' % j, mu=model['log(P_k)'][j], sigma=model['sigmas'][j], shape=nc) for j in range(nk)]

# Sampling step
mystep = 30.*u.arcsec # constant step (values larger than (1/7)*FWHM of the beam are not recommended)
# NOTE: when tf_source_team = 'SPT', be careful to adopt the same sampling step used for the transfer function

# Uncertainty level
ci = 68

# -------------------------------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------------------------------

def main():

    # Flux density data
    flux_data = [pfuncs.read_data(fl, ncol=3, units=flux_units) for fl in flux_filename] # radius, flux density, statistical error

    # Transfer function
    wn_as, tf = None, None
    # wn_as, tf = pfuncs.read_tf(tf_filename, tf_units=tf_units, approx=tf_approx, loc=loc, scale=scale, k=k) # wave number, transmission

    # PSF+tf filtering
    ell_spt, tf_1d = np.loadtxt('./data/sptsz_trough_filter_1d.dat', unpack=1)
    freq_spt_1d = (ell_spt/u.radian).to(1/u.arcsec)/2/np.pi
    freq, fb, filtering = pfuncs.filtering(mystep, eq_kpc_as, maxr_data=maxr_data, approx=beam_approx, beam_and_tf=beam_and_tf, crop_image=crop_image, 
                                           cropped_side=cropped_side, fwhm_beam=fwhm_beam, step_data=15*u.arcsec, w_tf_1d=freq_spt_1d, tf_1d=tf_1d)
    fwhm = fwhm_beam

    # Radius definition
    radius = np.arange(filtering.shape[0]//2+1)*mystep
    radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
    sep = radius.size//2 # index of radius 0
    # radius in kpc used to compute the pressure profile (radius 0 excluded)
    r_pp = [np.arange(1, R_b/mystep.to(u.kpc, equivalencies=eq_kpc_as)[i]+1)*mystep.to(u.kpc, equivalencies=eq_kpc_as)[i]
            for i in range(nc)]
    r_am = np.arange(1+min([len(r) for r in r_pp]))*mystep.to(u.arcmin, equivalencies=eq_kpc_as) # radius in arcmin (radius 0 included)

    # If required, temperature-dependent conversion factor from Compton to surface brightness data unit
    if not flux_units[1] == '':
        temp_data, conv_data = pfuncs.read_data(convert_filename, 2, conv_units)
        conv_fun = interp1d(temp_data, conv_data, 'linear', fill_value='extrapolate')
        conv_temp_sb = conv_fun(t_const)*conv_units[1]
    else:
        conv_temp_sb = 1*u.Unit('')

    # Set of SZ data required for the analysis
    sz = pfuncs.SZ_data(clus=clus, step=mystep, eq_kpc_as=eq_kpc_as, conv_temp_sb=conv_temp_sb, flux_data=flux_data, radius=radius, sep=sep, 
                        r_pp=r_pp, r_am=r_am, filtering=filtering, calc_integ=calc_integ, integ_mu=integ_mu, integ_sig=integ_sig)

    # Compute P500
    press.P500 = [pfuncs.get_P500((sz.r_pp[j]/r500[j]).value, cosmology, z[j], M500=M500[j]).value for j in range(nc)]
    press.r500 = [r for r in r500]
    press.clus = clus
    press.z = z

    # Other indexes
    if type(press) == pfuncs.Press_nonparam_plaw:
        press.ind_low = [np.maximum(0, np.digitize(sz.r_pp[i], press.knots[i])-1) for i in range(nc)] # lower bins indexes
        press.r_low = [p[i] for p, i in zip(press.knots, press.ind_low)] # lower radial bins
        press.alpha_ind = [np.minimum(press.ind_low[i], len(press.knots[i])-2) for i in range(nc)] # alpha indexes
    if type(press) == pfuncs.Press_rcs:
        press.kn = [pt.log10(k.value/r.value) for k, r in zip(press.knots, press.r500)]
        press.sv = [[(kn > kn[_])*(kn-kn[_])**3-(kn > kn[-2])*(kn-kn[_])*(kn-kn[-2])**2
                     for _ in range(press.N[i])] for i, kn in enumerate(press.kn)]
        press.X = [pt.concatenate((pt.atleast_2d(pt.ones(len(press.knots[i]))), pt.atleast_2d(kn), pt.as_tensor(sv))).T
                   for i, (kn, sv) in enumerate(zip(press.kn, press.sv))]

    # Save objects
    with open('%s/press_obj.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(press, f, -1)
    with open('%s/szdata_obj.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(sz, f, -1)

    # Add pedestal component to the model
    with model:
        pm.Normal("peds", 0, 1e-6, shape=nc)

    ## Sampling
    ilike = pt.as_tensor([np.inf])
    nn = 0
    while np.isinf(pt.sum(ilike).eval()):
        print('---\nSearching...')
        if nn > 0:
            pm.draw([m for m in model.free_RVs])
        vals = [x.eval() for x in model.free_RVs]
        if type(press) == pfuncs.Press_gNFW:
            pars = [[model.rvs_to_transforms[model.values_to_rvs[m]].forward(m2.eval(), *m2.owner.inputs) 
                     if model.rvs_to_transforms[model.values_to_rvs[m]] is not None else m2 
                     for m, m2, v in zip(model.continuous_value_vars[:nps], model.free_RVs[:nps], vals[:nps])]+
                    [model.rvs_to_transforms[model.values_to_rvs[m]].forward(m2.eval(), *m2.owner.inputs)
                     if model.rvs_to_transforms[model.values_to_rvs[m]] is not None else m2
                     for m, m2 in zip(model.continuous_value_vars[2*nps:][i:nc*nps:nc], model.free_RVs[2*nps:][i:nc*nps:nc])]+
                    [logr_p[i]]+[model['peds_%s' %i]] for i in range(nc)]
        else:
            pars = [[model.rvs_to_transforms[model.values_to_rvs[m]].forward(m2.eval(), *m2.owner.inputs)
                     if model.rvs_to_transforms[model.values_to_rvs[m]] is not None else m2
                     for m, m2, v in zip(model.continuous_value_vars[:2], model.free_RVs[:2], vals[:2])]+
                    [model.rvs_to_transforms[model.values_to_rvs[m]].forward(m2.eval(), *m2.owner.inputs)
                     if model.rvs_to_transforms[model.values_to_rvs[m]] is not None else m2
                     for m, m2, v in zip(model.continuous_value_vars[2:2+nk], model.free_RVs[2:2+nk], vals[2:2+nk])]+
                    [model['peds']]]
        with model:
            pars = [p[nps:] for p in pars] if type(press)==pfuncs.Press_gNFW else [[p[j] for p in pars[0][2:]] for j in range(nc)]
            like, pprof, maps, slopes = zip(*map(
                lambda i, pr, szr, szrd, sza, szl, dm, szfl: pfuncs.whole_lik(
                    pr, press, szr, szrd, sza, sz.dist.indices, sz.filtering, sz.conv_temp_sb, szl, sz.sep, dm, sz.radius[sz.sep:].value, szfl, i, 'll'),
                np.arange(nc), pars, sz.r_pp, sz.r_red, sz.abel_data, sz.dist.labels, sz.dist.d_mat, sz.flux_data))
            infs = [int(np.isinf(l.eval())) for l in like]
            print('likelihood:')
            print(pt.sum(like).eval())
            # [model.set_initval(n, v) for n, v in zip(model.free_RVs, vals)]
        if np.sum(infs) == 0:
            check = pt.sum(like).eval()
            print('logp: %f' % check)
            df = np.array(pars).flatten().size
            red_chisq = -2*check/df
            print('Reduced chisq: %f' % red_chisq)
            if red_chisq > 300:
                print('Too high! Retry')
                pm.draw([m for m in model.free_RVs])
            else:
                ilike = pt.sum([shared(check)])
        nn += 1
        if nn == 1000:
            raise RuntimeError('Valid starting point not found after 1000 attempts. Execution stopped')

    with model:
        map_prof = [pm.Deterministic('bright%s' % i, maps[i]) for i in range(nc)]
        p_prof = [pm.Deterministic('press%s' % i, pprof[i]) for i in range(nc)]
        like = pm.Potential('like', pt.sum(like))
        if slope_prior:
            slope = [pm.Deterministic('slope%s' % i, slopes[i]) for i in range(nc)]
        with open('%s/model.pickle' % savedir, 'wb') as m:
            cloudpickle.dump(model, m, -1)
        start_guess = [np.atleast_2d(m.eval()) for m in map_prof]
    pplots.plot_guess(start_guess, sz, knots=None if type(press) == pfuncs.Press_gNFW else 
                      [[r.to(sz.flux_data[0][0].unit, equivalencies=eq_kpc_as)[j].value for i, r in enumerate(press.knots[j])] 
                       for j in range(nc)], plotdir=plotdir)

    with model:
        step = assign_step_methods(model, None, methods=pm.STEP_METHODS, step_kwargs={})
        trace = pm.sample(draws=1000, tune=1000, chains=8, return_inferencedata=True, step=step,
                          initvals=model.initial_point())
    
    trace.to_netcdf("%s/trace.nc" % savedir)

if __name__ == '__main__':
    main()
