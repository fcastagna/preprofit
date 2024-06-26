import preprofit_funcs as pfuncs
import preprofit_plots as pplots
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.fftpack import fft2
import cloudpickle
import pymc as pm
from pytensor import shared
import arviz as az
import pytensor.tensor as pt
import astropy.constants as const
from pymc.sampling.mcmc import assign_step_methods

### Global and local variables

## Cluster cosmology
H0 = 70 # Hubble constant at z=0
Om0 = 0.3 # Omega matter
cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)	

# Cluster
clus = 'SPT-CLJ0500-5116'
z = 0.11 # redshift
M500 = 4.2e14*u.Msun  # M500
r500 = ((3/4*M500/(500.*cosmology.critical_density(z)*np.pi))**(1/3)).to(u.kpc)

kpc_as = cosmology.kpc_proper_per_arcmin([z]).to('kpc arcsec-1') # number of kpc per arcsec
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
flux_filename = '%s/press_data_%s.dat' % (files_dir, clus) # observed data
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
maxr_data = 1080*u.arcsec-3*fwhm_beam

# Name for outputs
name = 'preprofit'
plotdir = './' # directory for the plots
savedir = './' # directory for saved files

## Prior on the Integrated Compton parameter?
calc_integ = False # apply or do not apply?
integ_mu = .94/1e3 # from Planck
integ_sig = .36/1e3 # from Planck

## Prior on the pressure slope at large radii? (for example from Planck)
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
std_knots = np.std(logunivpars, axis=0)
nk = len(press_knots)

## Model definition
with pm.Model() as model:
    # Customize the prior distribution of the parameters using pymc distributions
    if type(press) == pfuncs.Press_gNFW:
        fitted = ['Ps', 'a', 'b', 'c'] # parameters that we aim to fit
        nps = len(fitted)
        pm.Normal('Ps', mu=logunivpars[0][0], sigma=.3) if 'Ps' in fitted else None
        pm.Normal('a', mu=logunivpars[0][1], sigma=.1) if 'a' in fitted else None
        pm.Normal('b', mu=logunivpars[0][2], sigma=.1) if 'b' in fitted else None
        pm.Normal('c', mu=logunivpars[0][3], sigma=.1) if 'c' in fitted else None
        c500=1.177
        logr_p = np.log10(r500.value/c500)
    else:
        [pm.Normal('P'+str(i), mu=press_knots[i], sigma=.5, initval=press_knots[i]) for i in range(nk)]
    # Add pedestal component to the model
    pm.Normal("ped", 0, 1e-6)

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

    # PSF computation and creation of the 2D image
    beam_2d, fwhm = pfuncs.mybeam(mystep, maxr_data, eq_kpc_as=eq_kpc_as, approx=beam_approx, filename=beam_filename, units=beam_units, 
                                  crop_image=crop_image, cropped_side=cropped_side, normalize=True, fwhm_beam=fwhm_beam)

    # The following depends on whether the beam image already includes the transfer function
    if beam_and_tf:
        filtering = fft2(beam_2d)
        if crop_image:
            from scipy.fftpack import fftshift, ifftshift
            filtering = ifftshift(pfuncs.get_central(fftshift(filtering), cropped_side))
    else:
        # Transfer function
        wn_as, tf = pfuncs.read_tf(tf_filename, tf_units=tf_units, approx=tf_approx, loc=loc, scale=scale, k=k) # wave number, transmission
        filt_tf = pfuncs.filt_image(wn_as, tf, tf_source_team, beam_2d.shape[0], mystep, eq_kpc_as) # transfer function matrix
        filtering = fft2(beam_2d)*filt_tf # filtering matrix including both PSF and transfer function

    # Radius definition
    mymaxr = [filtering.shape[0]//2*mystep if crop_image else (maxr_data+3*fwhm.to(maxr_data.unit, equivalencies=eq_kpc_as))//
              mystep.to(maxr_data.unit, equivalencies=eq_kpc_as)*mystep.to(maxr_data.unit, equivalencies=eq_kpc_as)][0] # max radius needed
    radius = np.arange(0., (mymaxr+mystep.to(mymaxr.unit, equivalencies=eq_kpc_as))[0].value, 
                       mystep.to(mymaxr.unit, equivalencies=eq_kpc_as).value)*mymaxr.unit # array of radii
    radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
    sep = radius.size//2 # index of radius 0
    # radius in kpc used to compute the pressure profile (radius 0 excluded)
    r_pp = [np.arange(mystep.to(u.kpc, equivalencies=eq_kpc_as)[0].value, 
                     (R_b.to(u.kpc, equivalencies=eq_kpc_as)+mystep.to(u.kpc, equivalencies=eq_kpc_as))[0].value, 
                      mystep.to(u.kpc, equivalencies=eq_kpc_as)[0].value)]*u.kpc
    r_am = np.arange(0., (mystep*(1+min([len(r) for r in r_pp]))).to(u.arcmin, equivalencies=eq_kpc_as).value,
                     mystep.to(u.arcmin, equivalencies=eq_kpc_as).value)*u.arcmin # radius in arcmin (radius 0 included)

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

    # Compute P500 according to the definition in Equation (5) from Arnaud's paper
    mu, mu_e, f_b = .59, 1.14, .175
    pnorm = mu/mu_e*f_b*3/8/np.pi*(const.G.value**(-1/3)*u.kg/u.m/u.s**2).to(u.keV/u.cm**3)/((u.kg/250**2/cosmology.H0**4/u.s**4/3e14/u.Msun).to(''))**(2/3)
    alpha_P = 1/.561-5/3
    alpha1_P = lambda x: .1-(alpha_P+.1)*(x/.5)**3/(1+(x/.5)**3)
    hz = cosmology.H(z)/cosmology.H0
    def conv(x): 
        return pnorm*hz**(8/3)*(M500/3e14/u.Msun)**(2/3)*(M500/3e14/u.Msun)**(alpha_P+alpha1_P(x))
    press.P500 = []
    press.P500.append(conv((sz.r_pp/r500).value).value)
    press.r500 = np.atleast_1d(r500)
    
    # Other indexes
    if type(press) == pfuncs.Press_nonparam_plaw:
        press.ind_low = [np.maximum(0, np.digitize(sz.r_pp[0], press.knots[0])-1)] # lower bins indexes
        press.r_low = [p[i] for p, i in zip(press.knots, press.ind_low)] # lower radial bins
        press.alpha_ind = [np.minimum(i, len(press.knots[0])-2) for i in press.ind_low] # alpha indexes
    
    # Save objects
    with open('%s/press_obj.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(press, f, -1)
    with open('%s/szdata_obj.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(sz, f, -1)

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
                    [logr_p[0]]+[model['ped']]]
        else:
            pars = [[model.rvs_to_transforms[model.values_to_rvs[m]].forward(m2.eval(), *m2.owner.inputs) 
                     if model.rvs_to_transforms[model.values_to_rvs[m]] is not None else m2 
                     for m, m2, v in zip(model.continuous_value_vars[:nk], model.free_RVs[:nk], vals[:nk])]+[model['ped']]]
        with model:
            like, pprof, maps, slopes = zip(*map(
                lambda i, pr, szr, sza, szl, dm, szfl: pfuncs.whole_lik(
                    pr, press, szr, sza, sz.filtering, sz.conv_temp_sb, szl, sz.sep, dm, sz.radius[sz.sep:].value, szfl, i, 'll'), 
                np.arange(1), pars, sz.r_pp, sz.abel_data, sz.dist.labels, sz.dist.d_mat, sz.flux_data))
            pm.Potential('pv_like'+str(nn), pt.sum(like))
            infs = [int(np.isinf(l.eval())) for l in like]
            print('likelihood:')
            print(pt.sum(like).eval())
            [model.set_initval(n, v) for n, v in zip(model.free_RVs, vals)]
        if np.sum(infs) == 0:
            check = pt.sum(like).eval()
            print('logp: %f' % check)
            df = np.array(pars).flatten().size
            red_chisq = -2*check/df
            print('Reduced chisq: %f' % red_chisq)
            if red_chisq > 100:
                print('Too high! Retry')
            else:
                np.savetxt('%s/starting_point.dat' % savedir, np.array(vals))
                ilike = pt.sum([shared(check)])
        nn += 1
        if nn == 1000:
            raise RuntimeError('Valid starting point not found after 100 attempts. Execution stopped')
    
    with model:
        map_prof = [pm.Deterministic('bright', maps[0])]
        p_prof = pm.Deterministic('press', pprof[0])
        like = pm.Potential('like', pt.sum(like))
        if slope_prior:
            pm.Deterministic('slope', slopes[0])
        with open('%s/model.pickle' % savedir, 'wb') as m:
            cloudpickle.dump(model, m, -1)
        start_guess = [np.atleast_2d(m.eval()) for m in map_prof]
    pplots.plot_guess(start_guess, sz, knots=None if type(press) == pfuncs.Press_gNFW else 
                      [[r.to(sz.flux_data[0][0].unit, equivalencies=eq_kpc_as)[0].value for i, r in enumerate(press.knots[0])]], 
                      plotdir=plotdir)
    
    with model:
        step = assign_step_methods(model, None, methods=pm.STEP_METHODS, step_kwargs={})
        trace = pm.sample(draws=1000, tune=1000, chains=8, return_inferencedata=True, step=step,
                          initvals=model.initial_point())
    
    trace.to_netcdf("%s/trace.nc" % savedir)
    # trace = az.from_netcdf("%s/trace.nc" % savedir)

    # Extract chain of parameters ((nwalkers x niter) x nparams)
    prs = [str(_) for _ in model.basic_RVs]
    samples = np.zeros((trace['posterior'][prs[-1]].size, len(prs)))
    for (i, par) in enumerate(prs):
        samples[:,i] = np.array(trace['posterior'][par]).flatten()
    # Extract surface brightness profiles
    flat_surbr = np.array([trace.posterior['bright']]).reshape(1, samples.shape[0], -1)
    # Median surface brightness profile + CI
    perc_sz = np.array([pplots.get_equal_tailed(f, ci=ci) for f in flat_surbr])

    # Posterior distribution parameters
    param_med = np.median(samples, axis=0)
    param_std = np.std(samples, axis=0)
    pfuncs.print_summary(prs, param_med, param_std, perc_sz[:,1], sz)
    pfuncs.save_summary('%s/%s' % (savedir, name), prs, param_med, param_std, ci=ci)

    # Traceplot
    pm.summary(trace, var_names=prs)
    pplots.traceplot(trace, prs, nc=1, trans_ped=lambda x: 1e4*x, plotdir=savedir)

    # Best fitting profile on SZ surface brightness
    pplots.fitwithmod(sz, perc_sz, eq_kpc_as, clus=clus, rbins=None if type(press)==pfuncs.Press_gNFW else press.knots.to(u.arcsec, equivalencies=eq_kpc_as).value,
                      peds=[trace.posterior['ped'].data.mean()], fact=1e5, ci=ci, plotdir=plotdir)

    # Forest plots
    for p in prs[:-1]:
        axes = az.plot_forest(trace, var_names=p)
        fig = axes.ravel()[0].figure
        fig.savefig('%s/forest_%s.pdf' % (plotdir, p))
    axes = az.plot_forest(trace, var_names=['ped'],
                          transform=lambda x: 1e5*x)
    fig = axes.ravel()[0].figure
    fig.savefig('%s/forest_ped.pdf' % plotdir)

    # Cornerplots
    pplots.triangle(samples, ['log(%s)' % _ for _ in prs], 
                    show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/logPs_')

    # Radial pressure profile
    p_prof = [trace.posterior['press'].data.reshape(samples.shape[0], -1)]
    p_quant = [pplots.get_equal_tailed(pp, ci=ci) for pp in p_prof]
    [np.savetxt('%s/press_prof_%s.dat' % (savedir, c), pq) for c, pq in zip(clus, p_quant)]
    univpress=None
    pplots.spaghetti_press(sz.r_pp, p_prof, clus=clus, nl=100, ci=ci, univpress=univpress, plotdir=plotdir, 
                           rbins=None if type(press)==pfuncs.Press_gNFW else press.knots)
    pplots.plot_press(sz.r_pp, p_quant, clus=clus, ci=ci, univpress=univpress, plotdir=plotdir, 
                      rbins=None if type(press)==pfuncs.Press_gNFW else press.knots)

    # Outer slope posterior distribution
    slopes = np.array([trace.posterior['slope'].data.flatten()]).flatten()
    pplots.hist_slopes(slopes.flatten(), ci=ci, plotdir=plotdir)

if __name__ == '__main__':
    main()
