import preprofit_funcs as pfuncs
import preprofit_likfuncs as lfuncs
import preprofit_plots as pplots
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

# Cluster list
clus = ['SPT-CLJ0500-5116', 'SPT-CLJ0637-4829', 'SPT-CLJ2055-5456'] # just 3 as an example, you need much more to determine the population parameters
nc = len(clus)
z = [.11, .2026, .139] # redshift list
# Overdensity measures (set them for defining the starting point for the MCMC)
r500 = [943.85207035, 1290.31531693, 1022.3744362]*u.kpc
M500 = (4/3*np.pi*cosmology.critical_density(z).to(u.g/u.kpc**3)*500*r500**3).to(u.Msun)

## Beam and transfer function
# Beam file already includes transfer function?
beam_and_tf = False

# Beam and transfer function. From input data or Gaussian approximation?
beam_approx = True
tf_approx = False
fwhm_beam = [75]*u.arcsec # fwhm of the normal distribution, if adopted
loc, scale, k = None, None, None # location, scale and normalization parameters of the normal cdf for the transfer function approximation, if adopted

# Transfer function provenance (not the instrument, but the team who derived it)
tf_source_team = 'SPT' # choose among 'NIKA', 'MUSTANG' or 'SPT'

## File names (FITS and ASCII formats are accepted)
# NOTE: if some of the files are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
# NOTE: if you have beam + transfer function in the same file, assign the name of the file to beam_filename and ignore tf_filename
files_dir = './data' # files directory
beam_filename = None # beam
tf_filename = '%s/sptsz_trough_filter_1d.dat' %files_dir # transfer function
flux_filename = ['%s/press_data_%s.dat' % (files_dir, cl) for cl in clus] # observed data
convert_filename = None # conversion Compton -> observed data

# Temperature used for the conversion factor above
t_const = 8*u.keV # if conversion is not required, preprofit ignores it

# Units (here users have to specify units of measurements for the input data, either a list of units for multiple columns or a single unit for a single 
# measure in the file)
# NOTE: if some of the units are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
# NOTE: base unit is u.Unit(''), e.g. used for Compton y measurements
beam_units = u.Unit('') # beam units
flux_units = [u.arcsec, u.Unit(''), u.Unit('')] # observed data units
tf_units = [1/u.radian, u.Unit('')] # transfer function units
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

## Prior constraint on the pressure slope at large radii?
slope_prior = True # apply or do not apply?
r_out = (r500.to(u.kpc).value)*1.4 # large radius for the slope prior
max_slopeout = 0. # maximum value for the slope at r_out

## Pressure modelization
knots = np.outer([.1, .4, .7, 1, 1.3], r500.to(u.kpc).value).T
# Restricted cubic spline model
press = pfuncs.Press_rcs(z=z, cosmology=cosmology, knots=knots, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)
# Generalized Navarro Frenk and White model
# press = pfuncs.Press_gNFW(z=z, cosmology=cosmology, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)
# Non parametric power-law model
# press = pfuncs.Press_nonparam_plaw(z=z, cosmology=cosmology, knots=knots, slope_prior=slope_prior, max_slopeout=max_slopeout)
# Cubic spline model
# press = pfuncs.Press_cubspline(z=z, cosmology=cosmology, knots=knots, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

## Get starting parameters of the population-averaged profile assuming an universal pressure profile
logunivpars = np.mean(press.get_universal_params(M500=M500), axis=0)
# if type(press)==pfuncs.Press_gNFW: 
#     c500=1.177
#     logunivpars[-1] = np.log10(10**(logunivpars[-1])/c500)
nk = len(logunivpars) # number of parameters for the population-averaged pressure profile

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
    wn_as, tf = [None, None] if beam_and_tf else pfuncs.read_tf(tf_filename, tf_units=tf_units, approx=tf_approx, loc=loc, scale=scale, k=k) # wave number, transmission

    # PSF+tf filtering
    freq, fb, filtering = pfuncs.filtering(mystep, press.eq_kpc_as, maxr_data=maxr_data, approx=beam_approx, filename=beam_filename, beam_and_tf=beam_and_tf, crop_image=crop_image, cropped_side=cropped_side, fwhm_beam=fwhm_beam, step_data=15*u.arcsec, w_tf_1d=wn_as, tf_1d=tf)
    
    # Radius definition
    radius = np.arange(filtering.shape[0]//2+1)*mystep
    radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
    sep = radius.size//2 # index of radius 0
    # radius in kpc used to compute the pressure profile (radius 0 excluded)
    r_pp = [np.arange(1, R_b/mystep.to(u.kpc, equivalencies=press.eq_kpc_as)[i]+1)*mystep.to(u.kpc, equivalencies=press.eq_kpc_as)[i] for i in range(nc)]
    r_am = np.arange(1+min([len(r) for r in r_pp]))*mystep.to(u.arcmin, equivalencies=press.eq_kpc_as) # radius in arcmin (radius 0 included)

    # If required, temperature-dependent conversion factor from Compton to surface brightness data unit
    if not flux_units[1] == '':
        temp_data, conv_data = pfuncs.read_data(convert_filename, 2, conv_units)
        conv_fun = interp1d(temp_data, conv_data, 'linear', fill_value='extrapolate')
        conv_temp_sb = conv_fun(t_const)*conv_units[1]
    else:
        conv_temp_sb = 1*u.Unit('')

    # Set of SZ data required for the analysis
    sz = pfuncs.SZ_data(clus=clus, step=mystep, eq_kpc_as=press.eq_kpc_as, conv_temp_sb=conv_temp_sb, flux_data=flux_data, radius=radius, sep=sep, r_pp=r_pp, r_am=r_am, filtering=filtering)

    # Compute P500
    press.P500 = [pfuncs.get_P500((sz.r_pp[j]/r500[j]).value, cosmology, z[j], M500=M500[j]).value for j in range(nc)]
    press.r500 = [r for r in r500]

    # Other indexes
    pfuncs.add_indices(press, nc, sz)
    
    # Save objects
    with open('%s/press_obj.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(press, f, -1)
    with open('%s/szdata_obj.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(sz, f, -1)
    
    ## Model definition
    evol = False
    P_dep = False
    with pm.Model() as model:
        if type(press)==pfuncs.Press_gNFW:
            pars_gnfw = ['P_0', 'a', 'b', 'c', 'r_p']
            nps = len(pars_gnfw)
            ind_pars = [p in pars_gnfw for p in ['P_0', 'a', 'b', 'c', 'r_p']]
            lgu = logunivpars[ind_pars]
            pm.Uniform('sigma_gnfw', 0, 10, initval=np.repeat(1, nps), shape=nps)
            pm.Normal('lgPgnfw', mu=lgu, sigma=[1,1,1,1.5,1], initval=lgu, shape=nps)
            [pm.StudentT('lgP_{%s,i}' % j, nu=10, mu=model['lgPgnfw'][j],
                        sigma=model['sigma_gnfw'][j]/np.sqrt(10/8), 
                        initval=np.repeat(lgu[j], nc), shape=nc) for j in range(nps)]
            inp_pars = [[model['lgP_{%s,i}' % (_-np.cumsum([ip==0 for ip in ind_pars])[_])][i] 
                         if ind_pars[_] else logunivpars[_] for _ in range(5)] for i in range(nc)]
        else:
            # Customize the prior distribution of the parameters using pymc distributions
            if nc > 1:
                pm.Uniform('sigma_{int,k}', 0., 1., initval=np.repeat(.2, nk), shape=nk)
                pm.Normal('lgP_k', mu=logunivpars, sigma=.5, initval=logunivpars, shape=nk)
            if evol:
                pm.StudentT('evol', mu=np.zeros(nk), nu=np.ones(nk), shape=nk, initval=np.zeros(nk))
            if P_dep:
                pm.StudentT('P_dep', mu=np.zeros(1), nu=np.ones(1), shape=1, initval=np.zeros(1))
            [pm.StudentT('lgP_{%s,i}' % j, nu=10, 
                         mu=model['lgP_k'][j] if nc > 1 else logunivpars
                         +model.evol[j]*pt.log10((1+press.z)/(1+.3)) if evol else 0
                         +model.P_dep*pt.log10(M500/8e14) if P_dep else 0,
                        sigma=model['sigma_{int,k}'][j]/np.sqrt(10/8) if nc > 1 else 1, 
                        initval=np.repeat(logunivpars[j], nc), shape=nc) for j in range(nk)]
            inp_pars = [[m[i] for m in [model['lgP_{%s,i}' % k] for k in range(nk)]] for i in range(nc)]
        # Add pedestal component to the model
        pm.Normal('peds', 0, 1e-6, shape=nc, initval=np.zeros(nc))

        # Likelihood function
        lprof, pprof, maps, slopes = zip(*map(
            lambda lgP_ki, ped_i, szr, szrr, sza, szl, szd, szf, i: lfuncs.whole_lik(
                model, lgP_ki, ped_i, press, szr.value, szrr.value, sza, sz.filtering.value, 
                szl, sz.sep, szd, sz.radius[sz.sep:].value, szf, i),
            inp_pars, [model['peds'][i] for i in range(nc)], sz.r_pp, sz.r_red, sz.abel_data, sz.dist.labels, sz.dist.d_mat, sz.flux_data, np.arange(nc)))
        [pm.Normal('like_%s' % i, mu=lprof[i], sigma=sz.flux_data[i][2], observed=sz.flux_data[i][1], shape=len(sz.flux_data[i][1])) for i in range(nc)]

        # Save useful measures
        [pm.Deterministic('press_%s' % i, p) for i, p in enumerate(pprof)]
        [pm.Deterministic('bright_%s' % i, m) for i, m in enumerate(maps)]
        if press.slope_prior:
            [pm.Deterministic('slope_%s' % i, s) for i, s in enumerate(slopes)]

        # Save model
        with open('%s/model.pickle' % savedir, 'wb') as f:
            cloudpickle.dump(model, f, -1)

        ## Sampling
        start_guess = [model['bright_%s' % j].eval({str(p): model.rvs_to_initial_values[model.named_vars[str(p)]] for p in model.free_RVs[-6:]}) for j in range(nc)]
        pplots.plot_guess(start_guess, sz, press, fact=1e4, plotdir=plotdir)
        
	# Fit
    trace = pm.sample(draws=4000, tune=4000, chains=4, initvals=model.rvs_to_initial_values)

    # Save chain
    trace.to_netcdf("%s/trace_t2u.nc" % savedir)


    ### Plots
    
    prs = [k for k in trace.posterior.keys()]
    prs = prs[:np.where([p[:2] in ['br', 'pr'] for p in prs])[0][0]]
    prs = np.roll(prs, 1 if any(p.startswith('sigma') for p in prs) else 0)
    samples = []
    for (i, par) in enumerate(prs):
        res = trace.posterior[par].data.reshape(np.prod(trace.posterior[prs[0]].shape[:2]), -1)
        for j in range(res.shape[1]):
            samples.append(res[:,j])
    samples = np.array(samples).T
    
    prs_ext_kn = [
            [p.replace('k', str(k)) for k in range(nk)] if nc > 1 else '' for p in (prs[:2])]+[
                ['evol_%s' % _ for _ in range(trace.posterior.evol.shape[-1])] if 'evol' in prs else '']+[
                    ['P_dep_%s' % _ for _ in range(trace.posterior.P_dep.shape[-1])] if 'P_dep' in prs else '']+[
            [p.replace('i', str(i)) for i in range(nc)] for p in
            (prs[np.where(np.array(prs)=='lgP_{0,i}')[0][0]:-1])]+[[
                prs[-1]+'_{%s}' % i for i in range(nc)]]
    prs_ext_clus = [
            [p.replace('k', str(k)) for k in range(nk)] if nc > 1 else '' for p in (prs[:2])]+[
                ['evol_%s' % _ for _ in range(trace.posterior.evol.shape[-1])] if 'evol' in prs else '']+[
                    ['P_dep_%s' % _ for _ in range(trace.posterior.P_dep.shape[-1])] if 'P_dep' in prs else '']+[
    	[p.replace('i', str(i)) for p in (prs[np.where(np.array(prs)=='lgP_{0,i}')[0][0]:-1]
                                       )] for i in range(nc)]+[[
                prs[-1]+'_{%s}' % i for i in range(nc)]]
    prs_ext_kn = [p for p in prs_ext_kn if p!='']
    prs_ext_clus = [p for p in prs_ext_clus if p!='']
    
    # Extract surface brightness profiles
    flat_surbr = np.array([trace.posterior['bright_%s' % i] for i in range(nc)]).reshape(nc, samples.shape[0], -1)
    # Median surface brightness profile + CI
    perc_sz = np.array([pplots.get_equal_tailed(f, ci=ci) for f in flat_surbr])
    
    # Posterior distributions summary
    pm.summary(trace, var_names=prs)
    
    # Traceplot
    pplots.traceplot(trace, prs, prs_ext_kn, compact=0, fact_ped=1e5, ppp=nk, fsize=14, plotdir=savedir)
    
    # Best fitting profile on SZ surface brightness
    pplots.fitwithmod(sz, perc_sz, press.eq_kpc_as, rbins=None if type(press)==pfuncs.Press_gNFW else np.array(
        [press.knots[_]/press.kpc_as[_]*u.arcsec for _ in range(nc)]), peds=np.mean(trace.posterior['peds'].data, axis=(0,1)), ind_fits=None, fact=1e5, ci=ci, plotdir=plotdir)
    
    # Cornerplots
    ind_clus = [[k+np.cumsum([len(p) for p in [[]]+prs_ext_clus[:-1]])[j] for k in range(len(p))] for j,p in enumerate(prs_ext_clus)]
    pplots.triangle([samples[:,i] for i in ind_clus[:-1]], prs_ext_clus[:-1], 
                    model, fact_ped=1e5
                    , plot_prior=0, show_lines=True, 
                    show_title=False, col_lines='b', ci=ci, plotdir=plotdir)

if __name__ == '__main__':
    main()
