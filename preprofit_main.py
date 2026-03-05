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
import yaml
import os

### Global and local variables

# Read configuration file
with open("./examples/spt_set.yaml", "r") as f:
    conf = yaml.safe_load(f)

## Cluster cosmology
H0 = 70 # Hubble constant at z=0
Om0 = 0.3 # Omega matter
cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)	

# Cluster list
clus = conf['clus'] # just 3 clusters as an example, you need much more to determine the population parameters
nc = len(clus)
z = np.atleast_1d(conf['z']) # redshift list
# Overdensity measures (set them for defining the starting point for the MCMC)
r500 = np.atleast_1d(conf['r500'])*u.kpc
M500 = (4/3*np.pi*cosmology.critical_density(z).to(u.g/u.kpc**3)*500*r500**3).to(u.Msun)

## Beam and transfer function
# Beam file already includes transfer function?
beam_and_tf = conf["beam_and_tf"]
# Beam and transfer function. From input data or Gaussian approximation?
beam_approx = conf["beam_approx"]
tf_approx = conf["tf_approx"]
fwhm_beam = np.atleast_1d(conf["fwhm_beam"])*u.arcsec if conf["fwhm_beam"] is not None else conf["fwhm_beam"] # fwhm of the normal distribution, if adopted
step_data = np.atleast_1d(conf["step_data"])*u.arcsec if conf["step_data"] is not None else conf["step_data"] # radial step of beam data provided, if required
loc, scale, k = None, None, None # location, scale and normalization parameters of the normal cdf for the transfer function approximation, if adopted

# Transfer function provenance (not the instrument, but the team who derived it)
tf_source_team = conf["tf_source_team"] # choose among 'NIKA', 'MUSTANG' or 'SPT'

 ## File names (FITS and ASCII formats are accepted)
# NOTE: if some of the files are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
# NOTE: if you have beam + transfer function in the same file, assign the name of the file to beam_filename and ignore tf_filename
files_dir = conf["files_dir"] # files directory
beam_filename = files_dir+conf["beam_filename"] if conf["beam_filename"] is not None else conf["beam_filename"] # beam
tf_filename = files_dir+conf["tf_filename"] if conf["tf_filename"] is not None else conf["tf_filename"] # transfer function
flux_filename = ['%s%s%s.dat' % (files_dir, conf["flux_filename"], c) for c in clus] if tf_source_team=='SPT' else [files_dir+conf["flux_filename"]] # observed data
convert_filename = files_dir+conf["convert_filename"] if conf["convert_filename"] is not None else conf["convert_filename"] # conversion Compton -> observed data

# Temperature used for the conversion factor above
t_const = 8*u.keV # if conversion is not required, preprofit ignores it

# Units (here users have to specify units of measurements for the input data, either a list of units for multiple columns or a single unit for a single 
# measure in the file)
# NOTE: if some of the units are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
# NOTE: base unit is u.Unit(''), e.g. used for Compton y measurements
beam_units = [u.Unit(_) for _ in conf["beam_units"]] if conf["beam_units"] is not None else conf["beam_units"] # beam units
flux_units = [u.Unit(_) for _ in conf["flux_units"]] # observed data units
tf_units = [u.Unit(_) for _ in conf["tf_units"]] if conf["tf_units"] is not None else conf["tf_units"] # transfer function units
conv_units = [u.Unit(_) for _ in conf["conv_units"]] if conf["conv_units"] is not None else conf["conv_units"] # conversion units

# Adopt a cropped version of the beam / beam + transfer function image? Be careful while using this option
crop_image = False # adopt or do not adopt?
cropped_side = 200 # side of the cropped image (automatically set to odd value)

# Maximum radius for line-of-sight Abel integration
R_b = 5000*u.kpc
# Maximum radius for radial profile computation
maxr_data = conf['maxr_data']*u.arcsec

# Name for outputs
name = 'preprofit'
plotdir = conf['plotdir'] # directory for the plots
savedir = conf['savedir'] # directory for saved files

## Prior constraint on the pressure slope at large radii?
slope_prior = True # apply or do not apply?
r_out = (r500.to(u.kpc).value)*1.4 # large radius for the slope prior
max_slopeout = 0. # maximum value for the slope at r_out

## Pressure modelization (4 options available)
press_mod = conf['press_mod']
knots = np.outer([.1, .4, .7, 1, 1.3], r500.to(u.kpc).value).T # set knots position
# 1. Restricted cubic spline model
if press_mod == 'rcs':
    press = pfuncs.Press_rcs(z=z, cosmology=cosmology, knots=knots, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)
# 2. Generalized Navarro Frenk and White model
if press_mod == 'gnfw':
    press = pfuncs.Press_gNFW(z=z, cosmology=cosmology, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)
    pars_gnfw = conf['pars_gnfw'] # leave in this list only the parameters you want to fit among ['P_0', 'a', 'b', 'r_p']
# 3. Non parametric power-law model
if press_mod == 'plaw':
    press = pfuncs.Press_nonparam_plaw(z=z, cosmology=cosmology, knots=knots, slope_prior=slope_prior, max_slopeout=max_slopeout)
# 4. Cubic spline model
if press_mod == 'cubspl':
    press = pfuncs.Press_cubspline(z=z, cosmology=cosmology, knots=knots, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

## Get starting parameters of the population-averaged profile assuming an universal pressure profile
logunivpars = np.mean(press.get_universal_params(M500=M500), axis=0)
if type(press)==pfuncs.Press_gNFW: 
    c500=1.177
    logunivpars[-1] = np.log10(10**(logunivpars[-1])/c500)
nk = len(logunivpars) # number of parameters for the population-averaged pressure profile

# If hierarchical model, do you want to include dependencies on redshift or mass?
z_dep = False # redshift-dependent parameter?
M_dep = False # mass-dependent parameter?

# Sampling step
mystep = conf['mystep']*u.arcsec # constant step (values larger than (1/7)*FWHM of the beam are not recommended)

# MCMC options
niter = 4000 # number of iterations
nburn = 4000 # number of burn-in iterations
nwalk = 8 # number of random walkers

# End of configuration part. You should not touch the code in the next part

# -------------------------------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------------------------------

def main():

    # Flux density data
    flux_data = [pfuncs.read_data(fl, ncol=3, units=flux_units) for fl in flux_filename] # radius, flux density, statistical error

    # Transfer function
    wn_as, tf = [None, None] if beam_and_tf else pfuncs.read_tf(
		tf_filename, tf_units=tf_units, approx=tf_approx, loc=loc, scale=scale, k=k) # wave number, transmission

    # PSF+tf filtering
    freq, fb, filtering = pfuncs.filtering(mystep, press.eq_kpc_as, maxr_data=maxr_data, approx=beam_approx, filename=beam_filename, 
										   beam_and_tf=beam_and_tf, crop_image=crop_image, cropped_side=cropped_side, fwhm_beam=fwhm_beam, 
										   step_data=step_data, w_tf_1d=wn_as, tf_1d=tf, plotdir=plotdir)
    
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
    sz = pfuncs.SZ_data(clus=clus, step=mystep, eq_kpc_as=press.eq_kpc_as, conv_temp_sb=conv_temp_sb, flux_data=flux_data, radius=radius, sep=sep, 
						r_pp=r_pp, r_am=r_am, filtering=filtering)

    # Compute P500
    press.P500 = [pfuncs.get_P500((sz.r_pp[j]/r500[j]).value, cosmology, z[j], M500=M500[j]).value for j in range(nc)]
    press.r500 = [r for r in r500]

    # Internal attributes for the pressure class
    pfuncs.add_attrs(press, nc, sz)
    
    # Save objects
    with open('%s/press_obj.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(press, f, -1)
    with open('%s/szdata_obj.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(sz, f, -1)
    
    ## Model definition
    with pm.Model() as model:
        # Customize the prior distribution of the parameters using pymc distributions
        if type(press)==pfuncs.Press_gNFW:
            if nc > 1:
                raise RuntimeError("Hierarchical model is not available when using a gNFW pressure model")
            nps = len(pars_gnfw) # number of fitted parameters
            ind_pars = [p in pars_gnfw for p in ['P_0', 'a', 'b', 'c', 'r_p']] # which gNFW parameters are fitted?
            [pm.StudentT('lgP_{%s,i}' % j, nu=10, mu=logunivpars[ind_pars][j], sigma=1, 
                         initval=logunivpars[ind_pars][j]) for j in range(nps)]
            inp_pars = [[model['lgP_{%s,i}' % (_-np.cumsum([ip==0 for ip in ind_pars])[_])] 
                         if ind_pars[_] else logunivpars[_] for _ in range(5)]]
        else:
            if nc > 1:
				# Population parameters
                pm.Uniform('sigma_{int,k}', 0., 1., initval=np.repeat(.2, nk), shape=nk)
                pm.Normal('lgP_k', mu=logunivpars, sigma=.5, initval=logunivpars, shape=nk)
                if z_dep:
                    pm.StudentT('z_dep', mu=np.zeros(nk), nu=np.ones(nk), shape=nk, initval=np.zeros(nk))
                if M_dep:
                    pm.StudentT('M_dep', mu=np.zeros(1), nu=np.ones(1), shape=1, initval=np.zeros(1))
			# Individual cluster parameters
            [pm.StudentT('lgP_{%s,i}' % j, nu=10, 
                         mu=model['lgP_k'][j] if nc > 1 else logunivpars
                         +model.z_dep[j]*pt.log10((1+press.z)/(1+.3)) if z_dep else 0
                         +model.M_dep*pt.log10(M500/8e14) if M_dep else 0,
                        sigma=model['sigma_{int,k}'][j]*np.sqrt(8/10) if nc > 1 else 1, 
                        initval=np.repeat(logunivpars[j], nc), shape=nc) for j in range(nk)]
            inp_pars = [[m[i] for m in [model['lgP_{%s,i}' % k] for k in range(nk)]] for i in range(nc)]
        # Add pedestal component to the model
        pm.Normal('peds', 0, 1e-6, shape=nc, initval=np.zeros(nc))

        # Likelihood function
        lprof, pprof, maps, slopes = zip(*map(
            lambda lgP_ki, ped_i, szr, szrr, sza, szl, szd, szf, i: lfuncs.whole_lik(
                model, lgP_ki, ped_i, press, szr.value, szrr.value, sza, sz.filtering.value, 
                szl, sz.sep, szd, sz.conv_temp_sb, sz.radius[sz.sep:].value, szf, i),
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
        start_guess = [model['bright_%s' % j].eval({
			str(p): model.rvs_to_initial_values[model.named_vars[str(p)]] for p in model.free_RVs[-nk-1:]}) for j in range(nc)]
        fact = 10**int(-np.sign(flux_data[0][1][0])*np.round(np.log10(np.abs(flux_data[0][1][0].value)), 0))
        pplots.plot_guess(start_guess, sz, press, fact=fact, plotdir=plotdir)

		# Fit
        print("Fitting %s cluster%s" % (nc, 's' if nc>1 else ''))
        trace = pm.sample(draws=niter, tune=nburn, chains=nwalk, cores=os.cpu_count(), initvals=model.rvs_to_initial_values)
	
	    # Save chain
        trace.to_netcdf("%s/trace.nc" % savedir)

if __name__ == '__main__':
    main()
