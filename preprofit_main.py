import ppf_mult_funcs as pfuncs
import ppf_mult_plots as pplots
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const
from scipy.interpolate import interp1d
from scipy.fftpack import fft2
import h5py
import cloudpickle
import pymc3 as pm
import pymc3_ext as pmx
from theano import shared
import arviz as az

### Global and local variables

## Cluster cosmology
H0 = 70 # Hubble constant at z=0
Om0 = 0.3 # Omega matter
names, reds, M500 = np.loadtxt('data/fullsample_SPT.txt', skiprows=1, dtype=('str', 'str'), usecols=(0,3,4), unpack=1)[:,:40]
clus = []
for i, n in enumerate(names):
    try:
        pfuncs.read_data('data/press_data_'+n+'.dat', ncol=3, units=[u.arcsec, u.Unit(''), u.Unit('')])
        clus.append(n)
    except:
        pass
nc = len(clus)
ind = [np.where(clus[i]==names)[0][0] for i in range(nc)]
z = np.float64(reds)[ind] # redshift
M500 = np.float64(M500)[ind]*u.Msun # M500

cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)	
kpc_as = cosmology.kpc_proper_per_arcmin(z).to('kpc arcsec-1') # number of kpc per arcsec
eq_kpc_as = [(u.arcsec, u.kpc, lambda x: x*kpc_as.value, lambda x: x/kpc_as.value)] # equation for switching between kpc and arcsec


# def r500fromM500(M500, z,omegam,omegal,omegak):
#     "Compute r500 in Mpc given M500 in msol"
#     Ez=np.sqrt(omegam*(1+z)**3+omegak*(1+z)**2+omegal)
#     r500 = (3/4* M500/(500. * 2.775e11*0.7*0.7 *Ez**2 * np.pi))**(1/3)
#     return r500
r500 = ((3/4*M500/(500.*cosmology.critical_density(z)*np.pi))**(1/3)).to(u.kpc)
c500 = 1.177
pnorm = .59/1.14*.175*3/8/np.pi*(const.G.value**(-1/3)*u.kg/u.m/u.s**2).to(u.keV/u.cm**3)/((u.kg/250**2/cosmology.H0**4/u.s**4/3e14/u.Msun).to(''))**(2/3)
hz = cosmology.H(z)/cosmology.H0
P500 = pnorm*hz**(8/3)*(M500/3e14/u.Msun)**(2/3)

## Beam and transfer function
# Beam file already includes transfer function?
beam_and_tf = True

# Beam and transfer function. From input data or Gaussian approximation?
beam_approx = False
tf_approx = False
fwhm_beam = None # fwhm of the normal distribution for the beam approximation
loc, scale, k = None, None, None # location, scale and normalization parameters of the normal cdf for the transfer function approximation

# Transfer function provenance (not the instrument, but the team who derived it)
tf_source_team = 'SPT' # alternatively, 'MUSTANG' or 'SPT'

## File names (FITS and ASCII formats are accepted)
# NOTE: if some of the files are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
# NOTE: if you have beam + transfer function in the same file, assign the name of the file to beam_filename and ignore tf_filename
files_dir = './data' # files directory
beam_filename = '%s/min_variance_flat_sky_xfer_1p25_arcmin.fits' %files_dir # beam
tf_filename = None # transfer function
flux_filename = ['%s/press_data_' %files_dir +cl+'.dat' for cl in clus]# observed data
convert_filename = None # conversion Compton -> observed data

# Temperature used for the conversion factor above
t_const = 12*u.keV # if conversion is not required, preprofit ignores it

# Units (here users have to specify units of measurements for the input data, either a list of units for multiple columns or a single unit for a single 
# measure in the file)
# NOTE: if some of the units are not required, either assign a None value or just let them like this, preprofit will automatically ignore them
# NOTE: base unit is u.Unit(''), e.g. used for Compton y measurements
beam_units = u.beam # beam units
flux_units = [u.arcsec, u.Unit(''), u.Unit('')] # observed data units
# tf_units = [1/u.arcsec, u.Unit('')] # transfer function units
# conv_units = [u.keV, u.Jy/u.beam] # conversion units

# Adopt a cropped version of the beam / beam + transfer function image? Be careful while using this option
crop_image = True # adopt or do not adopt?
cropped_side = 200 # side of the cropped image (automatically set to odd value)

# Maximum radius for line-of-sight Abel integration
R_b = 5000*u.kpc

# Name for outputs
name = 'preprofit'
plotdir = './multi/nonparamlaw/' # directory for the plots
savedir = '%s/save/' % plotdir # directory for saved files

## Prior on the Integrated Compton parameter?
calc_integ = False # apply or do not apply?
integ_mu = .94/1e3 # from Planck

integ_sig = .36/1e3 # from Planck
## Prior on the pressure slope at large radii?
slope_prior = 0#True # apply or do not apply?
r_out = 1e3*u.kpc # large radius for the slope prior
max_slopeout = -2. # maximum value for the slope at r_out

## Pressure modelization
# 3 models available: 1 parametric (Generalized Navarro Frenk and White), 2 non parametric (cubic spline / power law interpolation)

# Generalized Navarro Frenk and White
press = pfuncs.Press_gNFW(eq_kpc_as=eq_kpc_as, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

## Parameters setup
hz = cosmology.H(z)/cosmology.H0
h70 = cosmology.H0/(70*cosmology.H0.unit)
P500 = 1.65e-3*hz**(8/3)*(M500/(3e14*h70**-1*u.Msun))**(2/3)*h70**2*u.keV/u.cm**3
univpars = [[(8.403*h70**(-3/2)*P500).value[i], 1.051, 5.4905, 0.3081, 
             (r500.to(u.kpc, equivalencies=eq_kpc_as).value/c500)[i]] for i in range(nc)]
with pm.Model() as model:
    # Customize the prior distribution of the parameters using Pymc3 distributions, optionally setting the starting value as testval
    # To exclude a parameter from the fit, just set it at a fixed value 
    shape = 1
    # testval = np.array([[0., .5], [.15, .5], [2.81, 4], [6.29, 3.5], [380, 700]])
    testval = np.array([[0., -4.74017024e-05], [5.26077081e-01, .5], [4.25946548e+00, 4], [5.48547959e+00, 5.5],  [8.47405222e+02, 800]])
    # P_0 = pm.Uniform("P_0", lower=0, upper=1, testval=testval[1,:shape], shape=shape)
    Ps = pm.Normal("Ps", mu=np.log(1.5), sigma=np.log(3./1.5), shape=nc, testval=np.log([2.60908608, 2.03602627, 2.84339852, 1.71772373, 0.72905945, 0.68454561, 1.48586932, 0.65181141, 2.34198767, 1.574598])) #np.array([u[0] for u in univpars])/10))#np.array([1.01714532, 1.82576668, 1.20201582, 0.5638314, 0.2233534, 0.16041357, 0.83527021, 0.36059494, 1.00501252, 0.44485807])/25))
    a = 1.051#pm.Normal('a', mu=np.log(1.5), sigma=np.log(2/1.5), testval=np.log(1.5))#pm.Uniform('a', lower=0.5, upper=50., testval=testval[2,:shape], shape=shape)
    b = 5.4905#pm.Normal('b', mu=np.log(80), sigma=np.log(150/80), testval=np.log(80))#pm.Uniform('b', lower=3, upper=70, testval=testval[3,:shape], shape=shape)
    c = 0.3081#pm.Normal('c', mu=np.log(2.5), sigma=np.log(3/2.5), testval=np.log(2.5))
    # r_p = pm.Uniform('r_p', lower=100., upper=1000., testval=testval[4,:shape], shape=shape)
    r_p = r500.value/c500
    for i in range(nc):
        # pm.Uniform("P_"+str(i+1), lower=0, upper=1, testval=.5, shape=shape)
        pm.Uniform("ped"+str(i+1), lower=-1, upper=1, testval=0., shape=shape)

pars = [[Ps[i],#model["P_"+str(i+1)],
         a,
         b,
         c,
         r_p[i],
         model["ped"+str(i+1)]] for i in range(nc)]

'''
# Cubic spline
knots = [100, 300, 600, 1000, 2000]*u.kpc
press_knots = [9e-1, 8e-1, 2e-1, 5e-2, 5e-3]*u.Unit('keV/cm3')
press = pfuncs.Press_cubspline(knots=knots, pr_knots=press_knots, eq_kpc_as=eq_kpc_as, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

# Power law interpolation
rbins = [100, 300, 600, 1000, 2000]*u.kpc
pbins = [1e-1, 2e-2, 5e-3, 1e-3, 1e-4]*u.Unit('keV/cm3')
press = pfuncs.Press_nonparam_plaw(rbins=rbins, pbins=pbins, eq_kpc_as=eq_kpc_as, slope_prior=slope_prior, max_slopeout=max_slopeout)
press_knots = np.array([8.72610158e-01, 9.61980539e-01, 3.72659125e-01, 1.98599309e-03, 3.02911941e-04])#pbins.value

# ## Parameters setup
with pm.Model() as model:
    # Customize the prior distribution of the parameters using Pymc3 distributions, optionally setting the starting value as testval
    # To exclude a parameter from the fit, just set it at a fixed value 
    shape = 1
    # testval = np.array([[0., .5], [.15, .5], [2.81, 4], [6.29, 3.5], [380, 700]])
    testval = np.tile(press_knots, (shape,1)) if shape > 1 else press_knots
    for i in range(len(press_knots)):
        # pm.Uniform("P_"+str(i), lower=0., upper=1., testval=testval[i], shape=shape) 
        pm.Normal("P"+str(i)+"s", mu=np.log(press_knots[i]), sigma=np.log(5), shape=nc)#, testval=np.log([2.60908608, 2.03602627, 2.84339852, 1.71772373, 0.72905945, 0.68454561, 1.48586932, 0.65181141, 2.34198767, 1.574598]))
    for i in range(nc):
        pm.Uniform("ped"+str(i+1), lower=-1, upper=1, testval=0., shape=shape)
pars = [[model['P'+str(i)+'s'][j] for i in range(len(press_knots))] for j in range(nc)]
[p.append(model['ped'+str(i+1)]) for i, p in enumerate(pars)]
'''
# To start the MCMC not too far from the peak of the posterior, you can guess a value of r500 and
# JoXSZ will automatically apply the parameters of the universal pressure profile defined in Arnaud et al. 2010
# NOTE: this option is available for both parametric and non parametric pressure models
# press.set_universal_params(r500=600*u.kpc, cosmo=cosmology, z=z)

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
mystep = 15.*u.arcsec # constant step (values higher than (1/7)*FWHM of the beam are not recommended)
# NOTE: when tf_source_team = 'SPT', be careful to adopt the same sampling step used for the transfer function

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
    # press.fit_pars =  [x for x in press.pars if not press.pars[x].frozen]
    # press.max_val = [press.pars[name].maxval for name in press.fit_pars]
    # press.min_val = [press.pars[name].minval for name in press.fit_pars]
    # ndim = len(press.fit_pars)
    # press.indexes = {'ind_'+x: np.array(press.fit_pars)==x if x in press.fit_pars else press.pars[x].val for x in name_pars}

    # Flux density data
    flux_data = [pfuncs.read_data(fl, ncol=3, units=flux_units) for fl in flux_filename] # radius, flux density, statistical error
    maxr_data = [flux_data[i][0][-1].value for i in range(len(flux_data))]*flux_data[0][0][-1].unit # largest radius in the data
    maxr_data = maxr_data.mean()
    # press.pars['pedestal'].unit = flux_data[1].unit # automatically update pedestal parameter unit

    # PSF computation and creation of the 2D image
    beam_2d, fwhm = pfuncs.mybeam(mystep, maxr_data, eq_kpc_as=eq_kpc_as, approx=beam_approx, filename=beam_filename, units=beam_units, crop_image=crop_image, cropped_side=cropped_side, 
                                  normalize=True, fwhm_beam=fwhm_beam)

    # The following depends on whether the beam image already includes the transfer function
    if beam_and_tf:
        filtering = np.abs(fft2(beam_2d))
    else:
        # Transfer function
        wn_as, tf = pfuncs.read_tf(tf_filename, tf_units=tf_units, approx=tf_approx, loc=loc, scale=scale, k=k) # wave number, transmission
        filt_tf = pfuncs.filt_image(wn_as, tf, tf_source_team, beam_2d.shape[0], mystep, eq_kpc_as) # transfer function matrix
        filtering = np.abs(fft2(beam_2d))*filt_tf # filtering matrix including both PSF and transfer function

    # Radius definition
    mymaxr = [beam_2d.shape[0]//2*mystep if crop_image else (maxr_data+3*fwhm.to(maxr_data.unit, equivalencies=eq_kpc_as))//
              mystep.to(maxr_data.unit, equivalencies=eq_kpc_as)*mystep.to(maxr_data.unit, equivalencies=eq_kpc_as)][0] # max radius needed
    radius = np.arange(0., (mymaxr+mystep.to(mymaxr.unit, equivalencies=eq_kpc_as)).value, 
                       mystep.to(mymaxr.unit, equivalencies=eq_kpc_as).value)*mymaxr.unit # array of radii
    radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
    sep = radius.size//2 # index of radius 0
    r_pp = [np.arange(mystep.to(u.kpc, equivalencies=eq_kpc_as)[i].value, (R_b.to(u.kpc, equivalencies=eq_kpc_as)+mystep.to(u.kpc, equivalencies=eq_kpc_as)[i]).value,
                     mystep.to(u.kpc, equivalencies=eq_kpc_as)[i].value)*u.kpc for i in range(nc)] # radius in kpc used to compute the pressure profile (radius 0 excluded)
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
    sz = pfuncs.SZ_data(step=mystep, eq_kpc_as=eq_kpc_as, conv_temp_sb=conv_temp_sb, flux_data=flux_data, radius=radius, sep=sep, r_pp=r_pp, r_am=r_am, 
                        filtering=filtering, calc_integ=calc_integ, integ_mu=integ_mu, integ_sig=integ_sig)

    if type(press) == pfuncs.Press_nonparam_plaw:
        press.ind_low = [np.maximum(0, np.digitize(r, press.rbins)-1) for r in sz.r_pp]# lower bins indexes
        press.r_low = [press.rbins[i] for i in press.ind_low] # lower radial bins
        press.alpha_ind = [np.minimum(i, len(press.rbins)-2) for i in press.ind_low]# alpha indexes

    
    # Modeled profile resulting from starting parameters VS observed data (useful to adjust parameters if they are way off the target
    # with model:
    #     ll = (pfuncs.mylog_lik(shared(r_pp), P_0, a, b, c, r_p, ped, press, sz))
    #     try:
    #         np.isfinite(pmx.eval_in_model(ll))
    #         sb = pmx.eval_in_model(pfuncs.mylog_lik(shared(r_pp), P_0, a, b, c, r_p, ped, press, sz, output='bright'))
    #     except:
    #         raise Warning('The starting parameters are not in accordance with the prior distributions. Better change them!')
    
    # Save objects
    with open('%s/press_obj_mult.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(press, f, -1)
    with open('%s/szdata_obj_mult.pickle' % savedir, 'wb') as f:
        cloudpickle.dump(sz, f, -1)
    #print(model.test_point); import sys; sys.exit()
    with model:
        # print(len(pmx.eval_in_model(pfuncs.log_lik(P_0, a, b, c, r_p, model, press, sz, 'bright'))))
        with open('%s/model.pickle' % savedir, 'wb') as m:
            cloudpickle.dump(model, m, -1)
        pp = [pm.Deterministic('press'+str(i), pfuncs.log_lik_press(pr, press, model, sz, i)) for i, pr in enumerate(pars)]
        map_prof = [pm.Deterministic('bright'+str(i), pfuncs.log_lik_prof(pr, p, shape, sz, i)) for i, (pr, p) in enumerate(zip(pars, pp))]
        # map_prof = pm.Potential('br', pfuncs.log_lik_prof(pars, pp, shape, sz))
        # map_prof2 = [pm.Deterministic('br'+str(i), pfuncs.log_lik_prof2(m, sz, i)) for i, m in enumerate(map_prof)]
        like = pm.Potential('like', pfuncs.log_lik_final(map_prof, sz))
        ll = pm.Deterministic('loglik', model.logpt)
        start_guess = pmx.eval_in_model(map_prof)
        # print(start_guess[:3])
    pplots.plot_guess(start_guess, sz, plotdir=plotdir)
    # import sys; sys.exit()
    with model:
        start = pm.find_MAP(start=model.test_point, model = model)
        trace = pm.sample(draws=1000
                          , tune=300
                          , chains=4, return_inferencedata=True, step=pm.Slice(), start=start)

    trace.to_netcdf("%s/trace_mult.nc" % savedir)#; import sys; sys.exit()
    # trace = az.from_netcdf("%s/trace_mult.nc" % savedir)
    prs = [k for k in trace.posterior.keys()]
    prs = prs[:np.where([p[:2]=='pr' for p in prs])[0][0]]
    

    # samples = np.zeros((trace['posterior'][prs[-1]].size, len(prs)))
    # for (i, par) in enumerate(prs):
    #     samples[:,i] = np.array(trace['posterior'][par]).flatten()

    samples = np.zeros((trace['posterior'][prs[-1]].size, len(prs)+9))
    for i in range(10):
        samples[:,i] = np.array(trace['posterior'][prs[0]])[:,:,i].flatten()
    for (i, par) in enumerate(prs[1:]):
        i = i+10;samples[:,i] = np.array(trace['posterior'][par]).flatten() 
    np.savetxt('%s/trace_mult.txt' % savedir, samples)

    # Extract chain of parameters
    flat_chain = samples # ((nwalkers x niter) x nparams)
    # Extract surface brightness profiles
    flat_surbr = np.array([trace.posterior['bright'+str(i)] for i in range(nc)]).reshape(nc, samples.shape[0], -1)
    # Median surface brightness profile + CI
    perc_sz = np.array([pplots.get_equal_tailed(f, ci=ci) for f in flat_surbr])

    # Posterior distribution parameters
    param_med = np.median(flat_chain, axis=0)
    param_std = np.std(flat_chain, axis=0)
    pfuncs.print_summary(prs, param_med, param_std, perc_sz[:,1], sz)
    pfuncs.save_summary('%s/%s' % (savedir, name), prs, param_med, param_std, ci=ci)


    sl = sum([x in prs for x in ['a', 'b', 'c']])
    pm.summary(trace, var_names=prs)
    # axes = az.plot_trace(trace, var_names=prs)
    # fig = axes.ravel()[0].figure
    # fig.savefig('%s/tp.pdf' % plotdir)
    axes = az.plot_trace(trace, var_names=prs[0], transform=np.exp)
    fig = axes.ravel()[0].figure
    fig.savefig('%s/traceplot_P0.pdf' % plotdir)
    axes = az.plot_trace(trace, var_names=prs[1+sl:])
    fig = axes.ravel()[0].figure
    fig.savefig('%s/traceplot_peds.pdf' % plotdir)
    if sl != 0:
        axes = az.plot_trace(trace, var_names=prs[1:1+sl], transform=np.exp)
        fig = axes.ravel()[0].figure
        fig.savefig('%s/traceplot_slopes.pdf' % plotdir)
    # Cornerplots
    prs_new = ['P0_'+str(i+1) for i in range(nc)]
    [prs_new.append(p) for p in prs[1:]]
    flat_chain[:,:nc+sl] = np.exp(flat_chain[:,:nc+sl])  
    pplots.triangle(flat_chain[:,:nc], prs_new[:nc], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'P0s_')
    if sl != 0:
        pplots.triangle(flat_chain[:,nc:nc+sl], prs_new[nc:nc+sl], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'slopes_')
    pplots.triangle(flat_chain[:,nc+sl:], prs_new[nc+sl:], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'peds_')
    i1 = [it for s in [[x for x in np.arange(nc//2)], [x for x in np.arange(nc,nc+sl+nc//2)]] for it in s]
    i2 = [it for s in [[x for x in np.arange(nc//2)], [x for x in np.arange(nc,nc+sl)], [x for x in np.arange(nc+sl+nc//2,2*nc+sl)]] for it in s]
    i3 = [x for x in np.arange(nc//2,nc+sl+nc//2)]
    i4 = [it for s in [[x for x in np.arange(nc//2,nc+sl)], [x for x in np.arange(nc+sl+nc//2,2*nc+sl)]] for it in s]
    pplots.triangle(flat_chain[:,i1].reshape(flat_chain.shape[0], nc+sl), [prs_new[x] for x in i1], 
                    show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'mix1_')
    pplots.triangle(flat_chain[:,i2].reshape(flat_chain.shape[0], nc+sl), [prs_new[x] for x in i2], 
                    show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'mix2_')
    pplots.triangle(flat_chain[:,i3].reshape(flat_chain.shape[0], nc+sl), [prs_new[x] for x in i3],
                    show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'mix3_')
    pplots.triangle(flat_chain[:,i4].reshape(flat_chain.shape[0], nc+sl), [prs_new[x] for x in i4], 
                    show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'mix4_')

    # Best fitting profile on SZ surface brightness
    pplots.fitwithmod(sz, perc_sz, ci=ci, plotdir=plotdir)

    # Radial pressure profile
    p_prof = [trace.posterior['press'+str(i)].data.reshape(samples.shape[0], -1) for i in range(nc)]
    p_quant = [pplots.get_equal_tailed(pp, ci=ci) for pp in p_prof]
    [np.savetxt('%s/press_prof_%s.dat' % (savedir, c), pq) for c, pq in zip(clus, p_quant)]
    pplots.plot_press(sz.r_pp, p_quant, ci=ci, plotdir=plotdir)

    with model:
        univpress = [pmx.eval_in_model(press.functional_form(shared(sz.r_pp[i]), univpars[i]))[:,0] 
                     for i in range(nc)]

    # Outer slope posterior distribution
    # slopes = np.zeros(flat_chain.shape[0])
    # for i in range(slopes.size):
    #     with pm.Model() as mod:
    #         P_0 = pm.Uniform("P_0", lower=0, upper=1, testval=flat_chain[i,0])
    #         a = pm.Uniform('a', lower=0.5, upper=10., testval=flat_chain[i,1])
    #         b = pm.Uniform('b', lower=3, upper=20, testval=flat_chain[i,2])
    #         c = .014#pm.Deterministic('c', shared(np.repeat(.014, shape)))
    #         r_p = pm.Uniform('r_p', lower=100., upper=1000., testval=flat_chain[i,3])
    #         ped = pm.Uniform("ped", lower=-1, upper=1, testval=flat_chain[i,4])
    #         pars = P_0, a, b, c, r_p, ped
    #         slopes[i] = pmx.eval_in_model(press.functional_form(press.r_out, pars, logder=True))

    # slopes = pfuncs.get_outer_slope(flat_chain, press, press.r_out)
    # pplots.hist_slopes(slopes, ci=ci, plotdir=plotdir)

if __name__ == '__main__':
    main()
