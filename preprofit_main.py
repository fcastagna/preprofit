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

### Global and local variables

## Cluster cosmology
H0 = 70 # Hubble constant at z=0
Om0 = 0.3 # Omega matter
cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)	

# Select clusters to analyse
names, reds, M500 = np.loadtxt('./data/fullsample_SPT.txt', skiprows=1, dtype=('str', 'str'), usecols=(0,3,4), unpack=1)[:,:40]
clus = []
for i, n in enumerate(names):
    try:
        pfuncs.read_data('./data/press_data_'+n+'.dat', ncol=3, units=[u.arcsec, u.Unit(''), u.Unit('')])
        clus.append(n)
    except:
        pass
clus = [clus[x] for x in [0]]#[0,17,13,15,3,16,18,10,2,9]]

# 
nc = len(clus)
ind = [np.where(clus[i]==names)[0][0] for i in range(nc)]
z = np.float64(reds)[ind] # redshift
M500 = np.float64(M500)[ind]*u.Msun # M500
kpc_as = cosmology.kpc_proper_per_arcmin(z).to('kpc arcsec-1') # number of kpc per arcsec
eq_kpc_as = [(u.arcsec, u.kpc, lambda x: x*kpc_as.value, lambda x: x/kpc_as.value)] # equation for switching between kpc and arcsec
r500 = ((3/4*M500/(500.*cosmology.critical_density(z)*np.pi))**(1/3)).to(u.kpc)

## Beam and transfer function
# Beam file already includes transfer function?
beam_and_tf = True

# Beam and transfer function. From input data or Gaussian approximation?
beam_approx = True
tf_approx = False
fwhm_beam = [75]*u.arcsec#None # fwhm of the normal distribution for the beam approximation
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
crop_image = False#True # adopt or do not adopt?
cropped_side = 200 # side of the cropped image (automatically set to odd value)

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
r_out = r500*1.1 # large radius for the slope prior
max_slopeout = -2. # maximum value for the slope at r_out

## Pressure modelization
# 3 models available: 1 parametric (Generalized Navarro Frenk and White), 2 non parametric (restricted cubic spline / power law interpolation)

# 1. Generalized Navarro Frenk and White
# press = pfuncs.Press_gNFW(eq_kpc_as=eq_kpc_as, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)
## Parameters setup
# logunivpars = press.get_universal_params(cosmology, z, M500=M500)

# 2. Restricted cubic spline
knots = np.outer([.1, .3, .5, .75, 1], r500).T
press = pfuncs.Press_rcs(knots=knots, eq_kpc_as=eq_kpc_as, slope_prior=slope_prior, r_out=r_out, max_slopeout=max_slopeout)

# 3. Power law interpolation
# rbins = np.outer([.1, .3, .5, .75, 1], r500).T
# press = pfuncs.Press_nonparam_plaw(rbins=rbins, eq_kpc_as=eq_kpc_as, slope_prior=slope_prior, max_slopeout=max_slopeout)

univpars = press.get_universal_params(cosmology, z, M500=M500)
press_knots = np.mean(univpars, axis=0)
std_knots = np.std(univpars, axis=0)

nk = len(press_knots)
if type(press) != pfuncs.Press_gNFW:
    print("Knots")
    print(np.mean(rbins if type(press)==pfuncs.Press_nonparam_plaw else knots, axis=0))
    print("Universal pressure values (log10)")
    print(press_knots)

## Model definition
with pm.Model() as model:
    # Customize the prior distribution of the parameters using Pymc3 distributions, optionally setting the starting value as initval
    # To exclude a parameter from the fit, just set it at a fixed value 
    if type(press) == pfuncs.Press_gNFW:
        nps = 4
        [pm.HalfNormal('sig'+str(i), sigma=.5, initval=.5) for i in range(nps)]
        [pm.Normal(_, mu=logunivpars[0][i], sigma=.1) for i,_ in enumerate(['Ps', 'a', 'b', 'c'])]
        [pm.Normal("Ps_"+str(i), mu=model['Ps'], sigma=.1) for i in range(nc)]
        [pm.Normal('a_'+str(i), mu=model['a'], sigma=.5) for i in range(nc)]
        [pm.Normal('b_'+str(i), mu=model['b'], sigma=.5) for i in range(nc)]
        [pm.Normal('c_'+str(i), mu=model['c'], sigma=.5) for i in range(nc)]
        c500=1.177
        logr_p = np.log10(r500.value/c500)
    else:
        [pm.HalfNormal('sig'+str(i), sigma=.5, initval=.5) for i in range(nk)]
        [pm.Normal('P'+str(i), mu=press_knots[i], sigma=.1, initval=press_knots[i]) for i in range(nk)]
        [pm.Normal('P'+str(i)+'_'+str(j), mu=model['P'+str(i)], sigma=model['sig'+str(i)], #1
         initval=press_knots[i]) for j in range(nc) for i in range(nk)]

# Sampling step
mystep = 10.*u.arcsec # constant step (values higher than (1/7)*FWHM of the beam are not recommended)
# NOTE: when tf_source_team = 'SPT', be careful to adopt the same sampling step used for the transfer function

# Uncertainty level
ci = 68

# -------------------------------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------------------------------

def main():

    # Flux density data
    flux_data = [pfuncs.read_data(fl, ncol=3, units=flux_units) for fl in flux_filename] # radius, flux density, statistical error
    maxr_data = [flux_data[i][0][-1].value for i in range(len(flux_data))]*flux_data[0][0][-1].unit # largest radius in the data
    maxr_data = maxr_data.mean()

    # PSF computation and creation of the 2D image
    beam_2d, fwhm = pfuncs.mybeam(mystep, maxr_data, eq_kpc_as=eq_kpc_as, approx=beam_approx, filename=beam_filename, units=beam_units, crop_image=crop_image, 
                                  cropped_side=cropped_side, normalize=True, fwhm_beam=fwhm_beam)

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
    radius = np.arange(0., (mymaxr+mystep.to(mymaxr.unit, equivalencies=eq_kpc_as)).value, 
                       mystep.to(mymaxr.unit, equivalencies=eq_kpc_as).value)*mymaxr.unit # array of radii
    radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
    sep = radius.size//2 # index of radius 0
    r_pp = [np.arange(mystep.to(u.kpc, equivalencies=eq_kpc_as)[i].value, (R_b.to(u.kpc, equivalencies=eq_kpc_as)+mystep.to(u.kpc, equivalencies=eq_kpc_as)[i]).value, 
                      mystep.to(u.kpc, equivalencies=eq_kpc_as)[i].value)*u.kpc for i in 
            range(nc)] # radius in kpc used to compute the pressure profile (radius 0 excluded)
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
        press.ind_low = [np.maximum(0, np.digitize(sz.r_pp[i], press.rbins[i])-1) for i in range(nc)] # lower bins indexes
        press.r_low = [press.rbins[i][press.ind_low[i]] for i in range(nc)] # lower radial bins
        press.alpha_ind = [np.minimum(press.ind_low[i], len(press.rbins[i])-2) for i in range(nc)]# alpha indexes
        #pplots.pop_plot(sz, eq_kpc_as=eq_kpc_as, r500=r500, knots=press.rbins, plotdir=plotdir)

    # Add pedestal component to the model
    with model:
        [pm.Uniform("peds_"+str(i), lower=-10**int(np.round(np.log10(abs(sz.flux_data[0][1].value)), 0)[4:].max()), 
                    upper=10**int(np.round(np.log10(abs(sz.flux_data[0][1].value)), 0)[4:].max()), initval=0.) for i in range(nc)]

    # Save objects
    with open('%s/press_obj_mult_%s.pickle' % (savedir, nc), 'wb') as f:
        cloudpickle.dump(press, f, -1)
    with open('%s/szdata_obj_mult_%s.pickle' % (savedir, nc), 'wb') as f:
        cloudpickle.dump(sz, f, -1)

    ## Sampling
    
    # Starting point research
    ilike = pt.as_tensor([np.inf])
    nn = 0
    while np.isinf(pt.sum(ilike).eval()):
        print('Searching...')
        if nn > 0:
            npar = len(model.free_RVs)
            infs = np.where(infs)[0]
            inds = [i for s in [list(np.arange(npar)[nc:][_*nk:_*nk+nk])+
                                [np.arange(npar)[-nc:][_]] for _ in infs] for i in s]
            pm.draw([model.free_RVs[i] for i in inds])
        if type(press) == pfuncs.Press_gNFW:
            vals = [x.eval() for x in model.free_RVs]
            pars = [[[model.rvs_to_transforms[model.values_to_rvs[m]].forward(m2.eval(), *m2.owner.inputs)
                     if model.rvs_to_transforms[model.values_to_rvs[m]] is not None else m2
                     for m, m2 in zip(model.continuous_value_vars, model.free_RVs)][i]]+
                    [model.rvs_to_transforms[model.values_to_rvs[m]].forward(m2.eval(), *m2.owner.inputs)
                     if model.rvs_to_transforms[model.values_to_rvs[m]] is not None else m2
                     for m, m2 in zip(model.continuous_value_vars[nc:nc+3], model.free_RVs[nc:nc+3])]+
                    [logr_p[i]]+
                    [[m2 for m2 in model.free_RVs[-nc:]][i]]
                    for i in range(nc)]
        else:
            vals = [x.eval() for x in model.free_RVs]
            pars = [[model.rvs_to_transforms[model.values_to_rvs[m]].forward(m2.eval(), *m2.owner.inputs) 
                     if model.rvs_to_transforms[model.values_to_rvs[m]] is not None else m2 
                     for m, m2, v in zip(model.continuous_value_vars[:nk], model.free_RVs[:nk], vals[:nk])]+
                    [model.rvs_to_transforms[model.values_to_rvs[m]].forward(m2.eval(), *m2.owner.inputs) 
                     if model.rvs_to_transforms[model.values_to_rvs[m]] is not None else m2 
                     for m, m2, v in zip(model.continuous_value_vars[(2+i)*nk:(3+i)*nk], model.free_RVs[(2+i)*nk:(3+i)*nk], vals[(2+i)*nk:(3+i)*nk])]+
                    [[m2 for m2 in model.free_RVs[-nc:]][i]]
                    for i in range(nc)]
        with model:
            pars = pars if type(press)==pfuncs.Press_gNFW else [p[nk:] for p in pars]
            like, pprof, maps, slopes = zip(*map(
                lambda i, pr, szr, sza, szl, dm, szfl: pfuncs.whole_lik(
                    pr, press, szr, sza, sz.filtering, sz.conv_temp_sb, szl, sz.sep, dm, sz.radius[sz.sep:].value, szfl, i, 'll'), 
                np.arange(nc), pars, sz.r_pp, sz.abel_data, sz.dist.labels, sz.dist.d_mat, sz.flux_data))
            pm.Potential('pv_like'+str(nn), pt.sum(like))
            infs = [int(np.isinf(l.eval())) for l in like]
            print('like')
            print(pt.sum(like).eval())
            # factor_logps_fn = [pt.sum(factor) for factor in model.logp(model.basic_RVs + [model.potentials[-1]], sum=False)]
            [model.set_initval(n, v) for n, v in zip(model.free_RVs, vals)]
            # check = model.compile_fn(factor_logps_fn)(model.initial_point())
            # print('logp')
            # print(check[-1])
            # ilike = pt.sum([pt.sum(like), shared(check[-1])])
            # print('sum')
            print(ilike.eval())
            nn += 1
            if nn == 100:
                raise RuntimeError('Valid starting point not found after 100 attempts. Execution stopped')
    with model:
        map_prof = [pm.Deterministic('bright'+str(i), maps[i]) for i in range(nc)]
        p_prof = [pm.Deterministic('press'+str(i), pprof[i]) for i in range(nc)]
        like = pm.Potential('like', pt.sum(like))
        if slope_prior:
            slope = [pm.Deterministic('slope'+str(i), slopes[i]) for i in range(nc)]
        ll = pm.Deterministic('loglik', model.logp())
        with open('%s/model_%s.pickle' % (savedir, nc), 'wb') as m:
            cloudpickle.dump(model, m, -1)
        start_guess = [np.atleast_2d(m.eval()) for m in map_prof]
    pplots.plot_guess(
        start_guess, sz, knots=None if type(press) == pfuncs.Press_gNFW else 
        [[r.to(sz.flux_data[0][0].unit, equivalencies=eq_kpc_as)[j].value for i, r in enumerate(press.knots[j] if type(press == pfuncs.Press_rcs) 
                                                                                                else press.rbins[j])] for j in range(nc)], plotdir=plotdir)
    # pplots.Arnaud_press(sz.r_pp, [p.eval() for p in p_prof])

    with model:
        # start = pm.find_MAP(start=model.initial_point(), model = model)
        trace = pm.sample(draws=int(5000), tune=int(1000), chains=4,
                          return_inferencedata=True, step=pm.Metropolis(), initvals=model.initial_point())
    trace.to_netcdf("%s/trace_mult.nc" % savedir)
    # trace = az.from_netcdf("%s/trace_mult.nc" % savedir)
    prs = [k for k in trace.posterior.keys()]
    prs = prs[:np.where([p[:2]=='br' for p in prs])[0][0]]
    samples = np.zeros((trace['posterior'][prs[-1]].size, len(prs)))
    for (i, par) in enumerate(prs):
        samples[:,i] = np.array(trace['posterior'][par]).flatten()

    # samples = np.zeros((int(sum([trace['posterior'][p].size for p in prs])/2/nc), 2*nc))
    # for i in range(nc):
    #     samples[:,i] = np.array(trace['posterior'][prs[0]])[:,:,i].flatten()
    #     samples[:,nc+i] = np.array(trace['posterior'][prs[1]])[:,:,i].flatten()
    # np.savetxt('%s/trace_mult.txt' % savedir, samples)

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
 
    # sl = sum([x in prs for x in ['a', 'b', 'c']])
    pm.summary(trace, var_names=prs)
    pplots.traceplot_new(trace, prs, nc, trans_ped=lambda x: 1e4*x, plotdir=savedir)
    # axes = az.plot_trace(trace, var_names=prs)
    # fig = axes.ravel()[0].figure
    # fig.savefig('%s/tp.pdf' % plotdir)
    """
    axes = az.plot_trace(trace, var_names=['sig'+str(i) for i in range(nk)],#prs[-(nk+nc):-nc], 
                         transform=np.log10)
    fig = axes.ravel()[0].figure
    fig.savefig('%s/traceplots/traceplot_logsigmas.pdf' % plotdir)
    axes = az.plot_trace(trace, var_names=['P'+str(i) for i in range(nk)])
    fig = axes.ravel()[0].figure
    fig.savefig('%s/traceplots/traceplot_Ps_joint.pdf' % plotdir)
    for i in range(nc):
        axes = az.plot_trace(trace, var_names=['P'+str(i)+'_'+str(j) for j in range(nc) for i in range(nk)],#prs[(i+1)*nk:(i+2)*nk], 
                             transform=lambda x: 10**x) if type(press) == pfuncs.Press_gNFW else az.plot_trace(trace, var_names=prs[(i+1)*nk:(i+2)*nk])#, transform=np.log10)#lambda x: 10**x)#np.exp)
        fig = axes.ravel()[0].figure
        fig.savefig('%s/traceplots/traceplot_logP%ss.pdf' % (plotdir, i))
    axes = az.plot_trace(trace, var_names=['peds_'+str(i) for i in range(nc)], transform=lambda x: 1e5*x)
    fig = axes.ravel()[0].figure
    fig.suptitle('ped (10^5)')
    fig.savefig('%s/traceplots/traceplot_peds.pdf' % plotdir)
    if sl != 0:
        axes = az.plot_trace(trace, var_names=['a', 'b', 'c'],#prs[nc:nc+sl], 
                             transform=lambda x: 10**x)#np.exp)
        fig = axes.ravel()[0].figure
        fig.savefig('%s/traceplots/traceplot_slopes.pdf' % plotdir)
    # Best fitting profile on SZ surface brightness
    #rbins = np.tile(np.atleast_2d([50, 150, 300, 500]).T*kpc_as, nc).T.value*u.kpc
    """
    pplots.fitwithmod(sz, perc_sz, eq_kpc_as, clus=clus, rbins=None if type(press)==pfuncs.Press_gNFW else press.knots#rbins
                      , peds=[trace.posterior['peds_'+str(j)].data.mean() for j in range(nc)], fact=1e5, ci=ci, plotdir=plotdir)
    """
    # Forest plots
    for i in range(nk):
        axes = az.plot_forest(trace, var_names=['P'+str(i)+'_'+str(j) for j in range(nc)])#prs[i:i+(nc+1)*nk][::nk])
        fig = axes.ravel()[0].figure
        fig.savefig('%s/forests/forest_P%s.pdf' % (plotdir, i))
    axes = az.plot_forest(trace, var_names=['sig'+str(i) for i in range(nk)]#prs[-nc-nk:-nc]
                          , transform=np.log10)
    fig = axes.ravel()[0].figure
    fig.savefig('%s/forests/forest_sigmas.pdf' % plotdir)
    axes = az.plot_forest(trace, var_names=['peds_'+str(i) for i in range(nc)],#prs[-nc:], 
                          transform=lambda x: 1e5*x)
    fig = axes.ravel()[0].figure
    fig.savefig('%s/forests/forest_peds.pdf' % plotdir)
    # Cornerplots
    #flat_chain[:,:-nc] = 10**(flat_chain[:,:-nc])#np.exp(flat_chain[:,:-nc])
    if type(press) == pfuncs.Press_gNFW:
        pplots.triangle(flat_chain[:,np.where([p[:3]=='sig' for p in prs])[0]#:nc
                                   ], ['log'+x for x in prs[:nc]], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/cornerplots/logP0_')
    else:
        pplots.triangle(flat_chain[:,np.where([p[:3]=='sig' for p in prs])[0]], ['sig_'+str(i) for i in range(nk)], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/cornerplots/sigmas_')
        pplots.triangle(flat_chain[:,[np.where([p=='P'+str(i) for p in prs])[0][0] for i in range(nk)]], ['logP'+str(i) for i in range(nk)], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/cornerplots/logPs_joint_')
        # pplots.triangle(flat_chain[:,-(nk+nc):-nc], ['log'+x for x in prs[-(nk+nc):-nc]], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/cornerplots/sigmas_')
        # pplots.triangle(flat_chain[:,:nk], ['log'+x for x in prs[:nk]], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/cornerplots/logPs_joint_')
        for i in range(nc):
            pplots.triangle(flat_chain[:,[np.where([p=='P'+str(j)+'_'+str(i) for p in prs])[0][0] for j in range(nk)]], ['logP'+str(j)+'_'+str(i) for j in range(nk)], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/cornerplots/logP%ss_' % str(i))
            # pplots.triangle(flat_chain[:,(i+1)*nk:(i+2)*nk], ['log'+x for x in prs[(i+1)*nk:(i+2)*nk]], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/cornerplots/logP%ss_' % str(i))
    if sl != 0:
        pplots.triangle(10**flat_chain[:,nc:nc+sl], prs[nc:nc+sl], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/cornerplots/slopes_')
    pplots.triangle(flat_chain[:,[np.where([p=='peds_'+str(i) for p in prs])[0][0] for i in range(nc)]]*1e5, ['peds_'+str(i) for i in range(nc)], show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/cornerplots/peds_')
    # i1 = [it for s in [[x for x in np.arange(nc//2)], [x for x in np.arange(nc,nc+sl+nc//2)]] for it in s]
    # i2 = [it for s in [[x for x in np.arange(nc//2)], [x for x in np.arange(nc,nc+sl)], [x for x in np.arange(nc+sl+nc//2,2*nc+sl)]] for it in s]
    # i3 = [x for x in np.arange(nc//2,nc+sl+nc//2)]
    # i4 = [it for s in [[x for x in np.arange(nc//2,nc+sl)], [x for x in np.arange(nc+sl+nc//2,2*nc+sl)]] for it in s]
    # pplots.triangle(flat_chain[:,i1].reshape(flat_chain.shape[0], nc+sl), [prs_new[x] for x in i1], 
    #                 show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'mix1_')
    # pplots.triangle(flat_chain[:,i2].reshape(flat_chain.shape[0], nc+sl), [prs_new[x] for x in i2], 
    #                 show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'mix2_')
    # pplots.triangle(flat_chain[:,i3].reshape(flat_chain.shape[0], nc+sl), [prs_new[x] for x in i3],
    #                 show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'mix3_')
    # pplots.triangle(flat_chain[:,i4].reshape(flat_chain.shape[0], nc+sl), [prs_new[x] for x in i4], 
    #                 show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'mix4_')

    """
    # pplots.triangle(flat_chain, prs, show_lines=True, col_lines='r', ci=ci, plotdir=plotdir+'/cornerplots/all_')
    # Radial pressure profile
    # pars = [np.log(flat_chain[i,nk:2*nk]) for i in range(flat_chain.shape[0])]
    # p_prof = [[press.functional_form(shared(sz.r_pp[i]), pars[i], i) for i in range(nk)] for j in range(len(pars))]
    # p_quant = [pplots.get_equal_tailed(pp, ci=ci) for pp in p_prof]
    # [np.savetxt('%s/press_prof_%s.dat' % (savedir, c), pq) for c, pq in zip(clus, p_quant)]
    # import sys; sys.exit()

    p_prof = [trace.posterior['press'+str(i)].data.reshape(samples.shape[0], -1) for i in range(nc)]
    p_quant = [pplots.get_equal_tailed(pp, ci=ci) for pp in p_prof]
    [np.savetxt('%s/profiles/press_prof_%s.dat' % (savedir, c), pq) for c, pq in zip(clus, p_quant)]
    with model:
        # univpress=None
        univpress = [press.functional_form(shared(sz.r_pp[i]), univpars[i], i).eval()[0] for i in range(nc)]
    stef = None#np.loadtxt('%s/press_fit_gtm2_flat.dat' % savedir).T[1:]
    # print(sz.r_pp[0].size); import sys; sys.exit()
    pplots.spaghetti_press(sz.r_pp, p_prof, clus=clus, nl=100, ci=ci, univpress=univpress, plotdir=plotdir, rbins=None if type(press)==pfuncs.Press_gNFW else press.knots, stef=stef)
    pplots.plot_press(sz.r_pp, p_quant, clus=clus, ci=ci, univpress=univpress, plotdir=plotdir, rbins=None if type(press)==pfuncs.Press_gNFW else press.knots, stef=stef)

    # Compare gnfw vs p-law
    index = 0
    '''
    t_g = az.from_netcdf('./%s/%s/trace_mult.nc' % ('gnfw', clus[index]))
    t_p = az.from_netcdf('./%s/%s/trace_mult.nc' % ('plaw', clus[index]))
    #print(t_g.posterior['a'].data.shape);# import sys; sys.exit()
    #print(t_p.posterior['P0'].data.shape); import sys; sys.exit()
    p_g = [t_g.posterior['press'+str(i)].data.reshape(samples.shape[0], -1) for i in range(nc)]
    p_p = [t_p.posterior['press'+str(i)].data.reshape(samples.shape[0], -1) for i in range(nc)]
    p_g = [pplots.get_equal_tailed(pp, ci=ci) for pp in p_g]
    p_p = [pplots.get_equal_tailed(pp, ci=ci) for pp in p_p]
    pplots.press_compare(sz.r_pp, p_g, p_p, ci=ci, univpress=None, plotdir=plotdir, rbins=rbins, stef=stef)
    '''
    # Outer slope posterior distribution
    # slopes = pfuncs.get_outer_slope(trace, flat_chain, press, r_out=press.r_out if type(press)==pfuncs.Press_gNFW else None, r_p = r500.value/c500 if type(press)==pfuncs.Press_gNFW else None)
    slopes = np.array([trace.posterior['slope'+str(i)].data.flatten() for i in range(nc)]).flatten()
    pplots.hist_slopes(slopes.flatten(), ci=ci, plotdir=plotdir)

if __name__ == '__main__':
    main()
