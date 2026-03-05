import preprofit_funcs as pfuncs
import preprofit_plots as pplots
import numpy as np
from astropy import units as u
import cloudpickle
import pymc as pm
import arviz as az

### Global and local variables
savedir = './spt/hier/'
plotdir = savedir
with open('%s/press_obj.pickle' % savedir, 'rb') as f:
    press = cloudpickle.load(f)
with open('%s/szdata_obj.pickle' % savedir, 'rb') as f:
    sz = cloudpickle.load(f)
with open('%s/model.pickle' % savedir, 'rb') as f:
    model = cloudpickle.load(f)

name = 'preprofit'
ci = 68
nc = len(sz.clus)

trace = az.from_netcdf("%s/trace.nc" % savedir)
prs = [k for k in trace.posterior.keys()]
prs = prs[:np.where([p[:2] in ['br', 'pr'] for p in prs])[0][0]]
prs = np.roll(prs, 1 if any(p.startswith('sigma') for p in prs) else 0)

nk = sum(['lgP_{%s,i}' % j in prs for j in range(10)])

samples = []
for (i, par) in enumerate(prs):
    res = trace.posterior[par].data.reshape(np.prod(trace.posterior[prs[0]].shape[:2]), -1)
    for j in range(res.shape[1]):
        samples.append(res[:,j])
samples = np.array(samples).T

prs_ext_kn = [
        [p.replace('k', str(k)) for k in range(nk)] if nc > 1 else '' for p in (prs[:2])]+[
            ['z_dep_%s' % _ for _ in range(trace.posterior.z_dep.shape[-1])] if 'z_dep' in prs else '']+[
                ['M_dep_%s' % _ for _ in range(trace.posterior.M_dep.shape[-1])] if 'M_dep' in prs else '']+[
        [p.replace('i', str(i)) for i in range(nc)] for p in
        (prs[np.where(np.array(prs)=='lgP_{0,i}')[0][0]:-1])]+[[
            prs[-1]+'_{%s}' % i for i in range(nc)]]
prs_ext_clus = [
        [p.replace('k', str(k)) for k in range(nk)] if nc > 1 else '' for p in (prs[:2])]+[
            ['z_dep_%s' % _ for _ in range(trace.posterior.z_dep.shape[-1])] if 'z_dep' in prs else '']+[
                ['M_dep_%s' % _ for _ in range(trace.posterior.M_dep.shape[-1])] if 'M_dep' in prs else '']+[
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
fact = 10**int(-np.sign(sz.flux_data[0][1][0])*np.round(np.log10(np.abs(sz.flux_data[0][1][0].value)), 0))
pplots.fitwithmod(sz, perc_sz, press.eq_kpc_as, rbins=None if type(press)==pfuncs.Press_gNFW else np.array(
    [press.knots[_]/press.kpc_as[_]*u.arcsec for _ in range(nc)]), peds=np.mean(trace.posterior['peds'].data, axis=(0,1)), ind_fits=None, fact=fact, ci=ci, plotdir=plotdir)

# Cornerplots
ind_clus = [[k+np.cumsum([len(p) for p in [[]]+prs_ext_clus[:-1]])[j] for k in range(len(p))] for j,p in enumerate(prs_ext_clus)]
pplots.triangle([samples[:,i] for i in ind_clus], prs_ext_clus, model, fact_ped=1e5, 
                show_lines=True, show_title=False, col_lines='b', ci=ci, plotdir=plotdir)

# Cornerplots
if nc > 1:
    npop = np.where(np.atleast_1d(prs)=='lgP_{0,i}')[0][0]

    # Population parameters only
    pop_par = [l for i in prs_ext_kn[:npop] for l in i]
    pplots.triangle([samples[:,:len(pop_par)]], [pop_par], model, show_lines=False, 
                    show_title=False, col_lines='b', ci=ci, plotdir=plotdir+'pop_')

# Radial pressure profiles
p_prof = [trace.posterior['press_%s' % _].data.reshape(samples.shape[0], -1) for _ in range(nc)]
p_quant = [pplots.get_equal_tailed(pp, ci=ci) for pp in p_prof]
pplots.plot_press(sz.r_pp, p_quant, clus=sz.clus, ci=ci, plotdir=plotdir, rbins=None if type(press)==pfuncs.Press_gNFW else press.knots)

# Outer slope posterior distribution
if press.slope_prior:
    slopes = np.array([trace.posterior['slope_%s' % _].data.flatten() for _ in range(nc)])
    pplots.hist_slopes(slopes, sz.clus, ci=ci, plotdir=plotdir)
