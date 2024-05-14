import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy import units as u
import numpy as np
import corner
import arviz as az

plt.style.use('classic')
font = {'size': 10}
plt.rc('font', **font)

def plot_guess(out_prof, sz, knots=None, plotdir='./'):
    '''
    Modeled profile resulting from starting parameters VS observed data
    -------------------------------------------------------------------
    out_prof = modeled profile
    sz = class of SZ data
    plotdir = directory where to place the plot
    '''
    plt.clf()
    pdf = PdfPages(plotdir+'starting_guess.pdf')
    for i in range(len(sz.flux_data)):
        if len(sz.flux_data) > 1:
            plt.subplot(221+i%4)
        plt.plot(sz.radius[sz.sep:], out_prof[i][0], color='r', label='Starting guess')
        plt.errorbar(sz.flux_data[i][0].value, sz.flux_data[i][1].value, yerr=sz.flux_data[i][2].value,
                     fmt='o', fillstyle='none', color='black', label='Observed data')
        if knots is not None:
            [plt.axvline(k, linestyle=':') for k in knots[i]]
        if i == 0:
            plt.legend()
        plt.ylim(np.min([fl[1].min() for fl in sz.flux_data]), np.max([fl[1].max() for fl in sz.flux_data]))
        plt.xlim(0., (sz.flux_data[i][0][-1]+np.diff(sz.flux_data[i][0])[-1]).value)
        if i%4 > 1:
            plt.xlabel('Radius ('+str(sz.flux_data[i][0].unit)+')')
        if i%2 == 0:
            plt.ylabel('Surface brightness ('+str(sz.flux_data[i][1].unit)+')')
        if (i+1)%4 == 0:
            pdf.savefig()
            plt.clf()
    pdf.savefig()
    pdf.close()

def traceplot(cube_chain, param_names, plotw=20, seed=None, ppp=4, labsize=18., ticksize=10., plotdir='./'):
    '''
    Traceplot of the MCMC
    ---------------------
    cube_chain = 3d array of sampled values (nw x niter x nparam)
    param_names = names of the parameters
    plotw = number of random walkers that we wanna plot (default is 20)
    seed = random seed (default is None)
    ppp = number of plots per page
    labsize = label font size
    ticksize = ticks font size
    plotdir = directory where to place the plot
    '''
    plt.clf()
    nw, nsteps = cube_chain.shape[:2]
    np.random.seed(seed)
    ind_w = np.random.choice(nw, plotw, replace=False)
    param_latex = ['${}$'.format(i) for i in param_names]
    pdf = PdfPages(plotdir+'traceplot.pdf')
    for i in np.arange(cube_chain.shape[2]):
        plt.subplot(ppp, 1, i%ppp+1)
        for j in ind_w:
            plt.plot(np.arange(nsteps)+1, cube_chain[j,:,i], linewidth=.2)
            plt.tick_params(labelbottom=False)
        plt.ylabel('%s' %param_latex[i], fontdict={'fontsize': labsize})
        plt.tick_params('y', labelsize=ticksize)
        if (abs((i+1)%ppp) < 0.01):
            plt.tick_params(labelbottom=True)
            plt.xlabel('Iteration number')
            pdf.savefig(bbox_inches='tight')
            if i+1 < cube_chain.shape[2]:
                plt.clf()
        elif i+1 == cube_chain.shape[2]:
            plt.tick_params(labelbottom=True)
            plt.xlabel('Iteration number')
            pdf.savefig(bbox_inches='tight')
    pdf.close()

def traceplot_new(trace, prs, prs_latex, nc, fact_ped, nk=5, trans_ped=None, ppp=10, div=None, plotdir='./'):
    '''
    '''
    plt.clf()
    pdf = PdfPages(plotdir+'traceplot.pdf')
    for i in range(int((len(prs)-nc-nk-.5)//ppp)+1):
        axes = az.plot_trace(trace, var_names=prs[i*ppp:min((i+1)*ppp,len(prs))], divergences=div)
        [axes[_][j].set_title('') for j in [0,1] for _ in range(len(axes))]
        [axes[_][0].set_ylabel(prs_latex[i*ppp:min((i+1)*ppp,len(prs))][_], fontdict={'fontsize':20}) for _ in range(len(axes))]
        axes[-1][0].set_xlabel('Value')
        axes[-1][1].set_xlabel('Iteration')
        pdf.savefig(bbox_inches='tight')
    for i in range(int((nc-.5)//ppp)+1):
        axes = az.plot_trace(trace, var_names=prs[-nc-nk:-nk][i*ppp:min((i+1)*ppp,nc)],
                      transform=trans_ped, divergences=div)
        [axes[_][j].set_title('') for j in [0,1] for _ in range(len(axes))]
        [axes[_][0].set_ylabel(prs_latex[-nc-nk:-nk][i*ppp:min((i+1)*ppp,nc)][_], fontdict={'fontsize':20}) for _ in range(len(axes))]
        axes[-1][0].set_xlabel('Value [10$^%i$]' % np.log10(fact_ped))
        axes[-1][1].set_xlabel('Iteration')
        pdf.savefig(bbox_inches='tight')
    for i in range(int((nk-.5)//ppp)+1):
        axes = az.plot_trace(trace, var_names=prs[-nk:], divergences=div)
        [axes[_][j].set_title('') for j in [0,1] for _ in range(len(axes))]
        [axes[_][0].set_ylabel(prs_latex[-nk:][_], fontdict={'fontsize':20}) for _ in range(len(axes))]
        axes[-1][0].set_xlabel('Value')
        axes[-1][1].set_xlabel('Iteration')
        pdf.savefig(bbox_inches='tight')
    plt.clf()
    pdf.close()

def triangle(mat_chain, param_names, show_lines=True, col_lines='r', ci=95, labsize=25., titsize=15., legend=True, plotdir='./'):
    '''
    Univariate and multivariate distribution of the parameters in the MCMC
    ----------------------------------------------------------------------
    mat_chain = 2d array of sampled values ((nw x niter) x nparam)
    param_names = names of the parameters
    show_lines = whether to show lines for median and uncertainty interval (boolean, default is True)
    col_lines = line colour (default is red)
    ci = uncertainty level of the interval
    labsize = label font size
    titsize = titles font size
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'cornerplot.pdf')
    for _ in range(len(mat_chain)):
        mat_chain = [np.array(m) for m in mat_chain]
        plt.clf()
        param_latex = ['${}$'.format(i) for i in param_names[_]]
        fig = corner.corner(mat_chain[_], labels=param_latex, title_kwargs={'fontsize': titsize}, label_kwargs={'fontsize': labsize})
        axes = np.array(fig.axes).reshape((len(param_names[_]), len(param_names[_])))
        plb, pmed, pub = get_equal_tailed(mat_chain[_], ci=ci)
        for i in range(len(param_names[_])):
            l_err, u_err = pmed[i]-plb[i], pub[i]-pmed[i]
            axes[i,i].set_title('%s = $%.2f_{-%.2f}^{+%.2f}$' % (param_latex[i], pmed[i], l_err, u_err), fontdict={'fontsize': titsize})
            if show_lines:
                axes[i,i].axvline(pmed[i], color=col_lines, linestyle='--', label='Median')
                axes[i,i].axvline(plb[i], color=col_lines, linestyle=':', label='%i%% CI' % ci)
                axes[i,i].axvline(pub[i], color=col_lines, linestyle=':', label='_nolegend_')
                for yi in range(len(param_names[_])):
                    for xi in range(yi):
                        axes[yi,xi].axvline(pmed[xi], color=col_lines, linestyle='--')
                        axes[yi,xi].axhline(pmed[yi], color=col_lines, linestyle='--')
                        axes[yi,xi].plot(plb[xi], plb[yi], marker=1, color=col_lines)
                        axes[yi,xi].plot(plb[xi], plb[yi], marker=2, color=col_lines)
                        axes[yi,xi].plot(plb[xi], pub[yi], marker=1, color=col_lines)
                        axes[yi,xi].plot(plb[xi], pub[yi], marker=3, color=col_lines)
                        axes[yi,xi].plot(pub[xi], plb[yi], marker=0, color=col_lines)
                        axes[yi,xi].plot(pub[xi], plb[yi], marker=2, color=col_lines)
                        axes[yi,xi].plot(pub[xi], pub[yi], marker=0, color=col_lines)
                        axes[yi,xi].plot(pub[xi], pub[yi], marker=3, color=col_lines)
                if legend: fig.legend(('Median', '%i%% CI' % ci), loc='lower center', ncol=2, bbox_to_anchor=(0.55, 0.95), fontsize=titsize+len(param_names))
        pdf.savefig(bbox_inches='tight')
    pdf.close()

def get_equal_tailed(data, ci=95):
    '''
    Computes the median and lower/upper limits of the equal tailed uncertainty interval
    -----------------------------------------------------------------------------------
    ci = uncertainty level of the interval
    ----------------------------------------
    RETURN: lower bound, median, upper bound
    '''
    low, med, upp = map(np.atleast_1d, np.percentile(data, [50-ci/2, 50, 50+ci/2], axis=0))
    return np.array([low, med, upp])

def fitwithmod(sz, perc_sz, eq_kpc_as, clus, rbins=None, peds=None, fact=1, ci=95, plotdir='./'):
    '''
    Surface brightness profile (points with error bars) and best fitting profile with uncertainties
    -----------------------------------------------------------------------------------------------
    sz = class of SZ data
    perc_sz = best (median) SZ fitting profiles with uncertainties
    ci = uncertainty level of the interval
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'fit_on_data.pdf')
    for i in range(len(sz.flux_data)):
        plt.clf()
        plt.title(clus[i])
        lsz, msz, usz = perc_sz[i]*fact
        plt.plot(sz.radius[sz.sep:], msz, color='r', label='Best-fit')
        plt.fill_between(sz.radius[sz.sep:].value, lsz, usz, color='gold', label='%i%% CI' % ci)
        plt.errorbar(sz.flux_data[i][0].value, sz.flux_data[i][1].value*fact, yerr=sz.flux_data[i][2].value*fact, fmt='o', fillstyle='none', color='black', label='Observed data')
        plt.xlim(0., 50+np.ceil(sz.flux_data[i][0][-1].value))
        if rbins is not None:
            [plt.axvline(r, linestyle=':', color='grey', label='_nolegend_') for r in rbins[i]]
        if peds is not None:
            plt.axhline(peds[i]*fact, linestyle=':', color='grey', label='_nolegend_')
        plt.xlabel('Radius ('+str(sz.flux_data[i][0].unit)+')')
        plt.ylabel('Surface brightness ('+str(sz.flux_data[i][1].unit)+('' if sz.flux_data[i][1].unit else 'x ')+'$10^%i$)' % np.log10(fact) if fact != 1 else '')
        pdf.savefig()
    pdf.close()

def plot_press(r_kpc, press_prof, clus, xmin=np.nan, xmax=np.nan, ci=95, univpress=None, rbins=None, stef=None, plotdir='./'):
    '''
    Plot the radial pressure profiles
    ---------------------------------
    r_kpc = radius (kpc)
    press_prof = best fitting pressure profile (median and interval)
    xmin, xmax = x-axis boundaries for the plot (by default, they are obtained based on r_kpc)
    ci = uncertainty level of the interval
    plotdir = directory where to place the plot
    '''
    plt.style.use('classic')
    font = {'size': 10}
    plt.rc('font', **font)
    pdf = PdfPages(plotdir+'press_fit.pdf')
    for i in range(len(press_prof)):
        plt.clf()
        plt.title(clus[i])
        l_press, m_press, u_press = press_prof[i]
        xmin, xmax = np.nanmax([r_kpc[i][0].value, xmin]), np.nanmin([r_kpc[i][-1].value, xmax])
        ind = np.where((r_kpc[i].value > xmin) & (r_kpc[i].value < xmax))
        e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
        plt.plot(r_kpc[i][e_ind], m_press[e_ind])
        plt.fill_between(r_kpc[i][e_ind].value, l_press[e_ind], u_press[e_ind], color='powderblue', label='_nolegend_')
        if rbins is not None:
            [plt.axvline(r.value, linestyle=':', color='grey', label='_nolegend_') for r in rbins[i]]
        plt.xscale('log')
        plt.yscale('log')
        if univpress is not None:
            plt.plot(r_kpc[i][e_ind], univpress[i][e_ind])
        if stef is not None:
            plt.plot(r_kpc[i][e_ind], stef[0][e_ind], color='r', label='Stefano')
            plt.fill_between(r_kpc[i][e_ind].value, stef[1][e_ind], stef[2][e_ind], color='orange', alpha=.25)
        plt.ylim(1e-5, 1e-1)
        plt.xlabel('Radius ('+str(r_kpc[i].unit)+')')
        plt.ylabel('Pressure (keV cm$^{-3}$)')
        plt.suptitle('Radial pressure profile (median + %i%% CI)' % ci)
        if univpress is not None:
            plt.legend(('fitted', 'universal'), loc='lower left')
        plt.xlim(xmin, xmax)
        pdf.savefig()
    pdf.close()

def hist_slopes(slopes, ci=95, plotdir='./'):
    '''
    Plot the histogram of the outer slopes posterior distribution
    -------------------------------------------------------------
    slopes = array of slopes
    ci = uncertainty level of the interval
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'outer_slopes.pdf')
    for _ in range(len(slopes)):
        plt.clf()
        plt.title('Outer slope - Posterior distribution')
        low, med, upp = get_equal_tailed(slopes[_], ci=ci)
        plt.hist(slopes[_], density=True, histtype='step', color='black')
        plt.axvline(med, color='black', linestyle='--', label='Median')
        plt.axvline(low, color='black', linestyle='-.', label='%i%% CI' % ci)
        plt.axvline(upp, color='black', linestyle='-.', label='_nolegend_')
        plt.xlabel('Outer slope')
        plt.ylabel('Density')
        pdf.savefig(bbox_inches='tight')
    pdf.close()

def pop_plot(sz, eq_kpc_as, r500=None, knots=None, plotdir='./'):
    '''
    Surface brightness profile (points with error bars) and best fitting profile with uncertainties
    -----------------------------------------------------------------------------------------------
    sz = class of SZ data
    plotdir = directory where to place the plot
    '''
    plt.clf()
    pdf = PdfPages(plotdir+'population.pdf')
    r_kpc = [np.array([r.to(u.kpc, equivalencies=eq_kpc_as) for r in sz.flux_data[i][0]])[:,i] for i in range(len(sz.flux_data))]
    for i in range(len(sz.flux_data)):
        plt.errorbar(r_kpc[i]/r500[i], sz.flux_data[i][1].value/sz.flux_data[i][1][0])#, yerr=sz.flux_data[i][2].value, fmt='o', linestyle='-', fillstyle='none', label='Observed data')
        plt.plot(r_kpc[i][-1]/r500[i], sz.flux_data[i][1][-1]/sz.flux_data[i][1][0], marker='o', markersize=10)
        plt.xlabel('Normalized radius (r/r500)')
        plt.ylabel('Normalized surface brightness')# ('+str(sz.flux_data[0][1].unit)+')')
        if knots is not None:
            [plt.axvline(k/r500[i], linestyle=':') for k in knots[i]]
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def Arnaud_press(r_kpc, press_prof, xmin=6, xmax=1500, plotdir='./'):
    '''
    Plot the radial pressure profiles
    ---------------------------------
    r_kpc = radius (kpc)
    press_prof = best fitting pressure profile (median and interval)
    xmin, xmax = x-axis boundaries for the plot (by default, they are obtained based on r_kpc)
    ci = uncertainty level of the interval
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'press_Arnaud.pdf')
    for i in range(len(press_prof)):
        ind = np.where((r_kpc[i].value > xmin) & (r_kpc[i].value < xmax))
        e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
        plt.plot(r_kpc[i][e_ind[1:-1]], press_prof[i][0][e_ind[1:-1]])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Radius ('+str(r_kpc[i].unit)+')')
        plt.ylabel('Pressure (keV cm$^{-3}$)')
        plt.xlim(xmin, xmax)
        plt.ylim(1.5e-5, 6e-1)    
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def Arnaud_sing_press(cosmo, z, r500, c500=1.177, a=1.051, b=5.4905, c=0.3081, P0=None, mystep=15*u.arcsec, xmin=85, xmax=820, ymin=1.5e-4, ymax=1e-2, plotdir='./'):
    import preprofit_funcs as pfuncs
    kpc_as = cosmo.kpc_proper_per_arcmin(z).to('kpc arcsec-1')
    eq_kpc_as = [(u.arcsec, u.kpc, lambda x: x*kpc_as.value, lambda x: x/kpc_as.value)] 
    rbins = np.outer([.1,.3,.5,1,2], r500).T
    pbins = [1e-1, 2e-2, 5e-3, 1e-3, 1e-4]*u.Unit('keV/cm3')
    nc = 1
    R_b = 5000*u.kpc
    press = pfuncs.Press_nonparam_plaw(rbins, pbins, eq_kpc_as)
    r_kpc = [np.arange(mystep.to(u.kpc, equivalencies=eq_kpc_as)[i].value, (R_b.to(u.kpc, equivalencies=eq_kpc_as)+mystep.to(u.kpc, equivalencies=eq_kpc_as)[i]).value, 
                      mystep.to(u.kpc, equivalencies=eq_kpc_as)[i].value)*u.kpc for i in range(nc)]
    press.ind_low = [np.maximum(0, np.digitize(r_kpc[i], press.rbins[i])-1) for i in range(nc)] # lower bins indexes
    press.r_low = [press.rbins[i][press.ind_low[i]] for i in range(nc)] # lower radial bins
    press.alpha_ind = [np.minimum(press.ind_low[i], len(press.rbins[i])-2) for i in range(nc)]# alpha indexes
    press = pfuncs.Press_gNFW(eq_kpc_as=eq_kpc_as, slope_prior=1, r_out=1e3*u.kpc, max_slopeout=-2)
    univpars = press.get_universal_params(cosmo, z, r500=r500, c500=c500, a=a, b=b, c=c, P0=P0)
    from pytensor import shared
    press_prof = press.functional_form(shared(r_kpc[0]), univpars[0], 0)
    pdf = PdfPages(plotdir+'press_Arnaud_sing.pdf')
    ind = np.where((r_kpc[0].value > xmin) & (r_kpc[0].value < xmax))
    e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
    plt.plot(r_kpc[0][e_ind], (press_prof[0].eval())[e_ind])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Radius ('+str(r_kpc[0].unit)+')')
    plt.ylabel('Pressure (keV cm$^{-3}$)')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)    
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def Romero_plot(cosmo, z, r500, mystep=15*u.arcsec, xmin=85, xmax=820, ymin=1.5e-4, ymax=1e-2, plotdir='./'):
    import preprofit_funcs as pfuncs
    kpc_as = cosmo.kpc_proper_per_arcmin(z).to('kpc arcsec-1')
    eq_kpc_as = [(u.arcsec, u.kpc, lambda x: x*kpc_as.value, lambda x: x/kpc_as.value)] 
    rbins = np.outer([.073,.134,.216,.349,.564,.910], r500).T#[100, 300, 600, 1000, 2000]*u.kpc
    pbins = [.225, .15, .0744, .0358, .00508, .002]*u.Unit('keV/cm3')
    nc = 1
    R_b = 5000*u.kpc
    press = pfuncs.Press_nonparam_plaw(rbins, pbins, eq_kpc_as)
    r_kpc = [np.arange(mystep.to(u.kpc, equivalencies=eq_kpc_as)[i].value, (R_b.to(u.kpc, equivalencies=eq_kpc_as)+mystep.to(u.kpc, equivalencies=eq_kpc_as)[i]).value, 
                      mystep.to(u.kpc, equivalencies=eq_kpc_as)[i].value)*u.kpc for i in range(nc)]
    press.ind_low = [np.maximum(0, np.digitize(r_kpc[i], press.rbins[i])-1) for i in range(nc)] # lower bins indexes
    press.r_low = [press.rbins[i][press.ind_low[i]] for i in range(nc)] # lower radial bins
    press.alpha_ind = [np.minimum(press.ind_low[i], len(press.rbins[i])-2) for i in range(nc)]# alpha indexes
    from pytensor import shared
    pars = [np.log10(x.value) for x in pbins]
    press_prof = press.functional_form(shared(r_kpc[0]), pars, 0)
    pdf = PdfPages(plotdir+'press_Romero.pdf')
    r_kpc = r_kpc/r500
    ind = np.where((r_kpc[0].value > xmin) & (r_kpc[0].value < xmax))
    e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
    plt.plot(r_kpc[0][e_ind[1:-1]], (press_prof[0].eval())[e_ind[1:-1]])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('R / R500')
    plt.ylabel('Pressure (keV cm$^{-3}$)')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)    
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def spaghetti_press(r_kpc, press_prof, clus, xmin=np.nan, xmax=np.nan, nl=50, ci=95, univpress=None, rbins=None, stef=None, plotdir='./'):
    '''
    Plot the radial pressure profiles
    ---------------------------------
    r_kpc = radius (kpc)
    press_prof = best fitting pressure profile (median and interval)
    xmin, xmax = x-axis boundaries for the plot (by default, they are obtained based on r_kpc)
    ci = uncertainty level of the interval
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'spaghetti_press.pdf')
    for i in range(len(press_prof)):
        plt.clf()
        plt.title(clus[i])
        xmin, xmax = np.nanmax([r_kpc[i][0].value, xmin]), np.nanmin([r_kpc[i][-1].value, xmax])
        ind = np.where((r_kpc[i].value > xmin) & (r_kpc[i].value < xmax))
        e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
        [plt.plot(r_kpc[i][e_ind], p[e_ind], color='grey', linewidth=.4) for p in press_prof[i][:nl]]
        if stef is not None:
            plt.plot(r_kpc[i][e_ind], stef[0][e_ind], color='r', label='Stefano')
            plt.fill_between(r_kpc[i][e_ind].value, stef[1][e_ind], stef[2][e_ind], color='orange')        
        plt.ylim(1e-7, 2e-1)
        plt.xscale('log')
        plt.yscale('log')
        if univpress is not None:
            plt.plot(r_kpc[i][e_ind], univpress[i][e_ind])
        if rbins is not None:
            [plt.axvline(r.value, linestyle=':', color='grey') for r in rbins[i]]
        plt.xlabel('Radius ('+str(r_kpc[i].unit)+')')
        plt.ylabel('Pressure (keV cm$^{-3}$)')
        plt.suptitle('%s Radial pressure profiles' % str(nl))
        plt.xlim(xmin, xmax)
        pdf.savefig()
    pdf.close()

def press_compare(r_kpc, p_gnfw, p_plaw, xmin=np.nan, xmax=np.nan, ci=95, univpress=None, rbins=None, stef=None, plotdir='./'):
    plt.clf()
    pdf = PdfPages(plotdir+'press_compare.pdf')
    for i in range(len(p_gnfw)):
        if len(p_gnfw) > 1:
             plt.subplot(221+i%4)
        xmin, xmax = np.nanmax([r_kpc[i][0].value, xmin]), np.nanmin([r_kpc[i][-1].value, xmax])
        ind = np.where((r_kpc[i].value > xmin) & (r_kpc[i].value < xmax))
        e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
        plt.plot(r_kpc[i][e_ind], p_plaw[0][1][e_ind], color='b', label='p-law')
        plt.fill_between(r_kpc[i][e_ind].value, p_plaw[0][0][e_ind], p_plaw[0][2][e_ind], color='powderblue')
        plt.plot(r_kpc[i][e_ind], p_gnfw[0][1][e_ind], color='r', label='gnfw')
        plt.fill_between(r_kpc[i][e_ind].value, p_gnfw[0][0][e_ind], p_gnfw[0][2][e_ind], color='orange', alpha=.3)        
        plt.ylim(np.max([1e-15, min(np.concatenate([np.array(p_plaw).flatten(), np.array(p_gnfw).flatten()]))]), 1e0)#-1)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='lower left')
        if univpress is not None:
            plt.plot(r_kpc[i][e_ind], univpress[i][e_ind])
        if rbins is not None:
            [plt.axvline(r.to(r_kpc[i].unit).value, linestyle=':', color='grey') for r in rbins[i]]
        if i%2 == 0:
            plt.xlabel('Radius ('+str(r_kpc[i].unit)+')')
        if i%2 == 0:
            plt.ylabel('Pressure (keV cm$^{-3}$)')
        if i == 0:
            plt.title('Radial press. prof (median+%i%% CI)' % ci)
        plt.xlim(xmin, xmax)
        if (i+1)%4 == 0:
            pdf.savefig(bbox_inches='tight')
            plt.clf()
    pdf.savefig(bbox_inches='tight')
    pdf.close()
