import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from astropy import units as u
import corner
import arviz as az

plt.style.use('classic')
font = {'size': 10}
plt.rc('font', **font)

def ticks_size(fsize, axes=None):
    try:
        [a.tick_params(labelsize=fsize) for a in axes]
    except:
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)

def tf_diagnostic_plot(w_tf_1d, tf_1d, freq_2d, tf_2d, fsize=13, plotdir='./'):
    pdf = PdfPages('./%s/tf_diagnostics.pdf' % plotdir)
    plt.plot(w_tf_1d.to(1/u.arcmin), tf_1d, 'd', label='input')
    plt.plot(freq_2d[0,:freq_2d.shape[0]//2].to(1/u.arcmin), tf_2d[0,:freq_2d.shape[0]//2], '.', label='1d from 2d')
    plt.xlim(-.1, 2); plt.legend(numpoints=1)
    plt.title('Transfer function interpolation at large radii', fontsize=fsize)
    plt.xlabel('Frequency [arcmin$^{-1}$]', fontsize=fsize); plt.ylabel('Transfer function', fontsize=fsize)
    ticks_size(fsize)
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def plot_guess(out_prof, sz, press, fact=1, plotdir='./'):
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
        plt.title(sz.clus[i])
        plt.plot(sz.radius[sz.sep:], fact*out_prof[i], color='r', label='Starting guess')
        plt.errorbar(sz.flux_data[i][0].value, fact*sz.flux_data[i][1].value, yerr=fact*sz.flux_data[i][2].value,
                     fmt='o', fillstyle='none', color='black', label='Observed data')
        if hasattr(press, 'knots'):
            [plt.axvline(k, linestyle=':') for k in 
             press.knots[i]*u.kpc.to(u.arcsec, equivalencies=press.eq_kpc_as)[i]]
        if i == 0:
            plt.legend(numpoints=1)
        plt.xlim(0., (sz.flux_data[i][0][-1]+np.diff(sz.flux_data[i][0])[-1]).value)
        if (i%4 > 1) | (len(sz.flux_data)-i < 3):
            plt.xlabel('Radius ['+str(sz.flux_data[i][0].unit)+']')
        if i%2 == 0:
            plt.ylabel('Surface brightness ['+str(sz.flux_data[i][1].unit)+
                       ('' if sz.flux_data[i][1].unit == '' else '] [')+('$10^{%i}$]' % np.log10(fact) if fact != 1 else ''))
        if (i+1)%4 == 0:
            pdf.savefig(bbox_inches='tight')
            plt.clf()
        elif i == len(sz.flux_data)-1:
            pdf.savefig(bbox_inches='tight')
    pdf.close()

def traceplot(trace, prs, prs_ext, fact_ped=1, compact=False, ppp=5, legend=True, div=None, fsize=13, plotdir='./'):
    '''
    '''
    plt.clf()
    prs_latex = ['$%s%s$' % ('\\' if p[:3]=='sig' else '', p) if compact else ['$%s%s$' % ('\\' if pj[:3]=='sig' else '', pj) for pj in p] for _, p in enumerate(prs if compact else prs_ext)]
    prs_latex[-1] = (prs_latex[-1]+' [10$^{-%s}$]' % int(np.log10(fact_ped)) if fact_ped != 1 else '') if compact else [p+' [10$^{%s}$]' % int(np.log10(fact_ped)) if fact_ped != 1 else '' for p in prs_latex[-1]]
    trace.posterior['peds'] *= fact_ped
    pdf = PdfPages(plotdir+'traceplot.pdf')
    for j, p in enumerate(prs):#[:np.where(np.array(prs)=='lgP_{0,i}')[0][0]]):
        for i in range(int((len(prs_ext[j])-.5)//ppp)+1):
            plt.clf()
            axes = az.plot_trace(
                trace, var_names=prs if compact else p, 
                coords={} if compact else {p+'_dim_0': np.arange(i*ppp, np.min([(i+1)*ppp, trace.posterior[p].shape[-1]]))}, 
                divergences=div, compact=compact, figsize=(20,10) if len(prs_ext[j])>1 else None)
            nr = np.min([trace.posterior[p].shape[-1], ppp])
            nrows = len(axes) if compact else trace.posterior[p].shape[-1] // ppp if trace.posterior[p].shape[-1]-i*ppp < nr else nr
            [axes[_][j].set_title('') for j in [0,1] for _ in range(nrows)]
            [axes[_][0].set_ylabel(prs_latex[_] if compact else prs_latex[j][i*ppp+_], fontdict={'fontsize': fsize}) for _ in range(nrows)]
            axes[-1][0].set_xlabel('Value', fontsize=fsize)
            axes[-1][1].set_xlabel('Iteration', fontsize=fsize)
            if compact & legend:
                for _ in [0,2]:
                    axes[_][0].legend(
                        np.array([['$%s$=%s' % ('k' if _==0 else 'i', j)]+(
                            2*trace.posterior[prs[_]].shape[0]-1)*['_nolabel_'] for j in 
                            range(trace.posterior[prs[_]].shape[-1])]).flatten(), 
                        fontsize='small', ncols=np.ceil(len(prs_ext[_])/5))
            [ticks_size(fsize, a) for a in axes]
            [axes[_][1].set_ylabel(prs_latex[_] if compact else prs_latex[j][i*ppp+_], fontdict={'fontsize': fsize}) for _ in range(nrows)]
            pdf.savefig(bbox_inches='tight')
            plt.close()
            if compact: break
        if compact: break
    pdf.close()
    trace.posterior['peds'] /= fact_ped

def fitwithmod(sz, perc_sz, eq_kpc_as, rbins=None, peds=None, ind_fits=None, fact=1, ci=95, fsize=13, plotdir='./'):
    '''
    Surface brightness profile (points with error bars) and best fitting profile with uncertainties
    -----------------------------------------------------------------------------------------------
    sz = class of SZ data
    ci = uncertainty level of the interval
    perc_sz = best (median) SZ fitting profiles with uncertainties
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'fit_on_data.pdf')
    for i in range(len(sz.flux_data)):
        plt.clf()
        plt.title(np.atleast_1d(sz.clus)[i], fontsize=fsize)
        lsz, msz, usz = perc_sz[i]*fact
        plt.plot(sz.radius[sz.sep:], msz, color='b', label='Posterior median')
        plt.fill_between(sz.radius[sz.sep:].value, lsz, usz, color='powderblue', label='%i%% uncertainty' % ci)
        plt.errorbar(sz.flux_data[i][0].value, sz.flux_data[i][1].value*fact, yerr=sz.flux_data[i][2].value*fact, fmt='o', fillstyle='none', color='black', capsize=0, label='Observed data')
        plt.xlim(0., 50+np.ceil(sz.flux_data[i][0][-1].value))
        if rbins is not None:
            [plt.axvline(r, linestyle=':', color='grey', label='_nolegend_') for r in rbins[i]]
        if peds is not None:
            plt.axhline(peds[i]*fact, linestyle=':', color='grey', label='_nolegend_')
        plt.xlabel('Radius ['+str(sz.flux_data[i][0].unit)+']', fontsize=fsize+2)
        plt.ylabel('Surface brightness $[$'+str(sz.flux_data[i][1].unit)+
                   ('' if sz.flux_data[i][1].unit == '' else '$] [$')+('$10^{%i}]$' % np.log10(fact) if fact != 1 else ''), fontsize=fsize+2)
        ticks_size(fsize)
        if ind_fits is not None:
            plt.plot(sz.radius[sz.sep:], ind_fits[i]*fact, c='r', linestyle='--', label='Individual analysis')
        pdf.savefig(bbox_inches='tight')
    pdf.close()

def get_equal_tailed(data, ci=68, axis=0):
    '''
    Computes the median and lower/upper limits of the equal tailed uncertainty interval
    -----------------------------------------------------------------------------------
    ci = uncertainty level of the interval
    ----------------------------------------
    RETURN: lower bound, median, upper bound
    '''
    low, med, upp = map(np.atleast_1d, np.percentile(data, [50-ci/2, 50, 50+ci/2], axis=axis))
    return np.array([low, med, upp])

def triangle(mat_chain, param_names, model, tit=None, fact_ped=1, show_lines=True, show_title=True, col_lines='r', ci=95, labsize=25., fsize=14., titsize=15., legend=True, plotdir='./'):
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
    plt.clf()
    param_latex = [['$%s%s$' % ('\\' if pj[:5]=='sigma' else '', pj) for pj in p] for _, p in enumerate(param_names)]
    for _ in range(len(mat_chain)):
        plt.clf()
        fig = corner.corner(mat_chain[_]*(fact_ped if _==len(mat_chain)-1 else 1), 
                            labels=param_latex[_], title_kwargs={'fontsize': titsize}, 
                            label_kwargs={'fontsize': labsize}, hist_kwargs={'density': True})
        if tit is not None: fig.suptitle(tit[_], fontsize=30)
        axes = np.array(fig.axes).reshape((len(param_names[_]), len(param_names[_])))
        plb, pmed, pub = get_equal_tailed(mat_chain[_]*(fact_ped if _==len(mat_chain)-1 else 1), ci=ci)
        for i in range(len(param_names[_])):
            l_err, u_err = pmed[i]-plb[i], pub[i]-pmed[i]
            if show_title:
                axes[i,i].set_title('%s = $%.2f_{-%.2f}^{+%.2f}$' % (param_latex[_][i], pmed[i], l_err, u_err), fontdict={'fontsize': titsize})
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
        [ticks_size(fsize, a) for a in axes]
        pdf.savefig(bbox_inches='tight')
        plt.close()
    pdf.close()

def plot_press(r_kpc, press_prof, clus, xmin=np.nan, xmax=np.nan, ci=95, rbins=None, plotdir='./'):
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
        plt.title(np.atleast_1d(clus)[i])
        l_press, m_press, u_press = press_prof[i]
        xmin, xmax = np.nanmax([r_kpc[i][0].value, xmin]), np.nanmin([r_kpc[i][-1].value, xmax])
        ind = np.where((r_kpc[i].value > xmin) & (r_kpc[i].value < xmax))
        e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
        plt.plot(r_kpc[i][e_ind], m_press[e_ind])
        plt.fill_between(r_kpc[i][e_ind].value, l_press[e_ind], u_press[e_ind], color='powderblue', label='_nolegend_')
        if rbins is not None:
            [plt.axvline(r, linestyle=':', color='grey', label='_nolegend_') for r in rbins[i]]
        plt.xscale('log')
        plt.yscale('log')
        # plt.ylim(1e-5, 1e-1)
        plt.xlabel('Radius ['+str(r_kpc[i].unit)+']')
        plt.ylabel('Pressure [keV$/$cm$^{3}$]')
        plt.xlim(xmin, xmax)
        pdf.savefig(bbox_inches='tight')
    pdf.close()

def hist_slopes(slopes, clus, ci=95, plotdir='./'):
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
        plt.title('%s' % clus[_])
        low, med, upp = get_equal_tailed(slopes[_], ci=ci)
        plt.hist(slopes[_], density=True, histtype='step', color='black')
        plt.axvline(med, color='black', linestyle='--', label='Median')
        plt.axvline(low, color='black', linestyle='-.', label='%i%% CI' % ci)
        plt.axvline(upp, color='black', linestyle='-.', label='_nolegend_')
        plt.xlabel('Outer slope')
        plt.ylabel('Density')
        pdf.savefig(bbox_inches='tight')
    pdf.close()