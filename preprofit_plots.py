import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from astropy import units as u
import corner
import arviz as az

plt.style.use('classic')
font = {'size': 10}
plt.rc('font', **font)

def tf_diagnostic_plot(w_tf_1d, tf_1d, freq_2d, tf_2d, plotdir='./'):
    pdf = PdfPages('./%s/tf_diagnostics.pdf' % plotdir)
    plt.plot(w_tf_1d.to(1/u.arcmin), tf_1d, 'd', label='input')
    plt.plot(freq_2d[0,:freq_2d.shape[0]//2].to(1/u.arcmin), tf_2d[0,:freq_2d.shape[0]//2], '.', label='1d from 2d')
    plt.xlim(-.1, 2); plt.legend(numpoints=1)
    plt.title('Transfer function interpolation at large radii')
    plt.xlabel('Frequency [arcmin$^{-1}$]'); plt.ylabel('Transfer function')
    pdf.savefig()
    pdf.close()

def plot_guess(out_prof, sz, press, plotdir='./'):
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
        plt.plot(sz.radius[sz.sep:], fact*out_prof[i][0], color='r', label='Starting guess')
        plt.errorbar(sz.flux_data[i][0].value, fact*sz.flux_data[i][1].value, yerr=fact*sz.flux_data[i][2].value,
                     fmt='o', fillstyle='none', color='black', label='Observed data')
        if hasattr(press, 'knots'):
            [plt.axvline(k, linestyle=':') for k in 
             press.knots[i]*u.kpc.to(u.arcsec, equivalencies=press.eq_kpc_as)[i]]
        if i == 0:
            plt.legend(numpoints=1)
        plt.xlim(0., (sz.flux_data[i][0][-1]+np.diff(sz.flux_data[i][0])[-1]).value)
        if (i%4 > 1) | (len(sz.flux_data)-i==2):
            plt.xlabel('Radius ['+str(sz.flux_data[i][0].unit)+']')
        if i%2 == 0:
            plt.ylabel('Surface brightness ['+str(sz.flux_data[i][1].unit)+
                       ('' if sz.flux_data[i][1].unit else 'x ')+'$10^%i$]' % np.log10(fact) if fact != 1 else '')
        if (i+1)%4 == 0:
            pdf.savefig()
            plt.clf()
    pdf.savefig()
    pdf.close()

def traceplot(trace, prs, nc, nk=5, trans_ped=None, ppp=10, div=None, plotdir='./'):
    '''
    '''
    plt.clf()
    prs_latex = ['${}$'.format(i) for i in prs]
    pdf = PdfPages(plotdir+'traceplot.pdf')
    for i in range(int((len(prs)-.5)//ppp)+1):
        axes = az.plot_trace(trace, var_names=prs[i*ppp:min((i+1)*ppp,len(prs))], divergences=div)
        [axes[_][j].set_title('') for j in [0,1] for _ in range(len(axes))]
        [axes[_][0].set_ylabel(prs_latex[i*ppp:min((i+1)*ppp,len(prs))][_], fontdict={'fontsize':20}) for _ in range(len(axes))]
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
    plt.clf()
    param_latex = ['${}$'.format(i) for i in param_names]
    fig = corner.corner(mat_chain, labels=param_latex, title_kwargs={'fontsize': titsize}, label_kwargs={'fontsize': labsize})
    axes = np.array(fig.axes).reshape((len(param_names), len(param_names)))
    plb, pmed, pub = get_equal_tailed(mat_chain, ci=ci)
    for i in range(len(param_names)):
        l_err, u_err = pmed[i]-plb[i], pub[i]-pmed[i]
        axes[i,i].set_title('%s = $%.2f_{-%.2f}^{+%.2f}$' % (param_latex[i], pmed[i], l_err, u_err), fontdict={'fontsize': titsize})
        if show_lines:
            axes[i,i].axvline(pmed[i], color=col_lines, linestyle='--', label='Median')
            axes[i,i].axvline(plb[i], color=col_lines, linestyle=':', label='%i%% CI' % ci)
            axes[i,i].axvline(pub[i], color=col_lines, linestyle=':', label='_nolegend_')
            for yi in range(len(param_names)):
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
        plt.title(np.atleast_1d(clus)[i])
        lsz, msz, usz = perc_sz[i]*fact
        plt.plot(sz.radius[sz.sep:], msz, color='r', label='Best-fit')
        plt.fill_between(sz.radius[sz.sep:].value, lsz, usz, color='gold', label='%i%% CI' % ci)
        plt.errorbar(sz.flux_data[i][0].value, sz.flux_data[i][1].value*fact, yerr=sz.flux_data[i][2].value*fact, fmt='o', 
                     fillstyle='none', color='black', label='Observed data')
        plt.xlim(0., 50+np.ceil(sz.flux_data[i][0][-1].value))
        if rbins is not None:
            [plt.axvline(r, linestyle=':', color='grey', label='_nolegend_') for r in rbins[i]]
        if peds is not None:
            plt.axhline(peds[i]*fact, linestyle=':', color='grey', label='_nolegend_')
        plt.xlabel('Radius ('+str(sz.flux_data[i][0].unit)+')')
        plt.ylabel('Surface brightness ('+str(sz.flux_data[i][1].unit)+('' if sz.flux_data[i][1].unit else 'x ')+'$10^%i$)' % np.log10(fact) if fact != 1 else '')
        pdf.savefig()
    pdf.close()

def plot_press(r_kpc, press_prof, clus, xmin=np.nan, xmax=np.nan, ci=95, univpress=None, rbins=None, plotdir='./'):
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
        if univpress is not None:
            plt.plot(r_kpc[i][e_ind], univpress[i][e_ind])
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
    plt.clf()
    plt.title('Outer slope - Posterior distribution')
    low, med, upp = get_equal_tailed(slopes, ci=ci)
    plt.hist(slopes, density=True, histtype='step', color='black')
    plt.axvline(med, color='black', linestyle='--', label='Median')
    plt.axvline(low, color='black', linestyle='-.', label='%i%% CI' % ci)
    plt.axvline(upp, color='black', linestyle='-.', label='_nolegend_')
    plt.xlabel('Outer slope')
    plt.ylabel('Density')
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def spaghetti_press(r_kpc, press_prof, clus, xmin=np.nan, xmax=np.nan, nl=50, ci=95, univpress=None, rbins=None, plotdir='./'):
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
        plt.title(np.atleast_1d(clus)[i])
        xmin, xmax = np.nanmax([r_kpc[i][0].value, xmin]), np.nanmin([r_kpc[i][-1].value, xmax])
        ind = np.where((r_kpc[i].value > xmin) & (r_kpc[i].value < xmax))
        e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
        [plt.plot(r_kpc[i][e_ind], p[e_ind], color='grey', linewidth=.4) for p in press_prof[i][:nl]]
        plt.ylim(1e-7, 2e-1)
        plt.xscale('log')
        plt.yscale('log')
        if univpress is not None:
            plt.plot(r_kpc[i][e_ind], univpress[i][e_ind])
        if rbins is not None:
            [plt.axvline(r, linestyle=':', color='grey') for r in rbins[i]]
        plt.xlabel('Radius ('+str(r_kpc[i].unit)+')')
        plt.ylabel('Pressure (keV cm$^{-3}$)')
        plt.suptitle('%s Radial pressure profiles' % str(nl))
        plt.xlim(xmin, xmax)
        pdf.savefig()
    pdf.close()
