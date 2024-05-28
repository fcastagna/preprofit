import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import corner

plt.style.use('classic')
font = {'size': 20}
plt.rc('font', **font)

def plot_guess(out_prof, sz,  plotdir='./'):
    '''
    Modeled profile resulting from starting parameters VS observed data
    -------------------------------------------------------------------
    out_prof = modeled profile
    sz = class of SZ data
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'starting_guess.pdf')
    for i in range(len(sz.flux_data)):
        plt.subplot(221+i%4)
        plt.plot(sz.radius[sz.sep:], out_prof[i][0], color='r', label='Starting guess')
        plt.errorbar(sz.flux_data[i][0].value, sz.flux_data[i][1].value, yerr=sz.flux_data[i][2].value, 
                     fmt='o', fillstyle='none', color='black', label='Observed data')
        if i == 0:
            plt.legend()
        plt.ylim(np.min([fl[1].value for fl in sz.flux_data]), np.max([fl[1].value for fl in sz.flux_data]))
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

def triangle(mat_chain, param_names, show_lines=True, col_lines='r', ci=95, labsize=25., titsize=15., plotdir='./'):
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
    plt.clf()
    pdf = PdfPages(plotdir+'cornerplot.pdf')
    param_latex = ['${}$'.format(i) for i in param_names]
    fig = corner.corner(mat_chain, labels=param_latex, title_kwargs={'fontsize': titsize}, label_kwargs={'fontsize': labsize})
    axes = np.array(fig.axes).reshape((len(param_names), len(param_names)))
    plb, pmed, pub = get_equal_tailed(mat_chain, ci=ci)
    for i in range(len(param_names)):
        l_err, u_err = pmed[i]-plb[i], pub[i]-pmed[i]
        axes[i,i].set_title('%s = $%.2f_{-%.2f}^{+%.2f}$' % (param_latex[i], pmed[i], l_err, u_err))
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
            fig.legend(('Median', '%i%% CI' % ci), loc='lower center', ncol=2, bbox_to_anchor=(0.55, 0.95), fontsize=titsize+len(param_names))
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

def fitwithmod(sz, perc_sz, ci=95, plotdir='./'):
    '''
    Surface brightness profile (points with error bars) and best fitting profile with uncertainties
    -----------------------------------------------------------------------------------------------
    sz = class of SZ data
    perc_sz = best (median) SZ fitting profiles with uncertainties
    ci = uncertainty level of the interval
    plotdir = directory where to place the plot
    '''
    plt.clf()
    pdf = PdfPages(plotdir+'fit_on_data.pdf')
    for i in range(len(sz.flux_data)):
        plt.subplot(221+i%4)
        lsz, msz, usz = perc_sz[i]
        plt.plot(sz.radius[sz.sep:], msz, color='r', label='Best-fit')
        plt.fill_between(sz.radius[sz.sep:].value, lsz, usz, color='gold', label='%i%% CI' % ci)
        plt.errorbar(sz.flux_data[i][0].value, sz.flux_data[i][1].value, yerr=sz.flux_data[i][2].value, fmt='o', fillstyle='none', color='black', label='Observed data')
        plt.xlim(0., 50+np.ceil(sz.flux_data[i][0][-1].value))
        if i == 0:
            plt.legend()
        if i%4 > 1:
            plt.xlabel('Radius ('+str(sz.flux_data[i][0].unit)+')')
        if i%2 == 0:
            plt.ylabel('Surface brightness ('+str(sz.flux_data[i][1].unit)+')')
        if (i+1)%4 == 0:
            pdf.savefig()
            plt.clf()
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def plot_press(r_kpc, press_prof, xmin=np.nan, xmax=np.nan, univpress=None, ci=95, plotdir='./'):
    '''
    Plot the radial pressure profiles
    ---------------------------------
    r_kpc = radius (kpc)
    press_prof = best fitting pressure profile (median and interval)
    xmin, xmax = x-axis boundaries for the plot (by default, they are obtained based on r_kpc)
    ci = uncertainty level of the interval
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'press_fit.pdf')
    plt.clf()
    for i in range(len(press_prof)):
        plt.subplot(221+i%4)
        l_press, m_press, u_press = press_prof[i]
        xmin, xmax = np.nanmax([r_kpc[i][0].value, xmin]), np.nanmin([r_kpc[i][-1].value, xmax])
        ind = np.where((r_kpc[i].value > xmin) & (r_kpc[i].value < xmax))
        e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
        plt.plot(r_kpc[i][e_ind], m_press[e_ind])
        plt.fill_between(r_kpc[i][e_ind].value, l_press[e_ind], u_press[e_ind], color='powderblue')
        plt.xscale('log')
        plt.yscale('log')
        if univpress is not None:
            plt.plot(r_kpc[i][e_ind], univpress[i][e_ind])
        if i%4 > 1:
            plt.xlabel('Radius ('+str(r_kpc[i].unit)+')')
        if i%2 == 0:
            plt.ylabel('Pressure (keV cm$^{-3}$)')
        if i == 0:
            plt.title('Radial pressure profile (median with %i%% CI)' % ci)
            if univpress is not None:
                plt.legend(('fitted', 'universal'))
        plt.xlim(xmin, xmax)
        if (i+1)%4 == 0:
            pdf.savefig()
            plt.clf()
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def hist_slopes(slopes, ci=95, plotdir='./'):
    '''
    Plot the histogram of the outer slopes posterior distribution
    -------------------------------------------------------------
    slopes = array of slopes
    ci = uncertainty level of the interval
    plotdir = directory where to place the plot
    '''
    low, med, upp = get_equal_tailed(slopes, ci=ci)
    pdf = PdfPages(plotdir+'outer_slopes.pdf')
    plt.clf()
    plt.title('Outer slope - Posterior distribution')
    plt.hist(slopes, density=True, histtype='step', color='black')
    plt.axvline(med, color='black', linestyle='--', label='Median')
    plt.axvline(low, color='black', linestyle='-.', label='%i%% CI' % ci)
    plt.axvline(upp, color='black', linestyle='-.', label='_nolegend_')
    plt.xlabel('Outer slope')
    plt.ylabel('Density')
    pdf.savefig(bbox_inches='tight')
    pdf.close()
