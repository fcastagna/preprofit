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
    plt.clf()
    pdf = PdfPages(plotdir+'starting_guess.pdf')
    plt.plot(sz.radius[sz.sep:], out_prof, color='r', label='Best-fit')
    plt.errorbar(sz.flux_data[0].value, sz.flux_data[1].value, yerr=sz.flux_data[2].value, fmt='o', fillstyle='none', color='black', label='Observed data')
    plt.legend()
    plt.xlabel('Radius ('+str(sz.flux_data[0].unit)+')')
    plt.ylabel('Surface brightness ('+str(sz.flux_data[1].unit)+')')
    plt.xlim(0., (sz.flux_data[0][-1]+np.diff(sz.flux_data[0])[-1]).value)
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
    lsz, msz, usz = perc_sz
    plt.plot(sz.radius[sz.sep:], msz, color='r', label='Best-fit')
    plt.fill_between(sz.radius[sz.sep:], lsz, usz, color='gold', label='%i%% CI' % ci)
    plt.errorbar(sz.flux_data[0].value, sz.flux_data[1].value, yerr=sz.flux_data[2].value, fmt='o', fillstyle='none', color='black', label='Observed data')
    plt.xlabel('Radius ('+str(sz.flux_data[0].unit)+')')
    plt.ylabel('Surface brightness ('+str(sz.flux_data[1].unit)+')')
    plt.xlim(0., np.ceil(sz.flux_data[0][-1].value))
    plt.legend()
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def press_prof(cube_chain, press, r_kpc, num='all', seed=None, ci=95):
    '''
    Radial pressure profile (median and uncertainty interval) from given number of samples
    --------------------------------------------------------------------------------------
    cube_chain = 3d array of sampled values (nw x niter x nparam)
    press = pressure object of the class Pressure
    r_kpc = radius (kpc)
    num = number of samples to include (default is 'all', i.e. nw x niter parameters)
    seed = random seed (default is None)
    ci = uncertainty level of the interval
    ------------------------------------------------
    RETURN: median and uncertainty interval profiles
    '''
    nw = cube_chain.shape[0]
    if num == 'all':
        num = nw*cube_chain.shape[1]
    w, it = np.meshgrid(np.arange(nw), np.arange(cube_chain.shape[1]))
    w = w.flatten()
    it = it.flatten()
    np.random.seed(seed)
    rand = np.random.choice(w.size, num, replace=False)
    press_prof = []
    for j in rand:
        press.update_vals(press.fit_pars, cube_chain[w[j],it[j],:])
        press_prof.append(press.press_fun(r_kpc))
    perc_press = get_equal_tailed(press_prof, ci)
    return perc_press

def plot_press(r_kpc, press_prof, xmin=np.nan, xmax=np.nan, ci=95, plotdir='./'):
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
    l_press, m_press, u_press = press_prof
    xmin, xmax = np.nanmax([r_kpc[0].value, xmin]), np.nanmin([r_kpc[-1].value, xmax])
    ind = np.where((r_kpc.value > xmin) & (r_kpc.value < xmax))
    e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
    plt.plot(r_kpc[e_ind], m_press[e_ind])
    plt.fill_between(r_kpc[e_ind], l_press[e_ind], u_press[e_ind], color='powderblue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Radius ('+str(r_kpc.unit)+')')
    plt.ylabel('Pressure (keV cm$^{-3}$)')
    plt.title('Radial pressure profile (median with %i%% CI)' % ci)
    plt.xlim(xmin, xmax)
    pdf.savefig(bbox_inches='tight')
    pdf.close()
