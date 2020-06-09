import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import corner

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
            if i+1 < cube_chain.shape[1]:
                plt.clf()
        elif i+1 == cube_chain.shape[1]:
            plt.tick_params(labelbottom=True)
            plt.xlabel('Iteration number')
            pdf.savefig(bbox_inches='tight')
    pdf.close()

def triangle(mat_chain, param_names, labsize=25., titsize=15., plotdir='./'):
    '''
    Univariate and multivariate distribution of the parameters in the MCMC
    ----------------------------------------------------------------------
    mat_chain = 2d array of sampled values ((nw x niter) x nparam)
    param_names = names of the parameters
    labsize = label font size
    titsize = titles font size
    plotdir = directory where to place the plot
    '''
    param_latex = ['${}$'.format(i) for i in param_names]
    plt.clf()
    pdf = PdfPages(plotdir+'cornerplot.pdf')
    corner.corner(mat_chain, labels=param_latex, quantiles=np.repeat(.5, len(param_latex)), show_titles=True, 
                  title_kwargs={'fontsize': titsize}, label_kwargs={'fontsize': labsize})
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
    low, med, upp = np.percentile(data, [50-ci/2, 50, 50+ci/2], axis=0)
    return np.array([low, med, upp])

def best_fit_prof(cube_chain, log_lik, press, sz, num='all', seed=None, ci=95):
    '''
    Compute the surface brightness profile (median and uncertainty interval) for the best fitting parameters
    --------------------------------------------------------------------------------------------------------
    cube_chain = 3d array of sampled values (nw x niter x nparam)
    num = number of set of parameters to include (default is 'all', i.e. nw x niter parameters)
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
    profs_sz = []
    for j in rand:
        out_prof = log_lik(cube_chain[w[j],it[j],:], press.pars, press, sz, output='bright')
        profs_sz.append(out_prof)
    perc_sz = get_equal_tailed(profs_sz, ci)
    return perc_sz

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
    plt.errorbar(sz.flux_data[0], sz.flux_data[1], yerr=sz.flux_data[2], fmt='o', fillstyle='none', color='black', 
                 label='Observed data')
    plt.xlabel('Radius (arcsec)')
    plt.ylabel('Surface brightness (mJy·beam$^{-1}$)')
    plt.xlim(0., np.ceil(sz.flux_data[0][-1]))
    plt.legend(loc='lower right')
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def press_prof(cube_chain, log_lik, press, sz, num='all', seed=None, ci=95):
    '''
    Radial pressure profile (median and uncertainty interval) from given number of samples
    --------------------------------------------------------------------------------------
    cube_chain = 3d array of sampled values (nw x niter x nparam)
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
        press_prof.append(log_lik(cube_chain[w[j],it[j],:], press.pars, press, sz, output='pp'))
    perc_press = get_equal_tailed(press_prof, ci)
    return perc_press

def plot_press(r_kpc, press, xmin=np.nan, xmax=np.nan, ci=95, plotdir='./'):
    '''
    Plot the radial pressure profiles
    ---------------------------------
    r_kpc = radius (kpc)
    press = best fitting pressure profile (median and interval)
    xmin, xmax = x-axis boundaries for the plot (by default, they are obtained based on r_kpc)
    ci = uncertainty level of the interval
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'press_fit.pdf')
    plt.clf()
    xmin, xmax = np.nanmax([r_kpc[0], xmin]), np.nanmin([r_kpc[-1], xmax])
    ind = np.where((r_kpc > xmin) & (r_kpc < xmax))
    e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
    plt.plot(r_kpc[e_ind], press[1][e_ind])
    plt.fill_between(r_kpc[e_ind], press[0][e_ind], press[2][e_ind], color='powderblue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Pressure (keV cm$^{-3}$)')
    plt.title('Radial pressure profile (median with %i%% CI)' % ci)
    plt.xlim(xmin, xmax)
    pdf.savefig(bbox_inches='tight')
    pdf.close()
