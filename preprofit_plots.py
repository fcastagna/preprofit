import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import corner

font = {'size': 8}
plt.rc('font', **font)
plt.style.use('classic')

def traceplot(mysamples, param_names, nsteps, nw, plotw=20, ppp=4, plotdir='./'):
    '''
    Traceplot of the MCMC
    ---------------------
    mysamples = array of sampled values in the chain
    param_names = names of the parameters
    nsteps = number of steps in the chain (after burn-in) 
    nw = number of random walkers
    plotw = number of random walkers that we wanna plot (default is 20)
    ppp = number of plots per page
    plotdir = directory where to place the plot
    '''
    nw_step = int(np.ceil(nw/plotw))
    param_latex = ['${}$'.format(i) for i in param_names]
    pdf = PdfPages(plotdir+'traceplot.pdf')
    plt.figure().suptitle('Traceplot')
    for i in np.arange(mysamples.shape[1]):
        plt.subplot(ppp, 1, i%ppp+1)
        for j in range(nw)[::nw_step]:
            plt.plot(np.arange(nsteps)+1, mysamples[j::nw,i], linewidth=.2)
        plt.ylabel('%s' %param_latex[i])
        if (abs((i+1)%ppp) < 0.01):
            plt.xlabel('Iteration number')
            pdf.savefig()
            if i+1 < mysamples.shape[1]:
                plt.clf()
        elif i+1 == mysamples.shape[1]:
            plt.xlabel('Iteration number')
            pdf.savefig()
    pdf.close()

def triangle(mysamples, param_names, plotdir='./'):
    '''
    Univariate and multivariate distribution of the parameters in the MCMC
    ----------------------------------------------------------------------
    mysamples = array of sampled values in the chain
    param_names = names of the parameters
    plotdir = directory where to place the plot
    '''
    param_latex = ['${}$'.format(i) for i in param_names]
    plt.clf()
    pdf = PdfPages(plotdir+'cornerplot.pdf')
    corner.corner(mysamples, labels=param_latex, quantiles=np.repeat(.5, len(param_latex)), show_titles=True)
    pdf.savefig()
    pdf.close()
    
def plot_best(theta, fit_pars, mp_med, mp_lb, mp_ub, radius, sep, flux_data, ci, plotdir='./'):
    '''
    Plot of the Compton parameter profile compared to the flux density data
    -----------------------------------------------------------------------
    mp_med = Compton profile for the median parameter values
    mp_lb, mp_ub = CI of the Compton profile
    plotdir = directory where to place the plot
    '''
    r_sec, y_data, err = flux_data
    plt.clf()
    pdf = PdfPages(plotdir+'best_fit.pdf')
    plt.plot(radius[sep:sep+mp_med.size], mp_med)
    plt.fill_between(radius[sep:sep+mp_med.size], mp_lb, mp_ub, color='powderblue', label='_nolegend_')
    plt.errorbar(r_sec, y_data, yerr=err, fmt='.', color='r')
    plt.legend(('Filtered profile with %s%% CI' %ci, 'Flux density'), loc='lower right')
    plt.axhline(y=0, color='black', linestyle=':')
    plt.xlabel('Radius (arcsec)')
    plt.ylabel('Flux (mJy/beam)')
    plt.xlim(0, np.ceil(r_sec[-1]/60)*60*7/6)
    plt.title('Flux density profile: best-fit with %s%% CI' %ci)
    pdf.savefig()
    pdf.close()
