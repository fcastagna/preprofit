import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import corner

plt.style.use('classic')
font = {'size': 20}
plt.rc('font', **font)

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
    for i in np.arange(mysamples.shape[1]):
        plt.subplot(ppp, 1, i%ppp+1)
        for j in range(nw)[::nw_step]:
            plt.plot(np.arange(nsteps)+1, mysamples[j::nw,i], linewidth=.2)
            plt.tick_params(labelbottom=False)
        plt.ylabel('%s' %param_latex[i], fontdict={'fontsize': 20})
        if (abs((i+1)%ppp) < 0.01):
            plt.tick_params(labelbottom=True)
            plt.xlabel('Iteration number')
            pdf.savefig()
            if i+1 < mysamples.shape[1]:
                plt.clf()
        elif i+1 == mysamples.shape[1]:
            plt.tick_params(labelbottom=True)
            plt.xlabel('Iteration number')
            pdf.savefig()
    pdf.close()

def triangle(mysamples, param_names, plotdir='./', filename='cornerplot.pdf'):
    '''
    Univariate and multivariate distribution of the parameters in the MCMC
    ----------------------------------------------------------------------
    mysamples = array of sampled values in the chain
    param_names = names of the parameters
    plotdir = directory where to place the plot
    '''
    param_latex = ['${}$'.format(i) for i in param_names]
    plt.clf()
    #pdf = PdfPages(plotdir+'cornerplot.pdf')
    pdf = PdfPages(plotdir+filename)
    corner.corner(mysamples, labels=param_latex, quantiles=np.repeat(.5, len(param_latex)), show_titles=False, 
                 label_kwargs={'fontsize': 30})  #,title_kwargs={'fontsize': 20}
    pdf.savefig()
    pdf.close()
    
def plot_best(theta, fit_pars, mp_med, mp_lb, mp_ub, radius, sep, flux_data, ci=95, plotdir='./'):
    '''
    Surface brightness profile (points with error bars) and best fitting profile with CI
    ------------------------------------------------------------------------------------
    mp_med = best (median) fitting profile
    mp_lb, mp_ub = CI boundaries
    ci = confidence interval level
    plotdir = directory where to place the plot
    '''
    r_sec, y_data, err = flux_data
    plt.clf()
    #print(plt.rcParams.get('figure.figsize'))
    #matplotlib.use('TkAgg')
    #print(plt.rcParams.get('backend'))
    #plt.rcParams.update(plt.rcParamsDefault)
    #plt.style.use('classic')
    #print(plt.rcParams.get('backend'))
    #print(plt.rcParams)
    #plt.rcParams.update({'font.size': 20})   
    plt.rcParams['errorbar.capsize'] = 0
    pdf = PdfPages(plotdir+'fit_on_data.pdf')
    plt.plot(radius[sep:sep+mp_med.size], mp_med*1e6)
    plt.fill_between(radius[sep:sep+mp_med.size], mp_lb*1e6, mp_ub*1e6, color='powderblue', label='_nolegend_')
    plt.errorbar(r_sec, y_data*1e6, yerr=err*1e6, fmt='o', fillstyle='none', color='r')
    plt.legend(('A10 model (%i%% CI)' %ci, 'Observed data'), loc='lower right')
    plt.legend(('Model (%i%% CI)' %ci, 'Observed data'), loc='lower right')
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Surface brightness [$\mu$K]')
    #plt.xlim(0, np.ceil(r_sec[-1]/60)*60*7/6)
    plt.xlim(0, 2*60*7/6)
    #plt.xlim(0, 180)
    plt.ylim(-400,50)
    #plt.ylim(-120,14)
    pdf.savefig()
    pdf.close()


def plot_press(rkpc, med, low, hi,ci,plotdir='./'):
    plt.clf()
    pdf = PdfPages(plotdir+'press_fit.pdf')
    plt.plot(rkpc, med)
    plt.xscale('log')
    plt.yscale('log')   
    plt.fill_between(rkpc, low, hi, color='powderblue')
    #plt.legend(('Model (%i%% CI)' %ci), loc='lower right')
    plt.xlabel('Radius [kpc]')
    plt.ylabel('keV/cm3')
    #plt.xlim(0, np.ceil(rkpc[-1]*7/6))
    plt.xlim(50, 1000)
    plt.ylim(1e-5,3e-1)
    pdf.savefig()
    pdf.close()


def plot_guess(theta, fit_pars, mp_med, radius, sep, flux_data, plotdir='./'):
    '''
    Surface brightness profile (points with error bars) and guess profile 
    ------------------------------------------------------------------------------------
    mp_med = first guess profile
    plotdir = directory where to place the plot
    '''
    #print(plt.rcParams.get('figure.figsize'))
    #print(plt.rcParams)
    r_sec, y_data, err = flux_data
    plt.clf()
    pdf = PdfPages(plotdir+'starting_guess.pdf')
    plt.plot(radius[sep:sep+mp_med.size], 1e6*mp_med)
    plt.errorbar(r_sec, 1e6*y_data, yerr=1e6*err, fmt='o', fillstyle='none', color='r')
    plt.legend(('Guessed Model', 'Observed data'), loc='lower right')
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Surface brightness [muK]')
    plt.xlim(0, np.ceil(r_sec[-1]/60)*60*7/6)
    pdf.savefig()
    pdf.close()
