import numpy as np
from astropy.io import fits
from scipy.stats import norm
from scipy.interpolate import interp1d
from abel.direct import direct_transform
from scipy import optimize
from scipy.integrate import simps
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2
from scipy.optimize import minimize
import time
import h5py

class Param:
    '''
    Class for parameters
    --------------------
    val = value of the parameter
    minval, maxval = minimum and maximum allowed values
    frozen = whether the parameter is allowed to vary (True/False)
    '''
    def __init__(self, val, minval=-1e99, maxval=1e99, frozen=False):
        self.val = float(val)
        self.minval = minval       
        self.maxval = maxval
        self.frozen = frozen

    def __repr__(self):
        return '<Param: val=%.3g, minval=%.3g, maxval=%.3g, frozen=%s>' % (
            self.val, self.minval, self.maxval, self.frozen)

    def prior(self):
        if self.val < self.minval or self.val > self.maxval:
            return -np.inf
        return 0.

class Pressure:
    '''
    Class to parametrize the pressure profile
    -----------------------------------------    
    '''
    def __init__(self):
        self.pars = self.defPars()
        
    def defPars(self):
        '''
        Default parameter values
        ------------------------
        P_0 = normalizing constant
        a = rate of turnover between b and c
        b = logarithmic slope at r/r_p >> 1
        c = logarithmic slope at r/r_p << 1
        r_p = characteristic radius
        '''
        pars = {
            'P_0': Param(0.4, minval=0., maxval=20.),
            'a': Param(1.33, minval=0.1, maxval=10.),
            'b': Param(4.13, minval=0.1, maxval=15.),
            'c': Param(0.014, minval=0., maxval=3.),
            'r_p': Param(300., minval=100., maxval=1000.)
            }
        return pars

    def update_vals(self, fit_pars, pars_val):
        '''
        Update the parameter values
        ---------------------------
        fit_pars = name of the parameters to update
        pars_val = new parameter values
        '''
        for name, i in zip(fit_pars, range(len(fit_pars))):
            self.pars[name].val = pars_val[i] 

    def press_fun(self, r_kpc):
        '''
        Compute the gNFW pressure profile
        ---------------------------------
        r_kpc = radius (kpc)
        '''
        P_0 = self.pars['P_0'].val
        a = self.pars['a'].val
        b = self.pars['b'].val
        c = self.pars['c'].val
        r_p = self.pars['r_p'].val
        return P_0/((r_kpc/r_p)**c*(1+(r_kpc/r_p)**a)**((b-c)/a)) 

def read_xy_err(filename, ncol):
    '''
    Read the data from FITS or ASCII file
    -------------------------------------
    ncol = number of columns to read
    '''
    if filename[filename.find('.', -5)+1:] == 'fits':
        data = fits.open(filename)[''].data[0]
    elif filename[filename.find('.', -5)+1:] in ('txt', 'dat'):
        data = np.loadtxt(filename, unpack=True)
    else:
        raise RuntimeError('Unrecognised file extension (not in fits, dat, txt)')
    return data[:ncol]
    
def read_beam(filename):
    '''
    Read the beam data from the specified file up to the first negative or nan value
    --------------------------------------------------------------------------------
    '''
    radius, beam_prof = read_xy_err(filename, ncol=2)
    if np.isnan(beam_prof).sum() > 0.:
        first_nan = np.where(np.isnan(beam_prof))[0][0]
        radius = radius[:first_nan]
        beam_prof = beam_prof[:first_nan]
    if beam_prof.min() < 0.:
        first_neg = np.where(beam_prof < 0)[0][0]
        radius = radius[:first_neg]
        beam_prof = beam_prof[:first_neg]
    return radius, beam_prof

def mybeam(step, maxr_data, approx=False, filename=None, normalize=True, fwhm_beam=None):
    '''
    Set the 2D image of the beam, alternatively from file data or from a normal distribution with given FWHM
    --------------------------------------------------------------------------------------------------------
    step = binning step
    maxr_data = highest radius in the data
    approx = whether to approximate or not the beam to the normal distribution (boolean, default is False)
    filename = name of the file including the beam data
    normalize = whether to normalize or not the output 2D image (boolean, default is True)
    fwhm_beam = Full Width at Half Maximum
    -------------------------------------------------------------------
    RETURN: the 2D image of the beam and his Full Width at Half Maximum
    '''
    if not approx:
        r_irreg, b = read_beam(filename)
        f = interp1d(np.append(-r_irreg, r_irreg), np.append(b, b), 'cubic', bounds_error=False, fill_value=(0., 0.))
        inv_f = lambda x: f(x)-f(0.)/2
        fwhm_beam = 2*optimize.newton(inv_f, x0=5.) 
    maxr = (maxr_data+3*fwhm_beam)//step*step
    rad = np.arange(0., maxr+step, step)
    rad = np.append(-rad[:0:-1], rad)
    rad_cut = rad[np.where(abs(rad) <= 3*fwhm_beam)]
    beam_mat = centdistmat(rad_cut)
    if approx:
        sigma_beam = fwhm_beam/(2*np.sqrt(2*np.log(2)))
        beam_2d = norm.pdf(beam_mat, loc=0., scale=sigma_beam)
    else:
        beam_2d = f(beam_mat)
    if normalize:
        beam_2d /= beam_2d.sum()*step**2
    return beam_2d, fwhm_beam

def centdistmat(r, offset=0.):
    '''
    Create a symmetric matrix of distances from the radius vector
    -------------------------------------------------------------
    r = vector of negative and positive distances with a given step (center value has to be 0)
    offset = value to be added to every distance in the matrix (default is 0)
    ---------------------------------------------
    RETURN: the matrix of distances centered on 0
    '''
    x, y = np.meshgrid(r, r)
    return np.sqrt(x**2+y**2)+offset

def read_tf(filename, approx=False, loc=0., scale=0.02, c=0.95):
    '''
    Read the transfer function data from the specified file
    -------------------------------------------------------
    approx = whether to approximate or not the tf to the normal cdf (boolean, default is False)
    loc, scale, c = location, scale and normalization parameters for the normal cdf approximation
    ---------------------------------------------------------------------------------------------
    RETURN: the vectors of wave numbers and transmission values
    '''
    wn, tf = read_xy_err(filename, ncol=2) # wave number, transmission
    if approx:
        tf = c*norm.cdf(wn, loc, scale)
    return wn, tf

def dist(naxis):
    '''
    Returns a matrix in which the value of each element is proportional to its frequency 
    (https://www.harrisgeospatial.com/docs/DIST.html)
    If you shift the 0 to the centre using fftshift, you obtain a symmetric matrix
    ------------------------------------------------------------------------------------
    naxis = number of elements per row and per column
    -------------------------------------------------
    RETURN: the (naxis x naxis) matrix
    '''
    axis = np.linspace(-naxis//2+1, naxis//2, naxis)
    result = np.sqrt(axis**2+axis[:,np.newaxis]**2)
    return np.roll(result, naxis//2+1, axis=(0, 1))

def filt_image(wn_as, tf, side, step):
    '''
    Create the 2D filtering image from the transfer function data
    -------------------------------------------------------------
    wn_as = vector of wave numbers in arcsec
    tf = transmission data
    side = one side length for the output image
    step = binning step
    -------------------------------
    RETURN: the (side x side) image
    '''
    f = interp1d(wn_as, tf, 'cubic', bounds_error=False, fill_value=tuple([tf[0], tf[-1]])) # tf interpolation
    kmax = 1/step
    karr = dist(side)/side
    karr /= karr.max()
    karr *= kmax
    return f(karr)

class SZ_data:
    '''
    Class for the SZ data required for the analysis
    -----------------------------------------------
    phys_const = physical constants required (electron rest mass - keV, Thomson cross section - cm^2)
    step = binning step
    kpc_as = kpc in arcsec
    compt_mJy_beam = conversion factor Compton to mJy
    flux_data = radius (arcsec), flux density, statistical error
    beam_2d = 2D image of the beam
    radius = array of radii in arcsec
    sep = index of radius 0
    r_pp = radius in kpc used to compute the pressure profile
    d_mat = matrix of distances in kpc centered on 0 with given step
    filtering = transfer function matrix
    calc_integ = whether to include integrated Compton parameter in the likelihood (boolean, default is False)
    integ_mu = if calc_integ == True, prior mean
    integ_sig = if calc_integ == True, prior sigma
    '''
    def __init__(self, phys_const, step, kpc_as, compt_mJy_beam, flux_data, beam_2d, radius, sep, r_pp, d_mat, filtering, 
                 calc_integ=False, integ_mu=None, integ_sig=None):
        self.phys_const = phys_const
        self.step = step
        self.kpc_as = kpc_as
        self.compt_mJy_beam = compt_mJy_beam
        self.flux_data = flux_data
        self.beam_2d = beam_2d
        self.radius = radius
        self.sep = sep
        self.r_pp = r_pp
        self.d_mat = d_mat
        self.filtering = filtering
        self.calc_integ = calc_integ
        self.integ_mu = integ_mu
        self.integ_sig = integ_sig

def log_lik(pars_val, pars, press, sz, output='ll'):
    '''
    Computes the log-likelihood for the current pressure parameters
    ---------------------------------------------------------------
    pars_val = array of free parameters
    pars = set of pressure parameters
    press = pressure object of the class Pressure
    sz = class of SZ data
    output = desired output
        'll' = log-likelihood
        'chisq' = Chi-Squared
        'pp' = pressure profile
        'bright' = surface brightness profile
        'integ' = integrated Compton parameter (only if calc_integ == True)
    -----------------------------------------------------------------------
    RETURN: desired output or -inf when theta is out of the parameter space
    '''
    # update pars
    press.update_vals(press.fit_pars, pars_val)
    # prior on parameters (-inf if at least one parameter value is out of the parameter space)
    parprior = sum((pars[p].prior() for p in pars))
    if not np.isfinite(parprior):
        return -np.inf
    # pressure profile
    pp = press.press_fun(sz.r_pp)
    # abel transform
    ab = direct_transform(pp, r=sz.r_pp, direction='forward', backend='Python')
    # Compton parameter
    y = sz.phys_const[2]*sz.phys_const[1]/sz.phys_const[0]*ab
    f = interp1d(np.append(-sz.r_pp, sz.r_pp), np.append(y, y), 'cubic', bounds_error=False, fill_value=(0., 0.))
    # Compton parameter 2D image
    y_2d = f(sz.d_mat)
    # Convolution with the beam
    conv_2d = fftconvolve(y_2d, sz.beam_2d, 'same')*sz.step**2
    # Convolution with the transfer function
    FT_map_in = fft2(conv_2d)
    map_out = np.real(ifft2(FT_map_in*sz.filtering))
    # Conversion from Compton parameter to mJy/beam
    map_prof = map_out[conv_2d.shape[0]//2, conv_2d.shape[0]//2:]*sz.compt_mJy_beam
    g = interp1d(sz.radius[sz.sep:], map_prof, 'cubic', fill_value='extrapolate')
    # Log-likelihood calculation
    chisq = np.nansum(((sz.flux_data[1]-g(sz.flux_data[0]))/sz.flux_data[2])**2)
    log_lik = -chisq/2
    if sz.calc_integ:
        cint = simps(np.concatenate((f(0), y), axis=None)*
                     np.arange(0, sz.r_pp[-1]/sz.kpc_as/60+sz.step/60, sz.step/60), 
                     np.arange(0, sz.r_pp[-1]/sz.kpc_as/60+sz.step/60, sz.step/60))*2*np.pi
        new_chi = np.nansum(((cint-sz.integ_mu)/sz.integ_sig)**2)
        log_lik -= new_chi/2
        if output == 'integ':
            return cint
    if output == 'll':
        return log_lik
    elif output == 'chisq':
        return chisq
    elif output == 'pp':
        return pp
    elif output == 'bright':
        return map_prof
    else:
        raise RuntimeError('Unrecognised output name (must be "ll", "chisq", "pp", "bright" or "integ")')

def prelim_fit(sampler, pars, fit_pars, silent=False, maxiter=10):
    '''
    Preliminary fit on parameters to increase likelihood. Adapted from MBProj2
    --------------------------------------------------------------------------
    sampler = emcee EnsembleSampler object
    pars = set of pressure parameters
    fit_pars = name of the parameters to fit
    silent = print output during fitting (boolean, default is False)
    maxiter = maximum number of iterations (default is 10)
    '''
    print('Fitting (Iteration 1)')
    ctr = [0]
    def minfunc(prs):
        like = sampler.lnprobfn(prs)
        if ctr[0] % 1000 == 0 and not silent:
            print('%10i %10.1f' % (ctr[0], like))
        ctr[0] += 1
        return -like
    thawedpars = [pars[name].val for name in fit_pars]
    lastlike = sampler.lnprobfn(thawedpars)
    fpars = thawedpars
    for i in range(maxiter):
        fitpars = minimize(minfunc, fpars, method='Nelder-Mead')
        fpars = fitpars.x
        fitpars = minimize(minfunc, fpars, method='Powell')
        fpars = fitpars.x
        like = -fitpars.fun
        if abs(lastlike-like) < 0.1:
            break
        if not silent:
            print('Iteration %i' % (i+2))
        lastlike = like
    if not silent:
        print('Fit Result:   %.1f' % like)
    for val, name in zip(fpars, fit_pars):
        pars[name].val = val
    return like

class MCMC:
    '''
    Class for running Markov Chain Monte Carlo
    ------------------------------------------
    sampler = emcee EnsembleSampler object
    pars = set of pressure parameters
    fit_pars = name of the parameters to fit
    seed = random seed (default is None)
    initspread = random Gaussian width added to create initial parameters
    '''
    def __init__(self, sampler, pars, fit_pars, seed=None, initspread=0.01):
        self.pars = pars
        self.fit_pars = fit_pars
        self.seed = seed
        self.initspread = initspread
        # for doing the mcmc sampling
        self.sampler = sampler
        # starting point
        self.pos0 = None
        # header items to write to output file
        self.header = {
            'burn': 0,
            }

    def _generateInitPars(self):
        '''
        Generate initial set of parameters from fit
        -------------------------------------------
        '''
        thawedpars = np.array([self.pars[name].val for name in self.fit_pars])
        assert np.all(np.isfinite(thawedpars))
        # create enough parameters with finite likelihoods
        p0 = []
        _ = 0
        while len(p0) < self.sampler.k:
            if self.seed is not None:
                _ += 1
                np.random.seed(self.seed*_)
            p = thawedpars*(1+np.random.normal(0, self.initspread, size=len(self.fit_pars)))
            if np.isfinite(self.sampler.lnprobfn(p)):
                p0.append(p)
        return p0

    def mcmc_run(self, nburn, nsteps, nthin=1, comp_time=True, autorefit=True, minfrac=0.2, minimprove=0.01):
        '''
        MCMC execution
        --------------
        nburn = number of burn-in iterations
        nsteps = number of chain iterations (after burn-in)
        nthin = thinning
        comp_time = shows the computation time (boolean, default is True)
        autorefit = refit position if new minimum is found during burn in (boolean, default is True)
        minfrac = minimum fraction of burn in to do if new minimum found
        minimprove = minimum improvement in fit statistic to do a new fit
        '''
        def innerburn():
            '''
            Return False if new minimum found and autorefit is set. Adapted from MBProj2
            ----------------------------------------------------------------------------
            '''
            bestfit = None
            starting_guess = [self.pars[name].val for name in self.fit_pars]
            bestprob = initprob = self.sampler.lnprobfn(starting_guess)#np.mean(p0, axis=0))
            p0 = self._generateInitPars()
            self.header['burn'] = nburn
            for i, result in enumerate(self.sampler.sample(p0, iterations=nburn, thin=nthin, storechain=False)):
                if i%10 == 0:
                    print(' Burn %i / %i (%.1f%%)' %(i, nburn, i*100/nburn))
                self.pos0, lnprob, rstate0 = result[:3]
                if lnprob.max()-bestprob > minimprove:
                    bestprob = lnprob.max()
                    maxidx = lnprob.argmax()
                    bestfit = self.pos0[maxidx]
                if (autorefit and i > nburn*minfrac and bestfit is not None ):
                    print('Restarting burn as new best fit has been found (%g > %g)' % (bestprob, initprob))
                    for name, i in zip(self.fit_pars, range(len(self.fit_pars))):
                        self.pars[name].val = bestfit[i] 
                    self.sampler.reset()
                    return False
            self.sampler.reset()
            return True
        time0 = time.time()
        print('Starting burn-in')
        while not innerburn():#pars):
            print('Restarting, as new mininimum found')
            prelim_fit(self.sampler, self.pars, self.fit_pars)
        print('Finished burn-in')
        self.header['length'] = nsteps
        # initial parameters
        if self.pos0 is None:
            print(' Generating initial parameters')
            p0 = self._generateInitPars()
        else:
            print(' Starting from end of burn-in position')
            p0 = self.pos0
        for i, result in enumerate(self.sampler.sample(p0, iterations=nsteps, thin=nthin)):
            if i%10 == 0:
                print(' Sampling %i / %i (%.1f%%)' %(i, nsteps, i*100/nsteps))
        print('Finished sampling')
        time1 = time.time()
        if comp_time:
            h, rem = divmod(time1-time0, 3600)
            print('Computation time: '+str(int(h))+'h '+str(int(rem//60))+'m')
        print('Acceptance fraction: %s' %np.mean(self.sampler.acceptance_fraction))

    def save(self, outfilename, thin=1):
        '''
        Save chain to HDF5 file. Adapted from MBProj2
        ---------------------------------------------
        outfilename = output hdf5 filename
        thin = save every N samples from chain
        '''
        self.header['thin'] = thin
        print('Saving chain to', outfilename)
        with h5py.File(outfilename, 'w') as f:
            # write header entries
            for h in sorted(self.header):
                f.attrs[h] = self.header[h]
            # write list of parameters which are thawed
            f['thawed_params'] = [x.encode('utf-8') for x in self.fit_pars]
            # output chain
            f.create_dataset(
                'chain',
                data=self.sampler.chain[:, ::thin, :].astype(np.float32),
                compression=True, shuffle=True)
            # likelihoods for each walker, iteration
            f.create_dataset(
                'likelihood',
                data=self.sampler.lnprobability[:, ::thin].astype(np.float32),
                compression=True, shuffle=True)
            # acceptance fraction
            f['acceptfrac'] = self.sampler.acceptance_fraction.astype(np.float32)
            # last position in chain
            f['lastpos'] = self.sampler.chain[:, -1, :].astype(np.float32)
        print('Done')
