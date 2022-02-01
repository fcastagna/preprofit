import numpy as np
from astropy.io import fits
from scipy.stats import norm
from scipy.interpolate import interp1d
from astropy import units as u
from astropy import constants as const
import warnings
from scipy import optimize
from scipy.integrate import simps
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import mean
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
    unit = parameter unit in astropy.units format
    '''
    def __init__(self, val, minval=-1e99, maxval=1e99, frozen=False, unit=u.Unit('')):
        self.val = float(val)
        self.minval = minval
        self.maxval = maxval
        self.frozen = frozen
        self.unit = unit

    def __repr__(self):
        return '<Param: val=%.3g, minval=%.3g, maxval=%.3g, unit=%s, frozen=%s>' % (self.val, self.minval, self.maxval, self.unit, self.frozen)

    def prior(self):
        '''
        Checks accordance with parameter's prior distribution
        -----------------------------------------------------
        '''
        if self.val < self.minval or self.val > self.maxval:
            return -np.inf
        return 0.

class ParamGaussian(Param):
    '''
    Class for Gaussian parameters
    -----------------------------
    prior_mu = prior center
    prior_sigma = prior width
    '''
    def __init__(self, val, prior_mu, prior_sigma, frozen=False, minval=-1e99, maxval=1e99, unit=u.Unit('')):
        Param.__init__(self, val, frozen=frozen, minval=minval, maxval=maxval, unit=unit)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def __repr__(self):
        return '<ParamGaussian: val=%.3g, prior_mu=%.3g, prior_sigma=%.3g, frozen=%s>' % (self.val, self.prior_mu, self.prior_sigma, self.frozen)

    def prior(self):
        '''
        Checks accordance with parameter's prior distribution
        -----------------------------------------------------
        '''
        if self.maxval is not None and self.val > self.maxval:
            return -np.inf
        if self.minval is not None and self.val < self.minval:
            return -np.inf
        if self.prior_sigma == 0:
            return 0.
        return np.log(norm.pdf(self.val, self.prior_mu, self.prior_sigma))

class Pressure:
    '''
    Class to parametrize the pressure profile
    -----------------------------------------
    eq_kpc_as = equation for switching between kpc and arcsec
    '''
    def __init__(self, eq_kpc_as):
        self.pars = self.defPars()
        self.eq_kpc_as = eq_kpc_as
        
    def defPars(self):
        '''
        Default parameter values
        ------------------------
        pedestal = baseline level (mJy/beam)
        '''
        self.pars = {
            'pedestal': Param(0., minval=-1., maxval=1., unit=u.Unit('mJy beam-1'))
             }
        return self.pars

    def update_vals(self, fit_pars, pars_val):
        '''
        Update the parameter values
        ---------------------------
        fit_pars = name of the parameters to update
        pars_val = new parameter values
        '''
        for name, i in zip(fit_pars, range(len(fit_pars))):
            self.pars[name].val = pars_val[i] 

    def press_fun(self, r_kpc, pars):
        '''
        Compute the gNFW pressure profile
        ---------------------------------
        r_kpc = radius (kpc)
        '''
        return self.functional_form(r_kpc, pars)

class Press_gNFW(Pressure):
    '''
    Class to parametrize the pressure profile with a generalized Navarro Frenk & White model (gNFW)
    -----------------------------------------------------------------------------------------------
    slope_prior = apply a prior constrain on outer slope (boolean, default is True)
    r_out = outer radius (serves for outer slope determination)
    max_slopeout = maximum allowed value for the outer slope
    '''
    def __init__(self, eq_kpc_as, slope_prior=True, r_out=1e3*u.kpc, max_slopeout=-2.):
        Pressure.__init__(self, eq_kpc_as)
        self.slope_prior = slope_prior
        self.r_out = r_out.to(u.kpc, equivalencies=eq_kpc_as)
        self.max_slopeout = max_slopeout

    def defPars(self):
        '''
        Default parameter values
        ------------------------
        P_0 = normalizing constant (keV cm-3)
        a = rate of turnover between b and c
        b = logarithmic slope at r/r_p >> 1
        c = logarithmic slope at r/r_p << 1
        r_p = characteristic radius (kpc)
        '''
        self.pars = Pressure.defPars(self)
        self.pars.update({
            'P_0': Param(0.4, minval=0., maxval=2., unit=u.Unit('keV cm-3')),
            'a': Param(1.33, minval=0.1, maxval=20.),
            'b': Param(4.13, minval=0.1, maxval=15.),
            'c': Param(0.014, minval=0., maxval=3.),
            'r_p': Param(300., minval=100., maxval=3000., unit=u.kpc)
        })
        return self.pars

    def functional_form(self, r_kpc, pars, logder=False):
        '''
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        logder = if True returns first order log derivative of pressure, if False returns pressure profile (default is False)
        '''
        P_0, a, b, c, r_p = [((pars*self.indexes['ind_'+x]).sum(axis=-1) if x in self.fit_pars 
                              else self.pars[x].val)*self.pars[x].unit for x in ['P_0', 'a', 'b', 'c', 'r_p']]
        if logder == False:
            return (P_0/(np.outer(r_kpc, 1/r_p)**c*(1+np.outer(r_kpc, 1/r_p)**a)**((b-c)/a))).T
        return (b-c)/(1+(r_kpc/r_p)**a)-b

    def prior(self, pars):
        '''
        Checks accordance with prior constrains
        ---------------------------------------
        pars = set of pressure parameters
        '''
        if self.slope_prior == True:
            slope_out = self.functional_form(self.r_out, pars, logder=True)
            return np.nansum(np.array([np.zeros(slope_out.shape), np.array([slope_out > self.max_slopeout, -np.inf], dtype='O').prod(axis=0)], dtype='O'), axis=0)
        return [0.]
    
    def set_universal_params(self, r500, cosmo, z):
        '''
        Apply the set of parameters of the universal pressure profile defined in Arnaud et al. 2010 with given r500 value
        -----------------------------------------------------------------------------------------------------------------
        r500 = overdensity radius, i.e. radius within which the average density is 500 times the critical density at the cluster's redshift (kpc)
        cosmo = cosmology object
        z = redshift
        '''
        c500 = 1.177
        self.pars['r_p'].val = r500.to(u.kpc, equivalencies=self.eq_kpc_as).value/c500
        self.pars['a'].val = 1.051
        self.pars['b'].val = 5.4905
        self.pars['c'].val = .3081
        # Compute M500 from definition in terms of density and volume
        M500 = (4/3*np.pi*cosmo.critical_density(z)*500*r500.to(u.cm)**3).to(u.Msun)
        # Compute P500 according to the definition in Equation (5) from Arnaud's paper
        hz = cosmo.H(z)/cosmo.H0
        h70 = cosmo.H0/(70*cosmo.H0.unit)
        P500 = 1.65e-3*hz**(8/3)*(M500/(3e14*h70**-1*u.Msun))**(2/3)*h70**2*u.keV/u.cm**3
        self.pars['P_0'].val = (8.403*h70**(-3/2)*P500).value

class Press_cubspline(Pressure):
    '''
    Class to parametrize the pressure profile with a cubic spline model
    -------------------------------------------------------------------
    knots = spline knots
    pr_knots = pressure values corresponding to spline knots
    slope_prior = apply a prior constrain on outer slope (boolean, default is True)
    r_out = outer radius (serves for outer slope determination)
    max_slopeout = maximum allowed value for the outer slope
    '''
    def __init__(self, knots, pr_knots, eq_kpc_as, slope_prior=True, r_out=1e3*u.kpc, max_slopeout=-2.):
        self.knots = knots.to(u.kpc, equivalencies=eq_kpc_as)
        self.pr_knots = pr_knots
        Pressure.__init__(self, eq_kpc_as)
        self.slope_prior = slope_prior
        self.r_out = r_out.to(u.kpc, equivalencies=eq_kpc_as)
        self.max_slopeout = max_slopeout
        
    def defPars(self):
        '''
        Default parameter values
        ------------------------
        P_i = pressure values corresponding to spline knots (kev cm-3)
        '''
        self.pars = Pressure.defPars(self)
        for i in range(self.knots.size):
            self.pars.update({'P_'+str(i): Param(self.pr_knots[i].value, minval=0., maxval=1., unit=self.pr_knots.unit)})
        return self.pars

    def functional_form(self, r_kpc, pars, logder=False):
        '''
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        logder = if True returns first order log derivative of pressure, if False returns pressure profile (default is False)
        '''
        p_params = np.array([(pars*self.indexes['ind_'+x]).sum(axis=-1) if x in self.fit_pars else self.pars[x].val for x in ['P_'+str(i) for i in range(self.knots.size)]])
        mask = np.any(p_params <= 0, axis=0)
        if np.all(mask == True):
            return np.array(np.inf)
        try:
            f = interp1d(np.log10(self.knots.value), np.log10(p_params[:,~mask]).T, kind='cubic', fill_value='extrapolate')
        except:
            if self.knots.size < 4: raise RuntimeError('A minimum of 4 knots is required for a cubic spline model')
        if logder == False:
            out = np.nan*np.ones((mask.size, r_kpc.size))
            out[~mask,:] = 10**f(np.log10(r_kpc.value))
            return out*self.pars['P_0'].unit
        out = np.inf*np.ones(mask.size)
        out[~mask] = f._spline.derivative()(np.log10(r_kpc.value))
        return out*u.Unit('')

    def prior(self, pars):
        '''
        Checks accordance with prior constrains
        ---------------------------------------
        pars = set of pressure parameters
        '''
        if self.slope_prior == True:
            slope_out = self.functional_form(self.r_out, pars, logder=True)
            return np.nansum(np.array([np.zeros(slope_out.shape), np.array([slope_out > self.max_slopeout, -np.inf], dtype='O').prod(axis=0)], dtype='O'), axis=0)
        return [0.]
    
    def set_universal_params(self, r500, cosmo, z):
        '''
        Apply the set of parameters of the universal pressure profile defined in Arnaud et al. 2010 with given r500 value
        -----------------------------------------------------------------------------------------------------------------
        r500 = overdensity radius, i.e. radius within which the average density is 500 times the critical density at the cluster's redshift (kpc)
        cosmo = cosmology object
        z = redshift
        '''
        new_press = Press_gNFW(self.eq_kpc_as)
        new_press.set_universal_params(r500=r500.to(u.kpc, equivalencies=self.eq_kpc_as), cosmo=cosmo, z=z)
        new_press.fit_pars =  [x for x in new_press.pars if not new_press.pars[x].frozen]
        new_press.indexes = {'ind_'+x: np.array(new_press.fit_pars) == x if x in new_press.fit_pars else new_press.pars[x].val for x in list(new_press.pars)}
        p_params = new_press.press_fun(self.knots, [new_press.pars[x].val for x in new_press.fit_pars]).value
        for i in range(p_params.size):
            self.pars['P_'+str(i)].val = p_params[0][i]

class Press_nonparam_plaw(Pressure):
    '''
    Class to parametrize the pressure profile with a non parametric power-law model
    -------------------------------------------------------------------------------
    rbins = radial bins
    pbins = pressure values corresponding to radial bins
    slope_prior = apply a prior constrain on outer slope (boolean, default is True)
    max_slopeout = maximum allowed value for the outer slope
    '''
    def __init__(self, rbins, pbins, eq_kpc_as, slope_prior=True, max_slopeout=-2.):
        self.rbins = rbins.to(u.kpc, equivalencies=eq_kpc_as)
        self.pbins = pbins
        self.slope_prior = slope_prior
        self.max_slopeout = max_slopeout
        Pressure.__init__(self, eq_kpc_as)
        self.alpha = np.atleast_2d(np.ones_like(self.rbins))*u.Unit('')
        self.alpha_den = np.atleast_2d(np.log(self.rbins[:-1]/self.rbins[1:])) # denominator for alpha
        
    def defPars(self):
        '''
        Default parameter values
        ------------------------
        P_i = pressure values corresponding to radial bins (kev cm-3)
        '''
        self.pars = Pressure.defPars(self)
        for i in range(self.rbins.size):
            self.pars.update({'P_'+str(i): Param(self.pbins[i].value, minval=0., maxval=1., unit=self.pbins.unit)})
        return self.pars

    def functional_form(self, r_kpc, pars):
        '''
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        '''
        ind_low = np.maximum(0, np.digitize(r_kpc, self.rbins)-1) # lower bins indexes
        r_low = self.rbins[ind_low] # lower radial bins
        alpha_ind = np.minimum(ind_low, len(self.rbins)-2) # alpha indexes
        pbins = np.array([(pars*self.indexes['ind_'+x]).sum(axis=-1) if x in self.fit_pars 
                          else self.pars[x].val for x in ['P_'+str(i) for i in range(self.pbins.size)]])*self.pars['P_0'].unit
        p_low = pbins[ind_low]
        self.alpha = (np.log(pbins[:-1]/pbins[1:]).T/self.alpha_den)[:,alpha_ind]
        return p_low.T*(r_kpc/r_low)**self.alpha

    def prior(self, pars):
        '''
        Checks accordance with prior constrains
        ---------------------------------------
        pars = set of pressure parameters
        '''
        if self.slope_prior == True:
            i = len(self.rbins)
            P_n_1, P_n = np.array([(pars*self.indexes['ind_'+x]).sum(axis=-1) if x in self.fit_pars 
                                   else self.pars[x].val for x in ['P_'+str(i) for i in range(self.rbins.size-2, self.rbins.size)]])
            slope_out = np.log(P_n/P_n_1)/np.log(self.rbins[i-1]/self.rbins[i-2])
            return np.nansum(np.array([np.zeros(slope_out.shape), np.array([slope_out > self.max_slopeout, -np.inf], dtype='O').prod(axis=0)], dtype='O'), axis=0)
        return [0.]

    def set_universal_params(self, r500, cosmo, z):
        '''
        Apply the set of parameters of the universal pressure profile defined in Arnaud et al. 2010 with given r500 value
        -----------------------------------------------------------------------------------------------------------------
        r500 = overdensity radius, i.e. radius within which the average density is 500 times the critical density at the cluster's redshift (kpc)
        cosmo = cosmology object
        z = redshift
        '''
        new_press = Press_gNFW(self.eq_kpc_as)
        new_press.set_universal_params(r500=r500.to(u.kpc, equivalencies=self.eq_kpc_as), cosmo=cosmo, z=z)
        new_press.fit_pars =  [x for x in new_press.pars if not new_press.pars[x].frozen]
        new_press.indexes = {'ind_'+x: np.array(new_press.fit_pars) == x if x in new_press.fit_pars else new_press.pars[x].val for x in list(new_press.pars)}
        p_params = new_press.press_fun(self.rbins, [new_press.pars[x].val for x in new_press.fit_pars]).value
        for i in range(p_params.size):
            self.pars['P_'+str(i)].val = p_params[0][i]

def read_data(filename, ncol=1, units=u.Unit('')):
    '''
    Universally read data from FITS or ASCII file
    ---------------------------------------------
    ncol = number of columns to read
    units = units in astropy.units format
    '''
    if len([units]) != ncol:
        try:
            units = np.concatenate((units, np.repeat(u.Unit(''), ncol-len([units]))), axis=None)
        except:
            raise RuntimeError('The number of elements in units must equal ncol')
    if filename[-5:] == '.fits':
        data = fits.getdata(filename)
        try:
            if len(data) != len(data.columns):
                data = data[0]
        except:
            pass
    elif filename[-4:] in ('.txt', '.dat'):
        data = np.loadtxt(filename, unpack=True)
    else:
        raise RuntimeError('Unrecognised file extension (not in fits, dat, txt)')
    data = np.array(data, dtype=object)
    data.reshape(np.sort(data.shape))
    dim = np.squeeze(data).shape
    if len(dim) == 1:
        if ncol == 1:
            return data*units
        return list(map(lambda x, y: x*y, data[:ncol], np.array(units)))
    else:
        if dim[0] == dim[1]:
            return data*units
        return list(map(lambda x, y: x*y, data[:ncol], np.array(units)))

def read_beam(filename, ncol, units):
    '''
    Read the beam data from the specified file up to the first negative or nan value
    --------------------------------------------------------------------------------
    filename = name of the file including the beam data
    ncol = number of columns to read
    units = units in astropy.units format
    '''
    radius, beam_prof = read_data(filename, ncol=ncol, units=units)
    if np.isnan(beam_prof).sum() > 0.:
        first_nan = np.where(np.isnan(beam_prof))[0][0]
        radius = radius[:first_nan]
        beam_prof = beam_prof[:first_nan]
    if beam_prof.min() < 0.:
        first_neg = np.where(beam_prof < 0.)[0][0]
        radius = radius[:first_neg]
        beam_prof = beam_prof[:first_neg]
    return radius, beam_prof

def get_central(mat, side):
    '''
    Get the central square of a matrix with given side. If side is even, automatically adopts the subsequent odd number
    -------------------------------------------------------------------------------------------------------------------
    mat = 2D matrix
    side = side of the output matrix
    '''
    if side is None or side > mat.shape[0]:
        warnings.warn("Side value is None or exceeds the original matrix side. The original matrix is returned", stacklevel=2)
        return mat
    centre = mat.shape[0]//2
    return mat[centre-side//2:centre+side//2+1, centre-side//2:centre+side//2+1]

def mybeam(step, maxr_data, eq_kpc_as, approx=False, filename=None, units=[u.arcsec, u.beam], crop_image=False, cropped_side=None, normalize=True, fwhm_beam=None):
    '''
    Set the 2D image of the beam, alternatively from file data or from a normal distribution with given FWHM
    --------------------------------------------------------------------------------------------------------
    step = binning step
    maxr_data = highest radius in the data
    eq_kpc_as = equation for switching between kpc and arcsec
    approx = whether to approximate or not the beam to the normal distribution (boolean, default is False)
    filename = name of the file including the beam data
    units = units in astropy.units format
    crop_image = whether to crop or not the original 2D image (default is False)
    cropped_side = side of the cropped image (in pixels, default is None)
    normalize = whether to normalize or not the output 2D image (boolean, default is True)
    fwhm_beam = Full Width at Half Maximum
    -------------------------------------------------------------------
    RETURN: the 2D image of the beam and the Full Width at Half Maximum
    '''
    if not approx:
        try:
            r_irreg, b = read_beam(filename, ncol=2, units=units)
            f = interp1d(np.append(-r_irreg, r_irreg), np.append(b, b), 'cubic', bounds_error=False, fill_value=(0., 0.))
            inv_f = lambda x: f(x)-f(0.)/2
            fwhm_beam = 2*optimize.newton(inv_f, x0=5.)*r_irreg.unit
        except:
            b = read_data(filename, ncol=1, units=np.atleast_2d(units)[0][0])
            r = np.arange(0., b.shape[0]//2*step.value, step.value)*step.unit
            r = np.append(-r[:0:-1].value, r.value)*r.unit
            # If matrix dimensions are even, turn them odd
            if b.shape[0]%2 == 0:
                posmax = np.unravel_index(b.argmax(), b.shape) # get index of maximum value
                if posmax == (0, 0):
                    b = ifftshift(fftshift(b)[1:,1:])
                    b1d = fftshift(b[0,:])
                elif posmax == (b.shape[0]/2, b.shape[0]/2):
                    b = b[1:,1:]
                    b1d = b[b.shape[0]//2,:]
                elif posmax == (b.shape[0]/2-1, b.shape[0]/2-1):
                    b = b[:-1,:-1]
                    b1d = b[b.shape[0]//2,:]
                else:
                    raise RuntimeError('PreProFit is not able to automatically change matrix dimensions from even to odd. Please use an (odd x odd) matrix')
            g = interp1d(r, b1d, 'cubic', bounds_error=False, fill_value=(0., 0.))
            inv_g = lambda x: g(x)-g(0.)/2
            fwhm_beam = 2*optimize.newton(inv_g, x0=50*step.value)*r.unit
    maxr = (maxr_data+3*fwhm_beam.to(maxr_data.unit, equivalencies=eq_kpc_as))//step.to(maxr_data.unit, 
                                                                                        equivalencies=eq_kpc_as)*step.to(maxr_data.unit, equivalencies=eq_kpc_as)
    rad = np.arange(0., (maxr+step.to(maxr_data.unit, equivalencies=eq_kpc_as)).value, step.to(maxr_data.unit, equivalencies=eq_kpc_as).value)*maxr.unit
    rad = np.append(-rad[:0:-1].value, rad.value)*rad.unit
    beam_mat = centdistmat(rad)
    if approx:
        sigma_beam = fwhm_beam.to(step.unit, equivalencies=eq_kpc_as)/(2*np.sqrt(2*np.log(2)))
        beam_2d = norm.pdf(beam_mat, loc=0., scale=sigma_beam)*u.beam
    else:
        try:
            beam_2d = f(beam_mat)*u.beam
        except:
            beam_2d = b.copy()
    if crop_image:
        if beam_2d[0,0] > beam_2d[beam_2d.shape[0]//2, beam_2d.shape[0]//2]: # peak at the corner
            beam_2d = ifftshift(get_central(fftshift(beam_2d), cropped_side))
        else: # peak at the center
            beam_2d = get_central(beam_2d, cropped_side)     
    if normalize:
        beam_2d /= beam_2d.sum()
        beam_2d *= u.beam
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

def read_tf(filename, tf_units=[1/u.arcsec, u.Unit('')], approx=False, loc=0., scale=0.02, c=0.95):
    '''
    Read the transfer function data from the specified file
    -------------------------------------------------------
    approx = whether to approximate or not the tf to the normal cdf (boolean, default is False)
    loc, scale, c = location, scale and normalization parameters for the normal cdf approximation
    ---------------------------------------------------------------------------------------------
    RETURN: the vectors of wave numbers and transmission values
    '''
    wn, tf = read_data(filename, ncol=2, units=tf_units) # wave number, transmission
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

def filt_image(wn_as, tf, tf_source_team, side, step, eq_kpc_as):
    '''
    Create the 2D filtering image from the transfer function data
    -------------------------------------------------------------
    wn_as = vector of wave numbers
    tf = transmission data
    tf_source_team = transfer function provenance ('NIKA', 'MUSTANG', or 'SPT')
    side = one side length for the output image
    step = binning step
    eq_kpc_as = equation for switching between kpc and arcsec
    -------------------------------
    RETURN: the (side x side) image
    '''
    if not tf_source_team in ['NIKA', 'MUSTANG', 'SPT']:
        raise RuntimeError('Accepted values for tf_source_team are: NIKA, MUSTANG, SPT')
    f = interp1d(wn_as, tf, 'cubic', bounds_error=False, fill_value=tuple([tf[0], tf[-1]])) # tf interpolation
    kmax = 1/(step.to(wn_as.unit**-1, equivalencies=eq_kpc_as))
    karr = (dist(side)/side)*u.Unit('')
    if tf_source_team == 'NIKA':
        karr /= karr.max()
    karr *= kmax
    return f(karr)*tf.unit

class abel_data:
    '''
    Class of collection of data required for Abel transform calculation. Adapted from PyAbel
    ----------------------------------------------------------------------------------------
    r = array of radii
    '''
    def __init__(self, r):
        self.dx = abs(r[1]-r[0])
        R, Y = np.meshgrid(r, r, indexing='ij')
        II, JJ = np.meshgrid(np.arange(len(r)), np.arange(len(r)), indexing='ij')
        mask = (II < JJ)
        self.I_isqrt = np.zeros(R.shape)
        self.I_isqrt[mask] = 1./np.sqrt((Y**2 - R**2)[mask])
        self.mask2 = ((II > JJ-2) & (II < JJ+1)) # create a mask that just shows the first two points of the integral    
        self.isqrt = 1./self.I_isqrt[II+1 == JJ]
        if r[0] < r[1]*1e-8:  # special case for r[0] = 0
            ratio = np.append(np.cosh(1), r[2:]/r[1:-1])
        else:
            ratio = r[1:]/r[:-1]
        self.acr = np.arccosh(ratio)
            
def calc_abel(fr, r, abel_data):
    '''
    Calculation of the integral used in Abel transform. Adapted from PyAbel
    -----------------------------------------------------------------------
    fr = input array to which Abel transform will be applied
    r = array of radii
    abel_data = collection of data required for Abel transform calculation
    '''
    f = fr*2*r
    P = np.array(list(map(lambda x: x*abel_data.I_isqrt, f))) # set up the integral
    out = np.trapz(P, axis=2, dx=abel_data.dx)  # take the integral
    corr = np.array(list(map(lambda x: np.c_[np.zeros(r.size), np.append(x[abel_data.mask2][1::2], 0), 
                                             2*np.concatenate((np.ones(r.size-2), np.ones(2)/2))], P))) # build up correction factors
    out = out-0.5*np.trapz(corr[:,:,:2], dx=abel_data.dx, axis=-1)*corr[:,:,-1] # correct for the extra triangle at the start of the integral
    f_r = (f[:,1:]-f[:,:-1])/np.diff(r)
    out[:,:-1] += np.array([abel_data.isqrt*f_r, abel_data.acr*(f[:,:-1]-f_r*r[:-1])]).sum(axis=0)
    return out
    
class distances:
    '''
    Class of data involving distances required in likelihood computation
    --------------------------------------------------------------------
    radius = array of radii in arcsec
    sep = index of radius 0
    step = binning step
    eq_kpc_as = equation for switching between kpc and arcsec
    '''
    def __init__(self, radius, sep, step, eq_kpc_as):
        self.d_mat = centdistmat(radius.to(u.kpc, equivalencies=eq_kpc_as)) # matrix of distances (radially symmetric)
        self.indices = np.tril_indices(sep+1) # position indices of unique values within the matrix of distances
        self.d_arr = self.d_mat[sep:,sep:][self.indices] # array of unique values within the matrix of distances
        self.labels = np.rint(self.d_mat.to(step.unit, equivalencies=eq_kpc_as)/step).astype(int) # labels indicating different annuli within the matrix of distances
    
def interp_mat(mat, indices, func, sep):
    '''
    Quick interpolation on a radially symmetric matrix
    --------------------------------------------------
    mat = empty matrix to fill in with interpolated values
    indices = indices of unique values in the matrix of distances
    func = interpolation function
    sep = index of radius 0
    '''
    mat[sep:,sep:][indices] = func
    mat[sep:,sep:][indices[::-1]] = func
    mat[sep:,:sep+1] = np.fliplr(mat[sep:,sep:])
    mat[:sep+1,sep:] = np.transpose(mat[sep:,:sep+1])
    mat[:sep+1,:sep+1] = np.fliplr(mat[:sep+1,sep:])
    return mat
               
class SZ_data:
    '''
    Class for the SZ data required for the analysis
    -----------------------------------------------
    step = binning step
    eq_kpc_as = equation for switching between kpc and arcsec
    conv_temp_sb = temperature-dependent conversion factor from Compton to surface brightness data unit
    flux_data = radius, flux density, statistical error
    radius = array of radii in arcsec
    sep = index of radius 0
    r_pp = radius in kpc used to compute the pressure profile
    filtering = transfer function matrix
    abel_data = collection of data required for Abel transform calculation
    calc_integ = whether to include integrated Compton parameter in the likelihood (boolean, default is False)
    integ_mu = if calc_integ == True, prior mean
    integ_sig = if calc_integ == True, prior sigma
    '''
    def __init__(self, step, eq_kpc_as, conv_temp_sb, flux_data, radius, sep, r_pp, r_am, filtering, calc_integ=False, integ_mu=None, integ_sig=None):
        self.step = step
        self.eq_kpc_as = eq_kpc_as
        self.conv_temp_sb = conv_temp_sb
        self.flux_data = flux_data
        self.radius = radius.to(u.arcsec, equivalencies=eq_kpc_as)
        self.sep = sep
        self.r_pp = r_pp.to(u.kpc, equivalencies=eq_kpc_as)
        self.r_am = r_am
        self.dist = distances(radius, sep, step, eq_kpc_as)
        self.filtering = filtering
        self.abel_data = abel_data(r_pp.value)
        self.calc_integ = calc_integ
        self.integ_mu = integ_mu
        self.integ_sig = integ_sig

def log_lik(pars_val, press, sz, output='ll'):
    '''
    Computes the log-likelihood for the current pressure parameters
    ---------------------------------------------------------------
    pars_val = array of free parameters
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
    # prior on parameters
    parsprior = np.any([np.array(pars_val) <= press.min_val, np.array(pars_val) >= press.max_val], axis=0).sum(axis=-1)
    parsprior = np.nansum(np.array([np.zeros(parsprior.shape), np.array([parsprior, -np.inf], dtype='O').prod(axis=0)], dtype='O'), axis=0)
    # prior on pressure distribution
    pressprior = press.prior(pars_val)
    # overall prior
    parprior = np.sum([parsprior, pressprior], axis=0)
    # mask on infinite values
    mask = np.isfinite(np.float64(parprior))
    if mask.sum(axis=-1) == 0:
        if output == 'll':
            return np.concatenate((np.atleast_2d(parprior).T, np.array([[None]]*parprior.size)), axis=-1)
        return None
    # pressure profile
    pp = press.press_fun(r_kpc=sz.r_pp, pars=np.array(pars_val)[mask])
    if output == 'pp':
        return pp
    # abel transform
    ab = calc_abel(pp.value, r=sz.r_pp.value, abel_data=sz.abel_data)*pp.unit*sz.r_pp.unit
    # Compton parameter
    y = (const.sigma_T/(const.m_e*const.c**2)*ab).to('')
    f = interp1d(np.append(-sz.r_pp, sz.r_pp), np.append(y, y, axis=-1), 'cubic', bounds_error=False, fill_value=(0., 0.), axis=-1)
    # Compton parameter 2D image
    y_2d = f(sz.dist.d_mat)
    # Convolution with the beam and the transfer function at the same time
    map_out = np.real(fftshift(ifft2(np.abs(fft2(y_2d))*sz.filtering), axes=(-2, -1)))
    # Conversion from Compton parameter to mJy/beam
    map_prof = (list(map(lambda x: mean(x, labels=sz.dist.labels, index=np.arange(sz.sep+1)), map_out))*sz.conv_temp_sb).to(sz.flux_data[1].unit)
    ped = np.array([(pars_val*press.indexes['ind_pedestal']).sum(axis=-1) if 'pedestal' in press.fit_pars else press.pars['pedestal'].val])
    map_prof = map_prof+np.array([ped[0][mask] if 'pedestal' in press.fit_pars else ped]).T*press.pars['pedestal'].unit
    if output == 'bright':
        return map_prof
    g = interp1d(sz.radius[sz.sep:], map_prof, 'cubic', fill_value='extrapolate', axis=-1)
    # Log-likelihood calculation
    chisq = np.nansum(((sz.flux_data[1]-(g(sz.flux_data[0])*map_prof.unit).to(sz.flux_data[1].unit))/sz.flux_data[2])**2, axis=-1)
    log_lik = -chisq/2+parprior[mask]
    # Optional integrated Compton parameter calculation
    if sz.calc_integ:
        cint = simps(np.concatenate((np.atleast_2d(f(0.)).T, y), axis=-1)*sz.r_am, sz.r_am, axis=-1)*2*np.pi
        new_chi = ((cint-sz.integ_mu)/sz.integ_sig)**2
        log_lik -= new_chi/2
        if output == 'integ':
            return cint.value
    if output == 'll':
        if np.any(~mask):
            parprior[mask] = log_lik.value
            newmap_prof = np.repeat(None, parsprior.size*map_prof.shape[1]).reshape(parsprior.size, map_prof.shape[1])
            newmap_prof[mask] = map_prof.value
            return np.concatenate((np.atleast_2d(parprior).T, newmap_prof), axis=-1)
        return np.concatenate((np.atleast_2d(log_lik.value).T, map_prof.value), axis=-1)
    elif output == 'chisq':
        return chisq.value
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
        try:
            like = sampler.log_prob_fn(prs)[0][0]
        except:
            like = sampler.lnprobfn(prs)[0][0]
        if ctr[0] % 1000 == 0 and not silent:
            print('%10i %10.1f' % (ctr[0], like))
        ctr[0] += 1
        return -like
    thawedpars = [pars[name].val for name in fit_pars]
    try:
        lastlike = sampler.log_prob_fn(thawedpars)[0][0]
    except:
        lastlike = sampler.lnprobfn(thawedpars)[0][0]
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
    for val, name in zip(np.atleast_1d(fpars), fit_pars):
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
    initspread = random Gaussian width added to create initial parameters (either scalar or array of same length as fit_pars)
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
        try:
            nw = self.sampler.nwalkers
            lfun = self.sampler.log_prob_fn
        except:
            nw = self.sampler.k
            lfun = self.sampler.lnprobfn
        while len(p0) < nw:
            if self.seed is not None:
                _ += 1
                np.random.seed(self.seed*_)
            p = thawedpars+np.random.normal(0, self.initspread, size=len(self.fit_pars))
            if np.isfinite(lfun(p)[0][0]):
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
            try:
                bestprob = initprob = self.sampler.log_prob_fn(starting_guess)[0][0]
            except:
                bestprob = initprob = self.sampler.lnprobfn(starting_guess)[0][0]
            p0 = self._generateInitPars()
            self.header['burn'] = nburn
            try:
                for i, result in enumerate(self.sampler.sample(p0, iterations=nburn, thin=nthin, storechain=False)):
                    if i%10 == 0:
                        print(' Burn %i / %i (%.1f%%)' %(i, nburn, i*100/nburn))
                    self.pos0, lnprob, rstate = result[:3]
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
            except:
                for i, result in enumerate(self.sampler.sample(p0, iterations=nburn, thin=nburn//2, progress=True)):
                    self.pos0 = result.coords
                    lnprob = result.log_prob
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
        while not innerburn():
            print('Restarting, as new mininimum found')
            prelim_fit(self.sampler, self.pars, self.fit_pars)
        print('Finished burn-in')
        self.header['length'] = nsteps
        # initial parameters
        if self.pos0 is None:
            print('Generating initial parameters')
            p0 = self._generateInitPars()
        else:
            print('Starting from end of burn-in position')
            p0 = self.pos0
        if 'progress' in self.sampler.sample.__code__.co_varnames:
            for res in self.sampler.sample(p0, iterations=nsteps, thin=nthin, progress=True):
                pass
        else:
            for i, result in enumerate(self.sampler.sample(p0, iterations=nsteps, thin=nthin)):
                if i%10 == 0:
                    print(' Sampling %i / %i (%.1f%%)' %(i, nsteps, i*100/nsteps))
        print('Finished sampling')
        time1 = time.time()
        if comp_time:
            h, rem = divmod(time1-time0, 3600)
            print('Computation time: '+str(int(h))+'h '+str(int(rem//60))+'m')
        print('Acceptance fraction: %s' %np.mean(self.sampler.acceptance_fraction))

    def save(self, outfilename):
        '''
        Save chain to HDF5 file. Adapted from MBProj2
        ---------------------------------------------
        outfilename = output hdf5 filename
        '''
        print('Saving chain to', outfilename)
        with h5py.File(outfilename, 'w') as f:
            # write header entries
            for h in sorted(self.header):
                f.attrs[h] = self.header[h]
            # write list of parameters which are thawed
            f['thawed_params'] = [x.encode('utf-8') for x in self.fit_pars]
            # output chain + surface brightness
            f.create_dataset('chain', data=self.sampler.chain.astype(np.float32), compression=True, shuffle=True)
            f.create_dataset('bright', data=self.sampler.blobs.astype(np.float32), compression=True, shuffle=True)
            # likelihoods for each walker, iteration
            f.create_dataset('likelihood', data=self.sampler.lnprobability.astype(np.float32), compression=True, shuffle=True)
            # acceptance fraction
            f['acceptfrac'] = self.sampler.acceptance_fraction.astype(np.float32)
            # last position in chain
            f['lastpos'] = self.sampler.chain[:, -1, :].astype(np.float32)
        print('Done')

def print_summary(press, pmed, pstd, medsf, sz):
    '''
    Prints as output a statistical summary of the posterior distribution
    --------------------------------------------------------------------
    press = pressure object of the class Pressure
    pmed = array of means of parameters sampled in the chain
    pstd = array of standard deviations of parameters sampled in the chain
    medsf = median surface brightness profile 
    sz = class of SZ data
    '''
    g = interp1d(sz.radius[sz.sep:], medsf, 'cubic', fill_value='extrapolate', axis=-1)
    chisq = np.nansum(((sz.flux_data[1]-(g(sz.flux_data[0])*sz.flux_data[1].unit).to(sz.flux_data[1].unit))/sz.flux_data[2])**2, axis=-1)
    wid1 = len(max(press.fit_pars, key=len))
    wid2 = max(list(map(lambda x: len(format(x, '.2e')), pmed)))
    wid3 = max(list(map(lambda x: len(format(x, '.2e')), pstd)))
    units = [press.pars[n].unit for n in press.fit_pars]
    wid4 = len(max(map(str, units), key=len))
    print(('{:>%i}' % (wid1+2)).format('|')+
          ('{:>%i} Median |' % max(wid2-6,0)).format('')+
          ('{:>%i} Sd |' % max(wid3-2,0)).format('')+
          ('{:>%i} Unit' % max(wid4-4,0)).format('')+
          '\n'+'-'*(wid1+21+max(wid2-6,0)+max(wid3-2,0)+max(wid4-4,0)))
    for i in range(len(press.fit_pars)):
        print(('{:>%i}' % (wid1+2)).format('%s |' %press.fit_pars[i])+
              ('{:>%i}' % max(wid2+3, 9)).format(' %s |' %format(pmed[i], '.2e'))+
              ('{:>%i}' % max(wid3+3, 6)).format(' %s |' %format(pstd[i], '.2e'))+
              ('{:>%i}' % max(wid4+1, 5)).format(' %s' %format(units[i])))
    print('-'*(wid1+21+max(wid2-6,0)+max(wid3-2,0)+max(wid4-4,0))+
          '\nMedian profile Chi2 = %s with %s df' % ('{:.4f}'.format(chisq.value), sz.flux_data[1][~np.isnan(sz.flux_data[1])].size-len(press.fit_pars)))

def save_summary(filename, press, pmed, pstd, ci):
    '''
    Saves log file with a statistical summary of the posterior distribution
    -----------------------------------------------------------------------
    filename = name for log file
    press = pressure object of the class Pressure
    pmed = array of means of parameters sampled in the chain
    pstd = array of standard deviations of parameters sampled in the chain
    ci = uncertainty level of the interval
    '''
    units = [press.pars[n].unit for n in press.fit_pars]
    np.savetxt('%s.log' % filename, [pmed, pstd], fmt='%.8e', delimiter='\t', header='This file summarizes MCMC results\n'+
               'Posterior distribution medians + uncertainties (%s%% CI)\n' %ci + ' -- '.join(map(lambda a, b: a+' ('+str(b)+')', press.fit_pars, units)))

def get_outer_slope(flatchain, press, r_out):
    '''
    Get outer slope values from flatchain
    -------------------------------------
    flatchain = chain of parameters (2D format)
    press = pressure object of the class Pressure
    r_out = outer radius
    '''
    slopes = np.zeros(flatchain.shape[0])
    for j in range(slopes.size):
        press.update_vals(press.fit_pars, flatchain[j])
        try:
            slopes[j] = press.functional_form(r_out.to(u.kpc, equivalencies=press.eq_kpc_as), [press.pars[x].val for x in press.fit_pars], logder=True)
        except:
            i = len(press.rbins)
            slopes[j] = np.log(press.pars['P_'+str(i-1)].val/press.pars['P_'+str(i-2)].val)/np.log(press.rbins[i-1]/press.rbins[i-2])
    return slopes

def update_new(self, old_state, new_state, accepted, subset=None):
    '''
    Updated version of the MBProj2 function
    '''
    if subset is None:
        subset = np.ones(len(old_state.coords), dtype=bool)
    m1 = subset & accepted
    m2 = accepted[subset]
    old_state.coords[m1] = new_state.coords[m2]
    old_state.log_prob[m1] = new_state.log_prob[m2]

    if np.sum(list(map(lambda x: new_state.blobs[x] is not None, 
                       range(new_state.blobs.shape[0])))) > 0:
        if old_state.blobs is None:
            raise ValueError(
                "If you start sampling with a given log_prob, "
                "you also need to provide the current list of "
                "blobs at that position."
            )
        old_state.blobs[m1] = new_state.blobs[m2]
    return old_state
