import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
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
import theano.tensor as tt
from theano.compile.ops import as_op
from theano import shared

class Pressure:
    '''
    Class to parametrize the pressure profile
    -----------------------------------------
    eq_kpc_as = equation for switching between kpc and arcsec
    '''
    def __init__(self, eq_kpc_as):
        self.eq_kpc_as = eq_kpc_as

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

    def functional_form(self, r_kpc, pars, logder=False):
        '''
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        P_0, a, b, c, r_p = set of pressure parameters
        logder = if True returns first order log derivative of pressure, if False returns pressure profile (default is False)
        '''
        # r_kpc = shared(r_kpc.value)
        P_0, a, b, c, r_p = pars[:5]
        P_0 = tt.exp(P_0)
        # return P_0
        a = a if type(a).__bases__[-1] is not pm.model.PyMC3Variable else tt.exp(a) if type(a.distribution) is pm.distributions.continuous.Normal else a
        b = b if type(b).__bases__[-1] is not pm.model.PyMC3Variable else tt.exp(b) if type(b.distribution) is pm.distributions.continuous.Normal else b
        c = c if type(c).__bases__[-1] is not pm.model.PyMC3Variable else tt.exp(c) if type(c.distribution) is pm.distributions.continuous.Normal else c
        r_p = shared(r_p) if type(r_p).__bases__[-1] is not pm.model.PyMC3Variable else r_p
        if logder == False:
            den1 = tt.outer(r_kpc, 1/r_p)**c
            den2 = (1+tt.outer(r_kpc, 1/r_p)**a)**((b-c)/a)#(1+(r_kpc/r_p)**a)**((b-c)/a)
            return P_0/(den1*den2)
        else:
            den = 1+tt.outer(r_kpc, 1/r_p)**a# 1+(r_kpc/r_p)**a
            return (b-c)/den-b

    def prior(self, pars, shape=1):
        '''
        Checks accordance with prior constrains
        ---------------------------------------
        pars = set of pressure parameters
        '''
        if self.slope_prior == True:
            slope_out = self.functional_form(self.r_out, pars, logder=True)
            return np.nansum([pmx.eval_in_model(tt.zeros_like(slope_out)), 
                              pmx.eval_in_model(tt.prod([tt.gt(slope_out, self.max_slopeout), -np.inf*tt.ones_like(slope_out)], axis=0))], axis=0)
        return np.atleast_2d(np.zeros(shape))
    # def set_universal_params(self, r500, cosmo, z):
    #     '''
    #     Apply the set of parameters of the universal pressure profile defined in Arnaud et al. 2010 with given r500 value
    #     -----------------------------------------------------------------------------------------------------------------
    #     r500 = overdensity radius, i.e. radius within which the average density is 500 times the critical density at the cluster's redshift (kpc)
    #     cosmo = cosmology object
    #     z = redshift
    #     '''
    #     raise('Have a look')
    #     c500 = 1.177
    #     self.pars['r_p'].val = r500.to(u.kpc, equivalencies=self.eq_kpc_as).value/c500
    #     self.pars['a'].val = 1.051
    #     self.pars['b'].val = 5.4905
    #     self.pars['c'].val = .3081
    #     # Compute M500 from definition in terms of density and volume
    #     M500 = (4/3*np.pi*cosmo.critical_density(z)*500*r500.to(u.cm)**3).to(u.Msun)
    #     # Compute P500 according to the definition in Equation (5) from Arnaud's paper
    #     hz = cosmo.H(z)/cosmo.H0
    #     h70 = cosmo.H0/(70*cosmo.H0.unit)
    #     P500 = 1.65e-3*hz**(8/3)*(M500/(3e14*h70**-1*u.Msun))**(2/3)*h70**2*u.keV/u.cm**3
    #     self.pars['P_0'].val = (8.403*h70**(-3/2)*P500).value

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

    # def defPars(self):
    #     '''
    #     Default parameter values
    #     ------------------------
    #     P_i = pressure values corresponding to spline knots (kev cm-3)
    #     '''
    #     self.pars = Pressure.defPars(self)
    #     for i in range(self.knots.size):
    #         self.pars.update({'P_'+str(i): Param(self.pr_knots[i].value, minval=0., maxval=1., unit=self.pr_knots.unit)})
    #     return self.pars

    @as_op(itypes=[tt.Generic(), tt.dvector, tt.Generic()], otypes=[tt.dvector])
    def functional_form(press, r_kpc, pars, logder=False):
        '''
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        logder = if True returns first order log derivative of pressure, if False returns pressure profile (default is False)
        '''
        self = press
        # print(r_kpc); print(logder); import sys; sys.exit()
        # p_params = np.array([(pars*self.indexes['ind_'+x]).sum(axis=-1) if x in self.fit_pars else self.pars[x].val for x in ['P_'+str(i) for i in range(self.knots.size)]])
        p_ref = pars[np.where([type(p).__bases__[-1] is pm.model.PyMC3Variable for p in pars])[0][0]]
        shape = p_ref.model.test_point[p_ref.name+'_interval__'].size
        pars = np.array([pmx.eval_in_model(p) if type(p) is pm.model.TransformedRV else np.repeat(p, shape) for p in pars])
        try:
            f = interp1d(np.log10(self.knots.value), np.log10(pars).T, kind='cubic', fill_value='extrapolate')
            # f = interp1d(np.log10(self.knots.value), np.log10(pars).T, kind='cubic', fill_value='extrapolate', axis=-1)
        except:
            if self.knots.size < 4: raise RuntimeError('A minimum of 4 knots is required for a cubic spline model')
        if logder == False:
            # out = np.nan*np.ones((len(pars)[0], r_kpc.size))
            out = 10**f(np.log10(r_kpc))[0]
            # print(tt.as_tensor(out, ndim=2).type); import sys; sys.exit()
            # out = np.atleast_2d(10**f(np.log10(r_kpc)))
            return out.T#*self.pr_knots.unit
        # out = np.inf*np.ones(len(pars))
        out = np.atleast_2d(f._spline.derivative()(np.log10(r_kpc)))
        return out.T*u.Unit('')

    def prior(self, pars, shape=1):
        '''
        Checks accordance with prior constrains
        ---------------------------------------
        pars = set of pressure parameters
        '''
        if self.slope_prior == True:
            slope_out = self.fun(shared(self), shared(np.at_least_1d(self.r_out)), shared(pars), shared(1))
            return np.nansum([pmx.eval_in_model(tt.zeros_like(slope_out)), 
                              pmx.eval_in_model(tt.prod([tt.gt(slope_out, self.max_slopeout), -np.inf*tt.ones_like(slope_out)], axis=0))], axis=0)
        return np.atleast_2d(np.zeros(shape))

#     def set_universal_params(self, r500, cosmo, z):
#         '''
#         Apply the set of parameters of the universal pressure profile defined in Arnaud et al. 2010 with given r500 value
#         -----------------------------------------------------------------------------------------------------------------
#         r500 = overdensity radius, i.e. radius within which the average density is 500 times the critical density at the cluster's redshift (kpc)
#         cosmo = cosmology object
#         z = redshift
#         '''
#         new_press = Press_gNFW(self.eq_kpc_as)
#         new_press.set_universal_params(r500=r500.to(u.kpc, equivalencies=self.eq_kpc_as), cosmo=cosmo, z=z)
#         new_press.fit_pars =  [x for x in new_press.pars if not new_press.pars[x].frozen]
#         new_press.indexes = {'ind_'+x: np.array(new_press.fit_pars) == x if x in new_press.fit_pars else new_press.pars[x].val for x in list(new_press.pars)}
#         p_params = new_press.press_fun(self.knots, [new_press.pars[x].val for x in new_press.fit_pars]).value
#         for i in range(p_params.size):
#             self.pars['P_'+str(i)].val = p_params[0][i]

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

    # def defPars(self):
    #     '''
    #     Default parameter values
    #     ------------------------
    #     P_i = pressure values corresponding to radial bins (kev cm-3)
    #     '''
    #     self.pars = Pressure.defPars(self)
    #     for i in range(self.rbins.size):
    #         self.pars.update({'P_'+str(i): Param(self.pbins[i].value, minval=0., maxval=1., unit=self.pbins.unit)})
    #     return self.pars

    def functional_form(self, r_kpc, pars, i):
        '''
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        '''
        p_low = tt.as_tensor(pars, ndim=2)[self.ind_low[i]]
        # pars = [p if type(p).__bases__[-1] is not pm.model.PyMC3Variable else tt.exp(p) if type(p.distribution) is pm.distributions.continuous.Normal else p for p in pars]
        # print(pmx.eval_in_model((tt.transpose(tt.log(tt.mul(tt.as_tensor(pars[:-1]), 1/tt.as_tensor(pars[1:]))))/self.alpha_den)[:,self.alpha_ind[i]]))
        # print(pmx.eval_in_model(tt.dot(r_kpc, 1/tt.as_tensor(self.r_low[i].value, ndim=1))))
        self.alpha = (tt.transpose(tt.log(tt.mul(tt.exp(tt.as_tensor(pars[:-1])), 1/tt.exp(tt.as_tensor(pars[1:])))))/self.alpha_den)[:,self.alpha_ind[i]]
        # print(pmx.eval_in_model(self.alpha).shape)
        # import sys; sys.exit()
        return tt.transpose(p_low)*tt.mul(r_kpc, 1/tt.as_tensor(self.r_low[i].value))**self.alpha

    def prior(self, pars, shape=1):
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
        return np.atleast_2d([0.])

#     def set_universal_params(self, r500, cosmo, z):
#         '''
#         Apply the set of parameters of the universal pressure profile defined in Arnaud et al. 2010 with given r500 value
#         -----------------------------------------------------------------------------------------------------------------
#         r500 = overdensity radius, i.e. radius within which the average density is 500 times the critical density at the cluster's redshift (kpc)
#         cosmo = cosmology object
#         z = redshift
#         '''
#         new_press = Press_gNFW(self.eq_kpc_as)
#         new_press.set_universal_params(r500=r500.to(u.kpc, equivalencies=self.eq_kpc_as), cosmo=cosmo, z=z)
#         new_press.fit_pars =  [x for x in new_press.pars if not new_press.pars[x].frozen]
#         new_press.indexes = {'ind_'+x: np.array(new_press.fit_pars) == x if x in new_press.fit_pars else new_press.pars[x].val for x in list(new_press.pars)}
#         p_params = new_press.press_fun(self.rbins, [new_press.pars[x].val for x in new_press.fit_pars]).value
#         for i in range(p_params.size):
#             self.pars['P_'+str(i)].val = p_params[0][i]

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

def read_tf(filename, tf_units=[1/u.arcsec, u.Unit('')], approx=False, loc=0., scale=0.02, k=0.95):
    '''
    Read the transfer function data from the specified file
    -------------------------------------------------------
    approx = whether to approximate or not the tf to the normal cdf (boolean, default is False)
    loc, scale, k = location, scale and normalization parameters for the normal cdf approximation
    ---------------------------------------------------------------------------------------------
    RETURN: the vectors of wave numbers and transmission values
    '''
    wn, tf = read_data(filename, ncol=2, units=tf_units) # wave number, transmission
    if approx:
        tf = k*norm.cdf(wn, loc, scale)
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
        self.corr = np.c_[np.diag(self.I_isqrt), np.diag(self.I_isqrt), 2*np.concatenate((np.ones(r.size-2), np.ones(2)/2))]
 
def calc_abel(fr, r, abel_data):
    '''
    Calculation of the integral used in Abel transform. Adapted from PyAbel
    -----------------------------------------------------------------------
    fr = input array to which Abel transform will be applied
    r = array of radii
    abel_data = collection of data required for Abel transform calculation
    '''
    f = np.atleast_2d(fr*2*r)
    P = np.multiply(f[:,None,:], abel_data.I_isqrt[None,:,:]) # set up the integral
    out = np.trapz(P, axis=-1, dx=abel_data.dx) # take the integral
    c1 = np.zeros(f.shape) # build up correction factors
    c2 = np.c_[P[:,abel_data.mask2==1][:,1::2], np.zeros(c1.shape[0])]
    c3 = np.tile(np.atleast_2d(2*np.concatenate((np.ones(r.size-2), np.ones(2)/2))), (c1.shape[0],1))
    corr = np.c_[c1[:,:,None], c2[:,:,None], c3[:,:,None]]
    out = out-0.5*np.trapz(corr[:,:,:2], dx=abel_data.dx, axis=-1)*corr[:,:,-1] # correct for the extra triangle at the start of the integral
    f_r = (f[:,1:]-f[:,:-1])/np.diff(r)
    out[:,:-1] += (abel_data.isqrt*f_r+abel_data.acr*(f[:,:-1]-f_r*r[:-1]))
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
        self.d_mat = [centdistmat(np.array([r.to(u.kpc, equivalencies=eq_kpc_as) for r in radius]).T[i]*u.kpc) for i in range(len(u.arcsec.to(u.kpc, equivalencies=eq_kpc_as)))] # matrix of distances (radially symmetric)
        self.indices = np.tril_indices(sep+1) # position indices of unique values within the matrix of distances
        self.d_arr = [d[sep:,sep:][self.indices] for d in self.d_mat] # array of unique values within the matrix of distances
        self.labels = [np.rint(self.d_mat[i].value*self.d_mat[i].unit.to(step.unit, equivalencies=eq_kpc_as)[i]/step).astype(int) for i in range(len(self.d_mat))]# labels indicating different annuli within the matrix of distances
    
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
        self.r_pp = r_pp#.to(u.kpc, equivalencies=eq_kpc_as)
        self.r_am = r_am
        self.dist = distances(radius, sep, step, eq_kpc_as)
        self.filtering = filtering
        self.abel_data = [abel_data(r.value) for r in r_pp]
        self.calc_integ = calc_integ
        self.integ_mu = integ_mu
        self.integ_sig = integ_sig

'''
def cond_as_op(int_func_1):
    def wrapper(r_pp, pp, sz, ped, out):
        if pp.type.ndim == 1:
            return as_op(itypes=[tt.dvector, tt.dvector, tt.Generic(), tt.dscalar, tt.Generic()], otypes=[tt.dvector])
        return as_op(itypes=[tt.dvector, tt.dmatrix, tt.Generic(), tt.dvector, tt.Generic()], otypes=[tt.dmatrix])
    return wrapper

@cond_as_op#
'''
# @as_op(itypes=[tt.dvector, tt.dmatrix, tt.Generic(), tt.Generic(), tt.lscalar], otypes=[tt.dmatrix])
@as_op(itypes=[tt.dvector, tt.drow, tt.Generic(), tt.Generic(), tt.lscalar], otypes=[tt.dmatrix])
def int_func_1(r, pp, sz, output, i):
    '''
    First intermediate likelihood function
    --------------------------------------
    r = array of radii
    pp = pressure profile
    sz = class of SZ data
    output = desired output
    '''
    # abel transform
    ab = calc_abel(pp, r=r, abel_data=sz.abel_data[i])#[calc_abel(pp[i], r=r[i], abel_data=sz.abel_data[i]) for i in range(len(r))]
    # Compton parameter
    y = (const.sigma_T/(const.m_e*const.c**2)).to('cm3 keV-1 kpc-1').value*ab
    f = interp1d(np.append(-r, r), np.append(y, y, axis=-1), 'cubic', bounds_error=False, fill_value=(0., 0.), axis=-1)
    # Compton parameter 2D image
    y_2d = f(sz.dist.d_mat[i].value)
    # Convolution with the beam and the transfer function at the same time
    map_out = np.real(fftshift(ifft2(np.abs(fft2(y_2d))*sz.filtering), axes=(-2, -1)))
    # Conversion from Compton parameter to mJy/beam
    map_prof = list(map(lambda x: mean(x, labels=sz.dist.labels[i], index=np.arange(sz.sep+1)), map_out))*sz.conv_temp_sb.to(sz.flux_data[i][1].unit)
    return map_prof.value


@as_op(itypes=[tt.dmatrix, tt.Generic(), tt.lscalar], otypes=[tt.dmatrix])
def int_func_2(map_prof, sz, i):
    '''
    Second intermediate likelihood function
    ---------------------------------------
    map_prof = fitted profile
    sz = class of SZ data
    '''
    g = interp1d(sz.radius[sz.sep:].value, map_prof, 'cubic', fill_value='extrapolate', axis=-1)
    return g(sz.flux_data[i][0])

def log_lik_press(pars, press, model, sz, i, output='ll'):
    '''
    Computes the log-likelihood for the current pressure parameters
    ---------------------------------------------------------------
    P_0, a, b, c, r_p, ped = set of pressure parameters
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
    ped = pars[-1]
    # prior on pressure distribution
    pars = pars[:-1]
    p_pr = press.prior(pars, shape=model.test_point[next(iter(model.test_point))].size)
    # mask on infinite values
    mask = tt.isinf(p_pr)
    if tt.eq(mask.sum(axis=-1), mask.ndim).eval()[0]:
        if output == 'll':
            return shared(p_pr)
        return None
    # pressure profile
    # gnfw
    pp = press.functional_form(shared(sz.r_pp[i]), pars).T# for r in sz.r_pp]
    # cubspline
    # pp = press.functional_form(shared(press), shared(sz.r_pp[i]), shared(pars)).T# for r in sz.r_pp]
    # nonparam
    # pp = press.functional_form(shared(sz.r_pp[i]), pars, i)#.T# for r in sz.r_pp]
    # print(pp.type); import sys; sys.exit()
    return pp

def log_lik_prof(pars, pp, shape, sz, i, output='ll'):
    ped = pars[-1]
    int_prof = int_func_1(shared(sz.r_pp[i]), pp, shared(sz), shared(output), shared(i))
    int_prof = int_prof+tt.transpose(tt.as_tensor(ped, ndim=shape))
    return int_prof

    
def log_lik_final(int_prof, sz):
    # Log-likelihood calculation
    map_prof = [int_func_2(int_prof[i], shared(sz), shared(i)) for i in range(len(sz.r_pp))]
    # print([pmx.eval_in_model(x).shape for x in map_prof]); import sys; sys.exit()
    chisq = tt.sum([tt.sum(((fl[1].value-mp)/fl[2].value)**2, axis=-1) for fl, mp in zip(sz.flux_data, map_prof)], axis=0)
    log_lik = -chisq/2
    # log_lik = tt.switch(mask, -np.inf*tt.ones_like(mask), log_lik)
    return log_lik
    # # Optional integrated Compton parameter calculation
    # if sz.calc_integ:
    #     cint = simps(np.concatenate((np.atleast_2d(f(0.)).T, y), axis=-1)*sz.r_am, sz.r_am, axis=-1)*2*np.pi
    #     new_chi = ((cint-sz.integ_mu)/sz.integ_sig)**2
    #     log_lik -= new_chi/2
    #     if output == 'integ':
    #         return cint.value
    # if output == 'll':
    #     if np.any(~mask):
    #         parprior[mask] = log_lik.value
    #         newmap_prof = np.repeat(None, parsprior.size*map_prof.shape[1]).reshape(parsprior.size, map_prof.shape[1])
    #         newmap_prof[mask] = map_prof.value
    #         return np.concatenate((np.atleast_2d(parprior).T, newmap_prof), axis=-1)
    #     return np.concatenate((np.atleast_2d(log_lik.value).T, map_prof.value), axis=-1)
    # elif output == 'chisq':
    #     return chisq.value
    # else:
    #     raise RuntimeError('Unrecognised output name (must be "ll", "chisq", "pp", "bright" or "integ")')

def print_summary(prs, pmed, pstd, medsf, sz):
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
    chisq = np.sum([np.sum((fl[1]-(mp.to(fl[1].unit))/fl[2].value)**2, axis=-1) 
                    for fl, mp in zip(sz.flux_data, g(sz.flux_data[0][0])*sz.flux_data[0][1].unit)], axis=0)
    wid1 = len(max(prs, key=len))
    wid2 = max(list(map(lambda x: len(format(x, '.2e')), pmed)))
    wid3 = max(list(map(lambda x: len(format(x, '.2e')), pstd)))
    # units = [press.pars[n].unit for n in press.fit_pars]
    # wid4 = len(max(map(str, units), key=len))
    print(('{:>%i}' % (wid1+2)).format('|')+
          ('{:>%i} Median |' % max(wid2-6,0)).format('')+
          ('{:>%i} Sd |' % max(wid3-2,0)).format('')+
          # ('{:>%i} Unit' % max(wid4-4,0)).format('')+
          '\n'+'-'*(wid1+16+max(wid2-6,0)+max(wid3-2,0)))#+max(wid4-4,0)))
    for i in range(len(prs)):
        print(('{:>%i}' % (wid1+2)).format('%s |' %prs[i])+
              ('{:>%i}' % max(wid2+3, 9)).format(' %s |' %format(pmed[i], '.2e'))+
              ('{:>%i}' % max(wid3+3, 6)).format(' %s |' %format(pstd[i], '.2e')))#+
              # ('{:>%i}' % max(wid4+1, 5)).format(' %s' %format(units[i])))
    print('-'*(wid1+16+max(wid2-6,0)+max(wid3-2,0))+#max(wid4-4,0))+
          '\nMedian profile Chi2 = %s with %s df' % ('{:.4f}'.format(chisq), np.sum([f[1][~np.isnan(f[1])].size for f in sz.flux_data])-len(prs)))

def save_summary(filename, prs, pmed, pstd, ci):
    '''
    Saves log file with a statistical summary of the posterior distribution
    -----------------------------------------------------------------------
    filename = name for log file
    press = pressure object of the class Pressure
    pmed = array of means of parameters sampled in the chain
    pstd = array of standard deviations of parameters sampled in the chain
    ci = uncertainty level of the interval
    '''
    # units = [press.pars[n].unit for n in press.fit_pars]
    np.savetxt('%s.log' % filename, [pmed, pstd], fmt='%.8e', delimiter='\t', header='This file summarizes MCMC results\n'+
               'Posterior distribution medians + uncertainties (%s%% CI)\n' %ci + ' -- '.join(prs))

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
