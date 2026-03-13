import numpy as np
from astropy.io import fits
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import interp1d
from astropy import units as u
from astropy import constants as const
import warnings
from scipy import optimize
from preprofit_plots import tf_diagnostic_plot
from scipy.fftpack import fft2, fftshift, ifftshift
import pytensor.tensor as pt
from pytensor import shared
from pytensor.tensor.var import TensorVariable
from pytensor.tensor.linalg import solve
from pytensor.compile.ops import as_op

class Pressure:
    """
    Class to parametrize the pressure profile
    -----------------------------------------
    z = redshift of galaxy clusterscosmological model adopted
    cosmology = cosmological model adopted
    kpc_as = kpc to arcsec conversion factor
    eq_kpc_as = equation for switching between kpc and arcsec
    """
    def __init__(self, z, cosmology):
        self.z = np.atleast_1d(z)
        self.cosmology = cosmology
        self.kpc_as = cosmology.kpc_proper_per_arcmin(self.z).to('kpc arcsec-1') # number of kpc per arcsec
        self.eq_kpc_as = [(u.arcsec, u.kpc, lambda x: x*self.kpc_as.value, lambda x: x/self.kpc_as.value)] # equation for switching between kpc and arcsec

class Press_rcs(Pressure):
    """
    Class to parametrize the pressure profile with a restricted cubic spline model
    ------------------------------------------------------------------------------
    knots = set of knots
    slope_prior = apply a prior constraint on outer slope (boolean, default is True)
    r_out = outer radius (serves for outer slope determination)
    max_slopeout = maximum allowed value for the outer slope
    N = number of knots - 2
    betas = empty lists used to store parameter values
    """
    def __init__(self, z, cosmology, knots, slope_prior=True, r_out=1e3, max_slopeout=-2.):
        Pressure.__init__(self, z, cosmology)
        self.knots = knots
        self.slope_prior = slope_prior
        self.r_out = np.atleast_1d(r_out)
        self.max_slopeout = max_slopeout
        self.N = [len(k)-2 for k in self.knots]
        self.betas = [None]*len(self.knots)

    def prior(self, pars, r_kpc, i):
        """
        Checks accordance with prior constraints
        ----------------------------------------
        pars = set of pressure parameters
        r_kpc = radius (kpc)
        i = cluster index
        """
        pars = pt.as_tensor_variable(pars)
        if self.slope_prior:
            if self.r_out[i] < self.knots[i][-1]:
                raise RuntimeError("Outer radius should be larger than the outermost knot")
            slope_out = self.functional_form(self.r_out[i], pars, i, logder=True)
            cond = pt.gt(slope_out, self.max_slopeout)
            logp = pt.switch(pt.any(cond), -np.inf, 0.0)
            return logp, slope_out
        return pt.as_tensor([0.]), None

    def functional_form(self, r_kpc, pars, i=None, logder=False):
        """
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        i = cluster index
        logder = if True returns first order log derivative of pressure, if False returns pressure profile (default is False)
        """
        if self.betas[i] is None:
            self.betas[i] = solve(self.X, pars)
        if not logder:
            svr_sum = pt.dot(self.betas[i][2:2 + self.N[0]], self.svr[i])
            out = 10 ** (self.betas[i][0] + self.betas[i][1] * self.x[i] + svr_sum)
            self.betas[i] = None
            return out
        b2 = self.betas[i][2:]
        return pt.as_tensor(
            self.betas[i][1]
            +3*(
                pt.sum(b2 * self.kn[:self.N[0]]**2)
                - self.kn_sum * pt.sum(b2 * self.kn[:self.N[0]])
                + self.kn[-2] * self.kn[-1] * pt.sum(b2)
            )
        )
    
    def get_universal_params(self, r500=None, M500=None, c500=1.177, a=1.051, b=5.4905, c=0.3081, P0=None):
        """
        Apply the set of parameters of the universal pressure profile (Arnaud+10) based on either r500 or M500
        ------------------------------------------------------------------------------------------------------
        r500 = overdensity radius
        M500 = overdensity mass
        c500, a, b, c, P0 = parameters for computing the universal pressure profile (default to Arnaud+10 values)
        """
        new_press = Press_gNFW(z=self.z, cosmology=self.cosmology, r_out=self.r_out)
        gnfw_pars = new_press.get_universal_params(r500=r500, M500=M500, c500=c500, a=a, b=b, c=c, P0=P0)
        logunivpars = [np.squeeze(np.log10(new_press.functional_form(shared(self.knots[i]), gnfw_pars[i], i).eval())) for i in range(len(gnfw_pars))]
        return logunivpars

class Press_gNFW(Pressure):
    """
    Class to parametrize the pressure profile with a generalized Navarro Frenk & White model (gNFW)
    -----------------------------------------------------------------------------------------------
    slope_prior = apply a prior constraint on outer slope (boolean, default is True)
    r_out = outer radius (serves for outer slope determination)
    max_slopeout = maximum allowed value for the outer slope
    """
    def __init__(self, z, cosmology, slope_prior=True, r_out=1e3, max_slopeout=-2.):
        Pressure.__init__(self, z, cosmology)
        self.slope_prior = slope_prior
        self.r_out = np.atleast_1d(r_out)
        self.max_slopeout = max_slopeout

    def prior(self, pars, r_kpc, i):
        """
        Checks accordance with prior constraints
        ----------------------------------------
        pars = set of pressure parameters
        r_kpc = radius (kpc)
        i = cluster index
        """
        if self.slope_prior:
            slope_out = self.functional_form(shared(self.r_out[i]), pars, logder=True)
            cond = pt.gt(slope_out, self.max_slopeout)
            logp = pt.switch(pt.any(cond), -np.inf, 0.0)
            return logp, slope_out
        return pt.as_tensor([0.]), None

    def functional_form(self, r_kpc, pars, i=None, logder=False):
        """
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        i = cluster index
        logder = if True returns first order log derivative of pressure, if False returns pressure profile (default is False)
        """
        P_0, a, b, c = pt.as_tensor([10**p for p in pars[:4]])
        r_p = 10**pars[-1]
        r_p = shared(r_p) if type(r_p) is not TensorVariable else r_p
        if not logder:
            den1 = pt.mul(r_kpc, 1/r_p)**c
            den2 = (1+pt.mul(r_kpc, 1/r_p)**a)**((b-c)/a)
            return P_0/(den1*den2)
        else:
            den = 1+pt.mul(r_kpc, 1/r_p)**a
            return (b-c)/den-b

    def get_universal_params(self, r500=None, M500=None, c500=1.177, a=1.051, b=5.4905, c=0.3081, P0=None):
        """
        Apply the set of parameters of the universal pressure profile (Arnaud+10) based on either r500 or M500
        ------------------------------------------------------------------------------------------------------
        r500 = overdensity radius
        M500 = overdensity mass
        c500, a, b, c, P0 = parameters for computing the universal pressure profile (default to Arnaud+10 values)
        """
        h70 = self.cosmology.H0/(70*self.cosmology.H0.unit)
        if M500 is None:
            # Compute M500 from definition in terms of density and volume
            M500 = (4/3*np.pi*self.cosmology.critical_density(self.z)*500*r500.to(u.cm)**3).to(u.Msun)
        else:
            r500 = ((3/4*M500/(500.*self.cosmology.critical_density(self.z)*np.pi))**(1/3)).to(u.kpc)
        P0 = 8.403*h70**(-3/2) if P0 is None else P0
        logunivpars = np.atleast_2d([np.log10([P0, a, b, c, r500[i].value/c500]) for i in range(self.z.size)])
        logunivpars = [np.log10([P0.value, a, b, c, r500[i].value/c500]) for i in range(np.array(self.z).size)]
        return logunivpars

class Press_nonparam_plaw(Pressure):
    """
    Class to parametrize the pressure profile with a non parametric power-law model
    -------------------------------------------------------------------------------
    knots = radial bins
    slope_prior = apply a prior constraint on outer slope (boolean, default is True)
    max_slopeout = maximum allowed value for the outer slope
    alpha = empty array used to store parameter values 
    alpha_den = denominator for alpha
    """
    def __init__(self, z, cosmology, knots, slope_prior=True, max_slopeout=-2.):
        Pressure.__init__(self, z, cosmology)
        self.knots = knots
        self.slope_prior = slope_prior
        self.max_slopeout = max_slopeout
        self.alpha = pt.ones_like(self.knots)
        self.alpha_den = [pt.log10(r[1:]/r[:-1]) for r in self.knots]

    def prior(self, pars, r_kpc, i):
        """
        Checks accordance with prior constraints
        ----------------------------------------
        pars = set of pressure parameters
        r_kpc = radius (kpc)
        i = cluster index
        """
        if self.slope_prior:
            pars = pt.as_tensor([10**p for p in pars])
            P_n_1, P_n = pars[-2:]
            slope_out = pt.log10(P_n/P_n_1)/self.alpha_den[i][-1]
            cond = pt.gt(slope_out, self.max_slopeout)
            logp = pt.switch(pt.any(cond), -np.inf, 0.0)
            return logp, slope_out
        return pt.as_tensor([0.]), None

    def functional_form(self, r_kpc, pars, i=None, logder=False):
        """
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        i = cluster index
        logder = if True returns first order log derivative of pressure, if False returns pressure profile (default is False)
        """
        pars = pt.as_tensor([10**p for p in pars])
        self.alpha = (pt.log10(pt.mul(pars[1:], 1/pars[:-1]))/self.alpha_den[i])[self.alpha_ind[i]]
        self.q = pt.log10(pars[:-1][self.alpha_ind[i]])-pt.mul(self.alpha, pt.log10(self.knots[i][self.alpha_ind[i]]))
        out = 10**(pt.mul(self.alpha, pt.log10(r_kpc))+pt.as_tensor(self.q))
        return out

    def get_universal_params(self, r500=None, M500=None, c500=1.177, a=1.051, b=5.4905, c=0.3081, P0=None):
        """
        Apply the set of parameters of the universal pressure profile (Arnaud+10) based on either r500 or M500
        ------------------------------------------------------------------------------------------------------
        r500 = overdensity radius
        M500 = overdensity mass
        c500, a, b, c, P0 = parameters for computing the universal pressure profile (default to Arnaud+10 values)
        """
        new_press = Press_gNFW(z=self.z, cosmology=self.cosmology)
        gnfw_pars = new_press.get_universal_params(r500=r500, M500=M500, c500=c500, a=a, b=b, c=c, P0=P0)
        logunivpars = [np.squeeze(np.log10(new_press.functional_form(shared(self.knots[i]), gnfw_pars[i], i).eval())) for i in range(len(gnfw_pars))]
        return logunivpars

class Press_cubspline(Pressure):
    """
    Class to parametrize the pressure profile with a cubic spline model
    -------------------------------------------------------------------
    knots = spline knots
    slope_prior = apply a prior constrain on outer slope (boolean, default is True)
    r_out = outer radius (serves for outer slope determination)
    max_slopeout = maximum allowed value for the outer slope
    """
    def __init__(self, z, cosmology, knots, slope_prior=True, r_out=1e3*u.kpc, max_slopeout=-2.):
        self.knots = knots
        Pressure.__init__(self, z, cosmology)
        self.slope_prior = slope_prior
        self.r_out = np.atleast_1d(r_out)
        self.max_slopeout = max_slopeout

    def prior(self, pars, r_kpc, i):
        """
        Checks accordance with prior constraints
        ----------------------------------------
        pars = set of pressure parameters
        r_kpc = radius (kpc)
        i = cluster index
        """
        pars = pt.as_tensor(pars)
        if self.slope_prior == True:
            r_outs = r_kpc[r_kpc > self.r_out[i]]
            slope_out = self.functional_form(shared(self.knots[i]), shared(r_outs), pars, shared(i), shared(1))
            cond = pt.gt(slope_out, self.max_slopeout)
            logp = pt.switch(pt.any(cond), -np.inf, 0.0)
            return logp, slope_out
        return pt.as_tensor([0.]), None
    
    @as_op(itypes=[pt.dvector, pt.dvector, pt.dvector, pt.lscalar, pt.lscalar], otypes=[pt.dvector])
    def functional_form(knots, r_kpc, pars, i, logder=False):
        """
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        i = cluster index
        logder = if True returns first order log derivative of pressure, if False returns pressure profile (default is False)
        """
        ff = interp1d(np.log10(knots), pars, kind='cubic', bounds_error=False, fill_value='extrapolate')
        if not logder:
            out = 10**ff(np.log10(r_kpc))
        else:
            out = ff._spline.derivative()(np.log10(r_kpc))[:,0]
        return out

    def get_universal_params(self, r500=None, M500=None, c500=1.177, a=1.051, b=5.4905, c=0.3081, P0=None):
        """
        Apply the set of parameters of the universal pressure profile (Arnaud+10) based on either r500 or M500
        ------------------------------------------------------------------------------------------------------
        r500 = overdensity radius
        M500 = overdensity mass
        c500, a, b, c, P0 = parameters for computing the universal pressure profile (default to Arnaud+10 values)
        """
        new_press = Press_gNFW(z=self.z, cosmology=self.cosmology)
        gnfw_pars = new_press.get_universal_params(r500=r500, M500=M500, c500=c500, a=a, b=b, c=c, P0=P0)
        univpars = [np.squeeze(np.log10(new_press.functional_form(shared(self.knots[i]), gnfw_pars[i], i).eval())) for i in range(len(gnfw_pars))]
        return univpars

def get_P500(x, cosmo, z, M500=3e14*u.Msun, mu=.59, mu_e=1.14, f_b=.175, alpha_P=1/.561-5/3):
    """
    Compute P500 according to the definition in Equation (9) in Arnaud+10
    ---------------------------------------------------------------------
    x = scaled radii (r/r500)
    cosmo = cosmological model adopted
    z = redshift
    M500 = overdensity mass
    mu = mean molecular weight
    mu_e = mean molecular weight per free electron
    f_b = baryon fraction
    alpha_P = exponential coefficient defined in Equation (7)
    ---------------------------------------------------------------------
    RETURN: P500 value
    """
    pconst = (mu/mu_e*f_b*3/8/np.pi*(500*const.G**(-1/4)*cosmo.H0**2/2)**(4/3)*(3e14*u.Msun)**(2/3)).to(u.keV/u.cm**3)
    alpha1_P = lambda x: .1-(alpha_P+.1)*(x/.5)**3/(1+(x/.5)**3)
    hz = cosmo.H(z)/cosmo.H0
    P500 = pconst*hz**(8/3)*(M500/3e14/u.Msun)**(2/3)
    return P500*(M500/3e14/u.Msun)**(alpha_P+alpha1_P(x))

def read_data(filename, ncol=1, units=u.Unit('')):
    """
    Universally read data from FITS or ASCII file
    ---------------------------------------------
    filename = path to the file
    ncol = number of columns to read
    units = units in astropy.units format
    ---------------------------------------------
    RETURN: input data    
    """
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
    if len(np.unique([l.size for l in data])) > 1:
        data = np.array(data, dtype=object)
    dim = np.squeeze(data).shape
    if len(dim) == 1:
        if ncol == 1:
            return data*units
        return list(map(lambda x, y: x*y, data[:ncol], np.array(units)))
    else:
        if dim[0] == dim[1]:
            return data*units
        return list(map(lambda x, y: x*y, data[:ncol], np.array(units)))

def get_central(mat, side):
    """
    Get the central square of a matrix with given side. If side is even, automatically adopts the subsequent odd number
    -------------------------------------------------------------------------------------------------------------------
    mat = 2D matrix
    side = side of the output matrix
    """
    if side is None or side > mat.shape[0]:
        warnings.warn("Side value is None or exceeds the original matrix side. The original matrix is returned", stacklevel=2)
        return mat
    centre = mat.shape[0]//2
    return mat[centre-side//2:centre+side//2+1, centre-side//2:centre+side//2+1]

def turn_odd(mat):
    """
    If square matrix has even dimensions, turns them odd
    ----------------------------------------------------
    mat = matrix
    ----------------------------------------------------
    RETURN: reduced matrix
    """
    posmax = np.unravel_index(mat.argmax(), mat.shape) # get index of maximum value
    if posmax == (0, 0):
        return ifftshift(fftshift(mat)[1:,1:])
    elif posmax == (mat.shape[0]/2, mat.shape[0]/2):
        return mat[1:,1:]
    elif posmax == (mat.shape[0]/2-1, mat.shape[0]/2-1):
        return mat[:-1,:-1]
    else:
        raise RuntimeError('PreProFit is not able to automatically change matrix dimensions from even to odd. Please use an (odd x odd) matrix')

def read_beam(filename, ncol, units):
    """
    Read the beam data from the specified file up to the first negative or nan value
    --------------------------------------------------------------------------------
    filename = path to the file
    ncol = number of columns to read
    units = units in astropy.units format
    ---------------------------------------------
    RETURN: input data    
    """
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

def dist(naxis):
    """
    Returns a matrix in which the value of each element is proportional to its frequency 
    (https://www.harrisgeospatial.com/docs/DIST.html)
    If you shift the 0 to the centre using fftshift, you obtain a symmetric matrix
    ------------------------------------------------------------------------------------
    naxis = number of elements per row and per column
    -------------------------------------------------
    RETURN: the (naxis x naxis) matrix
    """
    axis = np.linspace(-naxis//2+1, naxis//2, naxis)
    result = np.sqrt(axis**2+axis[:,np.newaxis]**2)
    return np.roll(result, naxis//2+1, axis=(0, 1))

def read_beam_data(step, filename, units, step_data=None, beam_xy=None, crop_image=None, cropped_side=None, out='beam'):
    """
    Read beam from raw data (either 1D or 2D)
    -----------------------------------------
    step = binning step 
    filename = path to the file
    units = units in astropy.units format
    step_data = if beam data has no explicit radii, specify binning step  
    beam_xy = if 2D, grid of radii
    crop_image = whether to crop or not the original 2D image
    cropped_side = side of the cropped image
    out = required output (either 'fwhm' or 'beam')
    -----------------------------------------
    RETURN: 
        fwhm_beam = if out=='fwhm', returns the fwhm of input data
        freq_2d, beam_2d = otherwise, returns 2D frequency matrix and 2D beam
    """    
    try: 
        # 1D
        r_irreg, b = read_beam(filename, ncol=2, units=units)
        f = interp1d(np.append(-r_irreg, r_irreg), np.append(b, b), 'cubic', bounds_error=False, fill_value=(0., 0.))
        inv_f = lambda x: f(x)-f(0.)/2
        fwhm_beam = 2*optimize.newton(inv_f, x0=5.)*r_irreg.unit
        if out == 'fwhm':
            return fwhm_beam
        sigma_beam = fwhm_beam/(2*np.sqrt(2*np.log(2)))
        b = multivariate_normal([0,0], sigma_beam**2).pdf(beam_xy)
        freq_2d = dist(b.shape[0])/b.shape[0]/step
        return freq_2d, np.abs(fft2(b)*step**2)
    except: 
        # 2D
        b = read_data(filename, ncol=1, units=np.atleast_2d(units)[0][0])
        freq_2d_inp = dist(b.shape[0])/b.shape[0]/step_data
        if b.shape[0]%2 == 0:
            b = turn_odd(b)
            freq_2d_inp = -turn_odd(-freq_2d_inp)
        ind = np.round(freq_2d_inp/freq_2d_inp[0,1])
        tf1dfrom2d = np.array([np.mean(b.value[np.where(ind==_)]) for _ in np.arange(b.shape[0]//2+1)])
        gt_ = interp1d(freq_2d_inp[0,:b.shape[0]//2+1], tf1dfrom2d, 'cubic', bounds_error=False, fill_value=(tf1dfrom2d[0], tf1dfrom2d[-1]))
        side = cropped_side if crop_image else b.shape[0]
        freq_2d = dist(side)/side/step
        return freq_2d, gt_(freq_2d)*u.Unit('')

def filtering(step, eq_kpc_as, maxr_data=None, lenr=None, beam_and_tf=False, approx=False, 
              filename=None, units=[u.arcsec, u.beam], crop_image=False, cropped_side=None, 
              fwhm_beam=None, step_data=None, w_tf_1d=None, tf_1d=None, plotdir='./'):
    """
    Set the 2D image for the beam + transfer function filtering, 
    alternatively from file data or from a normal distribution with given FWHM
    --------------------------------------------------------------------------
    step = binning step (reference unit)
    eq_kpc_as = equation for switching between kpc and arcsec
    maxr_data = maximum radius for radial profile computation
    lenr = if maxr_data is not specified, set length of radius array 
    beam_and_tf = whether the beam already includes the transfer function filtering
    approx = whether to approximate or not the beam to the normal distribution
    filename = path to the file
    units = units in astropy.units format
    crop_image = whether to crop or not the original 2D image
    cropped_side = side of the cropped image
    fwhm_beam = Full Width at Half Maximum
    step_data = if beam data has no explicit radii, specify binning step  
    w_tf_1d = 1D transfer function data (frequency)
    tf_1d = 1D transfer function data (transfer function)
    --------------------------------------------------------------------------
    RETURN: 
        freq_2d = matrix of frequencies
        fft_beam = matrix for beam filtering
        filtering = matrix for beam+tf filtering
    """
    if fwhm_beam is None:
        fwhm_beam = read_beam_data(step, filename, units, step_data, out='fwhm')
    fwhm_beam = fwhm_beam.to(step.unit, equivalencies=eq_kpc_as)
    if maxr_data is not None:
        # set outermost radius 3xfwhm_beam larger than the largest radius of observed data
        maxr_data = maxr_data.to(step.unit, equivalencies=eq_kpc_as)
        maxr = np.ceil((maxr_data+3*fwhm_beam)/step)*step
        lenr = maxr//step+1
    # set up 2D grid
    x, y = np.mgrid[-lenr:lenr+1, -lenr:lenr+1]*step.value
    beam_xy = np.dstack((x, y))
    side = beam_xy.shape[0]
    freq_2d = dist(side)/side/step
    if approx:
        # Apply gaussian approximation
        sigma_beam = fwhm_beam.to(step.unit, equivalencies=eq_kpc_as)/(2*np.sqrt(2*np.log(2)))
        sigma_fft_beam = 1/(2*np.pi*sigma_beam)
        filtering = fft_beam = np.exp(-freq_2d**2/2/sigma_fft_beam**2)
    else:
        # Read from data
        freq_2d, fft_beam = read_beam_data(step, filename, units, step_data, beam_xy, crop_image, cropped_side)
        filtering = fft_beam
    if not beam_and_tf:
        # Apply transfer function filtering
        gt = interp1d(w_tf_1d, tf_1d, 'cubic', bounds_error=False, fill_value=(tf_1d[0], tf_1d[-1]))
        tf_2d = gt(freq_2d)
        filtering = fft_beam*tf_2d
        # Diagnostic plot
        tf_diagnostic_plot(w_tf_1d, tf_1d, freq_2d, tf_2d, plotdir=plotdir)
    return freq_2d, fft_beam, filtering

def centdistmat(r, offset=0.):
    """
    Create a symmetric matrix of distances from the radius vector
    -------------------------------------------------------------
    r = vector of negative and positive distances with a given step (center value has to be 0)
    offset = value to be added to every distance in the matrix
    -------------------------------------------------------------
    RETURN: the matrix of distances centered on 0
    """
    x, y = np.meshgrid(r, r)
    return np.sqrt(x**2+y**2)+offset

def read_tf(filename, tf_units=[1/u.arcsec, u.Unit('')], approx=False, loc=0., scale=0.02, k=0.95):
    """
    Read the transfer function data from the specified file
    -------------------------------------------------------
    filename = path to the file
    tf_units = units in astropy.units format
    approx = whether to approximate or not the tf to the normal cdf
    loc, scale, k = location, scale and normalization parameters for the normal cdf approximation
    -------------------------------------------------------
    RETURN: the vectors of wave numbers and transmission values
    """
    wn, tf = read_data(filename, ncol=2, units=tf_units) # wave number, transmission
    wn_as = wn.to(1/u.arcsec)
    if wn.unit == u.radian**-1:
        wn_as /= 2*np.pi
    if approx:
        tf = k*norm.cdf(wn, loc, scale)
    return wn_as, tf

class abel_data:
    """
    Class of collection of data required for Abel transform calculation. Adapted from PyAbel
    ----------------------------------------------------------------------------------------
    r = array of radii
    """
    def __init__(self, r):
        R, Y = np.meshgrid(r, r, indexing='ij')
        II, JJ = np.meshgrid(np.arange(len(r)), np.arange(len(r)), indexing='ij')
        mask = (II < JJ)
        self.I_isqrt = np.zeros(R.shape)
        self.I_isqrt[mask] = 1./np.sqrt((Y**2 - R**2)[mask])
        self.mask2 = ((II > JJ-2) & (II < JJ+1)) # create a mask that just shows the first two points of the integral    
        self.isqrt = 1./self.I_isqrt[II+1 == JJ]
        if r[0] < r[1]*1e-8: # special case for r[0] = 0
            ratio = np.append(np.cosh(1), r[2:]/r[1:-1])
        else:
            ratio = r[1:]/r[:-1]
        self.acr = np.arccosh(ratio)
        self.corr = np.c_[np.diag(self.I_isqrt), np.diag(self.I_isqrt), 2*np.concatenate((np.ones(r.size-2), np.ones(2)/2))]

class distances:
    """
    Class of data involving distances required in likelihood computation
    --------------------------------------------------------------------
    radius = array of radii in arcsec
    sep = index of radius 0
    step = binning step
    eq_kpc_as = equation for switching between kpc and arcsec
    """
    def __init__(self, radius, sep, step, eq_kpc_as):
        self.d_mat = [centdistmat(np.array([r.to(u.kpc, equivalencies=eq_kpc_as) for r in radius]).T[i]) for i in 
                      range(len(u.arcsec.to(u.kpc, equivalencies=eq_kpc_as)))] # matrix of distances (radially symmetric)
        self.indices = np.tril_indices(sep+1) # position indices of unique values within the matrix of distances
        self.d_arr = [d[sep:,sep:][self.indices] for d in self.d_mat] # array of unique values within the matrix of distances
        self.labels = [np.rint(self.d_mat[i]*u.kpc.to(step.unit, equivalencies=eq_kpc_as)[i]/step.value).astype(int) for i in 
                       range(len(self.d_mat))] # labels indicating different annuli within the matrix of distances
    
class SZ_data:
    """
    Class for the SZ data required for the analysis
    -----------------------------------------------
    clus = names of analyzed clusters
    step = binning step
    eq_kpc_as = equation for switching between kpc and arcsec
    conv_temp_sb = temperature-dependent conversion factor from Compton to surface brightness data unit
    flux_data = radius, flux density, statistical error
    radius = array of radii in arcsec
    sep = index of radius 0
    r_pp = radius in kpc used to compute the pressure profile
    dist = class of distances data
    filtering = transfer function matrix
    abel_data = collection of data required for Abel transform calculation
    """
    def __init__(self, clus, step, eq_kpc_as, conv_temp_sb, flux_data, radius, sep, r_pp, filtering):
        self.clus = clus
        self.step = step
        self.eq_kpc_as = eq_kpc_as
        self.conv_temp_sb = conv_temp_sb.to(flux_data[0][1].unit).value
        self.flux_data = flux_data
        self.radius = radius.to(u.arcsec, equivalencies=eq_kpc_as)
        self.sep = sep
        self.r_pp = r_pp
        self.r_red = [10**np.linspace(np.log10(r.value)[0], np.log10(r.value)[-1], r.size//5)*r.unit for r in r_pp]
        self.dist = distances(radius, sep, step, eq_kpc_as)
        self.filtering = filtering
        self.abel_data = [abel_data(r.value) for r in self.r_red]

def add_attrs(press, nc, sz):
    """
    Add required attributes for the pressure class
    ----------------------------------------------
    press = pressure profile class
    nc = number of clusters analyzed 
    sz = class with SZ data
    """
    if type(press) == Press_nonparam_plaw:
        press.ind_low = [np.maximum(0, np.digitize(sz.r_pp[i].value, press.knots[i])-1) for i in range(nc)] # lower bins indexes
        press.r_low = [p[i] for p, i in zip(press.knots, press.ind_low)] # lower radial bins
        press.alpha_ind = [np.minimum(press.ind_low[i], len(press.knots[i])-2) for i in range(nc)] # alpha indexes
    if type(press) == Press_rcs:
        press.kn = pt.log10(press.knots[0]/press.r500[0])
        # Expand kn for broadcasting
        press.kn_row = press.kn[None, :]
        press.kn_col = press.kn[:, None]
        # Step function masks
        press.gt_mask = press.kn_row > press.kn_col  # shape: (N, N)
        press.gt_mask_last2 = press.kn_row > press.kn[-2]
        # Cubic spline-like terms
        press.diff = press.kn_row - press.kn_col
        press.sv = press.gt_mask * press.diff**3 - press.gt_mask_last2 * (press.diff * (press.kn_row - press.kn[-2])**2)
        # Build matrix X (each row corresponds to one knot)
        press.ones = pt.ones_like(press.kn)
        press.X = pt.stack([press.ones, press.kn] + [press.sv[j, :] for j in range(press.N[0])], axis=1)
        press.kn_sum = press.kn[-2:].sum()
        press.x = [pt.log10(sz.r_pp[i]/press.r500[i]) for i in range(nc)]
        # Compute vectorized spline terms for input radii
        press.x_row = [press.x[i][None, :] for i in range(nc)]
        press.gt_mask_x = [press.x_row[i] > press.kn[:press.N[0], None] for i in range(nc)]
        press.diff_x = [press.x_row[i] - press.kn[:press.N[0], None] for i in range(nc)]
        press.last_term = [(1 / (press.kn[-1] - press.kn[-2])) * (
            (press.kn[-1] - press.kn[:press.N[0], None]) * (press.x_row[i] > press.kn[-2]) * (press.x_row[i] - press.kn[-2])**3
            - (press.kn[-2] - press.kn[:press.N[0], None]) * (press.x_row[i] > press.kn[-1]) * (press.x_row[i] - press.kn[-1])**3
        ) for i in range(nc)]
        press.svr = [press.gt_mask_x[i] * press.diff_x[i]**3 - press.last_term[i] for i in range(nc)]
