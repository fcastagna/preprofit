import numpy as np
from astropy.io import fits
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import interp1d
from astropy import units as u
from astropy import constants as const
import warnings
from scipy import optimize
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import mean
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from pytensor import shared
from pytensor.tensor.var import TensorVariable
from pytensor.link.c.type import Generic
from pytensor.tensor.linalg import solve

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
    slope_prior = apply a prior constraint on outer slope (boolean, default is True)
    r_out = outer radius (serves for outer slope determination)
    max_slopeout = maximum allowed value for the outer slope
    '''
    def __init__(self, eq_kpc_as, slope_prior=True, r_out=[1e3]*u.kpc, max_slopeout=-2.):
        Pressure.__init__(self, eq_kpc_as)
        self.slope_prior = slope_prior
        self.r_out = np.atleast_1d(r_out.to(u.kpc, equivalencies=eq_kpc_as))
        self.max_slopeout = max_slopeout

    def functional_form(self, r_kpc, pars, i=None, logder=False):
        '''
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        P_0, a, b, c, r_p = set of pressure parameters
        logder = if True returns first order log derivative of pressure, if False returns pressure profile (default is False)
        '''
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

    def prior(self, pars, r_kpc, i):
        '''
        Checks accordance with prior constraints
        ---------------------------------------
        pars = set of pressure parameters
        '''
        if self.slope_prior:
            slope_out = self.functional_form(shared(self.r_out[i].value), pars, logder=True)
            return pt.switch(pt.gt(pt.gt(slope_out, self.max_slopeout).sum(), 0), -np.inf, 0.), slope_out
        return pt.as_tensor([0.]), None

    def get_universal_params(self, cosmo, z, r500=None, M500=None, c500=1.177, a=1.051, b=5.4905, c=0.3081, P0=None):
        '''
        '''
        h70 = cosmo.H0/(70*cosmo.H0.unit)
        if M500 is None:
            # Compute M500 from definition in terms of density and volume
            M500 = (4/3*np.pi*cosmo.critical_density(z)*500*r500.to(u.cm)**3).to(u.Msun)
        else:
            r500 = ((3/4*M500/(500.*cosmo.critical_density(z)*np.pi))**(1/3)).to(u.kpc)
        P0 = 8.403*h70**(-3/2) if P0 is None else P0
        logunivpars = [np.log10([(P0).value, a, b, c, [r500.to(u.kpc, equivalencies=self.eq_kpc_as).value/c500][i]]) for i in range(np.array(z).size)]
        return logunivpars

class Press_rcs(Pressure):

    def __init__(self, knots, eq_kpc_as, slope_prior=True, r_out=1e3*u.kpc, max_slopeout=-2.):
        self.knots = knots.to(u.kpc, equivalencies=eq_kpc_as)
        Pressure.__init__(self, eq_kpc_as)
        self.slope_prior = slope_prior
        self.r_out = np.atleast_1d(r_out.to(u.kpc, equivalencies=eq_kpc_as))
        self.max_slopeout = max_slopeout
        self.N = [len(k)-2 for k in self.knots]
        self.betas = [None]*len(self.knots)

    def prior(self, pars, r_kpc, i):
        pars = pt.as_tensor(pars)
        if self.slope_prior:
            if self.r_out[i] < self.knots[i][-1]:
                raise RuntimeError("Outer radius should be larger than the outermost knot")
            slope_out = self.functional_form(shared(self.r_out), pars, i, True)
            if pt.gt(slope_out, self.max_slopeout).eval(): self.betas[i] = None
            return pt.switch(pt.gt(pt.gt(slope_out, self.max_slopeout).sum(), 0), -np.inf, 0.), slope_out
        return pt.as_tensor([0.]), None
        
    def functional_form(self, r_kpc, pars, i=None, logder=False):
        kn = self.kn[i]
        if self.betas[i] is None:
            X = self.X[i]
            self.betas[i] = solve(X, pars)
        if not logder:
            x = pt.log10(r_kpc/self.r500[i])
            svr = [(x > kn[_])*(x-kn[_])**3-1/(kn[-1]-kn[-2])*
                   ((kn[-1]-kn[_])*(x > kn[-2])*(x-kn[-2])**3 
                    -(kn[-2]-kn[_])*(x > kn[-1])*(x-kn[-1])**3) 
                    for _ in range(self.N[i])]
            out = 10**(self.betas[i][0]+self.betas[i][1]*x+pt.sum([self.betas[i][2+_]*svr[_] for _ in range(self.N[i])], axis=0))
            self.betas[i] = None
            return out
        return pt.as_tensor(
            self.betas[i][1]+3*(
                pt.sum([self.betas[i][2+_]*kn[_]**2 for _ in range(self.N[i])], axis=0)
                -kn[-2:].sum()*
                pt.sum([self.betas[i][2+_]*kn[_] for _ in range(self.N[i])], axis=0)+kn[-2]*kn[-1]*self.betas[i][2:].sum()))

    def get_universal_params(self, cosmo, z, r500=None, M500=None, c500=1.177, a=1.051, b=5.4905, c=0.3081, P0=None):
        new_press = Press_gNFW(self.eq_kpc_as, r_out=self.r_out)
        gnfw_pars = new_press.get_universal_params(cosmo, z, r500=r500, M500=M500, c500=c500, a=a, b=b, c=c, P0=P0)
        logunivpars = [np.squeeze(np.log10(new_press.functional_form(shared(self.knots[i]), gnfw_pars[i], i).eval())) for i in range(len(gnfw_pars))]
        return logunivpars

class Press_nonparam_plaw(Pressure):
    '''
    Class to parametrize the pressure profile with a non parametric power-law model
    -------------------------------------------------------------------------------
    knots = radial bins
    pbins = pressure values corresponding to radial bins
    slope_prior = apply a prior constraint on outer slope (boolean, default is True)
    max_slopeout = maximum allowed value for the outer slope
    '''
    def __init__(self, knots, eq_kpc_as, slope_prior=True, max_slopeout=-2.):
        self.knots = knots.to(u.kpc, equivalencies=eq_kpc_as)
        self.slope_prior = slope_prior
        self.max_slopeout = max_slopeout
        Pressure.__init__(self, eq_kpc_as)
        self.alpha = pt.ones_like(self.knots)
        self.alpha_den = [pt.log10(r[1:]/r[:-1]) for r in self.knots] # denominator for alpha

    def functional_form(self, r_kpc, pars, i=None, logder=False):
        '''
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        '''
        pars = pt.as_tensor([10**p for p in pars])
        self.alpha = (pt.log10(pt.mul(pars[1:], 1/pars[:-1]))/self.alpha_den[i])[self.alpha_ind[i]]
        self.q = pt.log10(pars[:-1][self.alpha_ind[i]])-pt.mul(self.alpha, pt.log10(self.knots[i][self.alpha_ind[i]]))#self.r_low[i]))
        out = 10**(pt.mul(self.alpha, pt.log10(r_kpc))+pt.as_tensor(self.q))
        return out

    def prior(self, pars, r_kpc, i, decr_prior=False):
        '''
        Checks accordance with prior constraints
        ---------------------------------------
        pars = set of pressure parameters
        '''
        if self.slope_prior:
            pars = pt.as_tensor([10**p for p in pars])
            if decr_prior: # doesn't seem to work
                decr = pt.all(pt.diff(pars) < 0)
                if not decr.eval():
                    return pt.as_tensor([np.inf])
            P_n_1, P_n = pars[-2:]
            slope_out = pt.log10(P_n/P_n_1)/self.alpha_den[i][-1]
            return pt.switch(pt.gt(pt.gt(slope_out, self.max_slopeout).sum(), 0), -np.inf, 0.), slope_out
        return pt.as_tensor([0.]), None

    def get_universal_params(self, cosmo, z, r500=None, M500=None, c500=1.177, a=1.051, b=5.4905, c=0.3081, P0=None):#, sz=None):
        '''
        Apply the set of parameters of the universal pressure profile defined in Arnaud et al. 2010 with given r500 value
        -----------------------------------------------------------------------------------------------------------------
        r500 = overdensity radius, i.e. radius within which the average density is 500 times the critical density at the cluster's redshift (kpc)
        cosmo = cosmology object
        z = redshift
        '''
        new_press = Press_gNFW(self.eq_kpc_as)
        gnfw_pars = new_press.get_universal_params(cosmo, z, r500=r500, M500=M500, c500=c500, a=a, b=b, c=c, P0=P0)
        logunivpars = [np.squeeze(np.log10(new_press.functional_form(shared(self.knots[i]), gnfw_pars[i], i).eval())) for i in range(len(gnfw_pars))]
        return logunivpars

def get_P500(x, cosmo, z, M500=3e14*u.Msun, mu=.59, mu_e=1.14, f_b=.175, alpha_P=1/.561-5/3):
    '''
    Compute P500 according to the definition in Equation (5) from Arnaud's paper
    '''
    pconst = (mu/mu_e*f_b*3/8/np.pi*(500*const.G**(-1/4)*cosmo.H0**2/2)**(4/3)*(3e14*u.Msun)**(2/3)).to(u.keV/u.cm**3)
    alpha1_P = lambda x: .1-(alpha_P+.1)*(x/.5)**3/(1+(x/.5)**3)
    hz = cosmo.H(z)/cosmo.H0
    P500 = pconst*hz**(8/3)*(M500/3e14/u.Msun)**(2/3)
    return P500*(M500/3e14/u.Msun)**(alpha_P+alpha1_P(x))

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

def turn_odd(mat):
    posmax = np.unravel_index(mat.argmax(), mat.shape) # get index of maximum value
    if posmax == (0, 0):
        return ifftshift(fftshift(mat)[1:,1:])
    elif posmax == (mat.shape[0]/2, mat.shape[0]/2):
        return mat[1:,1:]
    elif posmax == (mat.shape[0]/2-1, mat.shape[0]/2-1):
        return mat[:-1,:-1]
    else:
        raise RuntimeError('PreProFit is not able to automatically change matrix dimensions from even to odd. Please use an (odd x odd) matrix')

def read_beam_data(step, beam_xy, filename, units, step_data,
                   crop_image, cropped_side):
    try: # 1D
        r_irreg, b = read_beam(filename, ncol=2, units=units)
        f = interp1d(np.append(-r_irreg, r_irreg), np.append(b, b), 'cubic', bounds_error=False, fill_value=(0., 0.))
        inv_f = lambda x: f(x)-f(0.)/2
        fwhm_beam = 2*optimize.newton(inv_f, x0=5.)*r_irreg.unit
        sigma_beam = fwhm_beam/(2*np.sqrt(2*np.log(2)))
        b = multivariate_normal([0,0], sigma_beam**2).pdf(beam_xy)
        freq_2d = dist(b.shape[0])/b.shape[0]/step
        return freq_2d, np.abs(fft2(b)*step**2) # without abs it messes up
    except: # 2D
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
        return freq_2d, gt_(freq_2d)
    
def filtering(step, eq_kpc_as, maxr_data=None, lenr=None, beam_and_tf=False, approx=False, 
              filename=None, units=[u.arcsec, u.beam], crop_image=False, cropped_side=None, 
              fwhm_beam=None, step_data=None, w_tf_1d=None, tf_1d=None):
    '''
    Set the 2D image for the beam + transfer function filtering, 
    alternatively from file data or from a normal distribution with given FWHM
    --------------------------------------------------------------------------------------------------------
    step = binning step (reference unit)
    maxr_data = highest radius in the data
    eq_kpc_as = equation for switching between kpc and arcsec
    beam_and_tf = whether the beam already includes the transfer function filtering (boolean, default is False)
    approx = whether to approximate or not the beam to the normal distribution (boolean, default is False)
    filename = name of the file including the beam data
    units = units in astropy.units format
    crop_image = whether to crop or not the original 2D image (default is False)
    cropped_side = side of the cropped image (in pixels, default is None)
    fwhm_beam = Full Width at Half Maximum
    step_data =
    w_tf_1d = 
    tf_1d = 
    -------------------------------------------------------------------
    RETURN: the 2D image of the beam and the Full Width at Half Maximum
    '''
    # check fwhm_beam, step and maxr_data unit agreement
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
        freq_2d, fft_beam = read_beam_data(
            step, beam_xy, filename, units, step_data, crop_image, cropped_side)
        filtering = fft_beam
    if not beam_and_tf:
        # Apply transfer function filtering
        gt = interp1d(w_tf_1d, tf_1d, 'cubic', bounds_error=False, fill_value=(tf_1d[0], tf_1d[-1]))
        tf_2d = gt(freq_2d)
        filtering = fft_beam*tf_2d
    return freq_2d, fft_beam, filtering

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
    out = np.trapz(P, r, axis=-1) # take the integral
    c1 = np.zeros(f.shape) # build up correction factors
    c2 = np.c_[P[:,abel_data.mask2==1][:,1::2], np.zeros(c1.shape[0])]
    c3 = np.tile(np.atleast_2d(np.concatenate((np.ones(r.size-2), np.ones(2)/2))), (c1.shape[0],1))
    corr = np.c_[c1[:,:,None], c2[:,:,None], c3[:,:,None]]
    rn = np.concatenate((r, [2*r[-1]-r[-2], 3*r[-1]-2*r[-2]]))
    r_lim = np.array([[rn[_], rn[_+1], rn[_+2]] for _ in range(r.size)])
    out = out-0.5*np.trapz(np.c_[corr[:,:,:2], np.atleast_3d(np.zeros(r.size))], 
                           r_lim, axis=-1)*corr[:,:,-1] # correct for the extra triangle at the start of the integral
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
        self.d_mat = [centdistmat(np.array([r.to(u.kpc, equivalencies=eq_kpc_as) for r in radius]).T[i]*u.kpc) for i in 
                      range(len(u.arcsec.to(u.kpc, equivalencies=eq_kpc_as)))] # matrix of distances (radially symmetric)
        self.indices = np.tril_indices(sep+1) # position indices of unique values within the matrix of distances
        self.d_arr = [d[sep:,sep:][self.indices] for d in self.d_mat] # array of unique values within the matrix of distances
        self.labels = [np.rint(self.d_mat[i].value*self.d_mat[i].unit.to(step.unit, equivalencies=eq_kpc_as)[i]/step.value).astype(int) for i in 
                       range(len(self.d_mat))]# labels indicating different annuli within the matrix of distances
    
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
    clus = names of analyzed clusters
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
    def __init__(self, clus, step, eq_kpc_as, conv_temp_sb, flux_data, radius, sep, r_pp, r_am, filtering, calc_integ=False, integ_mu=None, integ_sig=None):
        self.clus = clus
        self.step = step
        self.eq_kpc_as = eq_kpc_as
        self.conv_temp_sb = conv_temp_sb
        self.flux_data = flux_data
        self.radius = radius.to(u.arcsec, equivalencies=eq_kpc_as)
        self.sep = sep
        self.r_pp = r_pp
        self.r_red = [10**np.linspace(np.log10(r.value)[0], np.log10(r.value)[-1], r.size//2) for r in r_pp]*r_pp.unit
        self.r_am = r_am
        self.dist = distances(radius, sep, step, eq_kpc_as)
        self.filtering = filtering
        self.abel_data = [abel_data(r.value) for r in self.r_red]
        self.calc_integ = calc_integ
        self.integ_mu = integ_mu
        self.integ_sig = integ_sig

@as_op(itypes=[pt.dvector, pt.dvector, pt.drow, Generic(), Generic(), pt.dmatrix, pt.dscalar, pt.lmatrix, 
               pt.lscalar, pt.dmatrix, Generic()], otypes=[pt.dvector])
def int_func_1(r, szrd, pp, sza, szi, szf, szc, szl, szs, dm, output):
    '''
    First intermediate likelihood function
    --------------------------------------
    r = array of radii
    pp = pressure profile
    sz = class of SZ data
    output = desired output
    '''
    # abel transform
    gg = interp1d(np.log10(r), np.log10(pp), 'cubic')
    new_pp = 10**gg(np.log10(szrd))
    new_ab = calc_abel(new_pp, r=szrd, abel_data=sza)[0]
    gn = interp1d(np.log10(szrd[:-1]), np.log10(new_ab[:-1]), fill_value='extrapolate')
    ab = np.atleast_2d(np.append(10**gn(np.log10(r[:-1])), 0))
    # Compton parameter
    y = (const.sigma_T/(const.m_e*const.c**2)).to('cm3 keV-1 kpc-1').value*ab
    f = interp1d(np.append(-r, r), np.append(y, y, axis=-1), 'cubic', bounds_error=False, fill_value=(0., 0.), axis=-1)
    # Compton parameter 2D image
    y_2d = np.atleast_3d(interp_mat(np.zeros_like(dm), szi, f(dm[szs:,szs:][szi]), szs)).T
    # Convolution with the beam and the transfer function at the same time
    map_out = np.real(ifft2(fft2(y_2d)*szf))
    # Conversion from Compton parameter to mJy/beam
    map_prof = list(map(lambda x: mean(x, labels=szl, index=np.arange(szs+1)), map_out))
    return map_prof

@as_op(itypes=[pt.dvector, pt.dvector, Generic()], otypes=[pt.dvector])
def int_func_2(map_prof, szrv, szfl):
    '''
    Second intermediate likelihood function
    ---------------------------------------
    map_prof = fitted profile
    sz = class of SZ data
    '''
    g = interp1d(szrv, map_prof, 'cubic', fill_value='extrapolate', axis=-1)
    return g(szfl[0].to(u.arcsec))

def whole_lik(pars, press, szr, szrd, sza, szi, szf, szc, szl, szs, dm, szrv, szfl, i, output):
    ped = pt.as_tensor(pars[-1])
    pars = pars[:-1]
    p_pr, slope = press.prior(pars, szr, i)
    if np.isinf(p_pr.eval()):
        return p_pr, pt.zeros_like(szfl[0]), pt.zeros_like(szfl[0]), slope
    pp = press.functional_form(shared(szr), pt.as_tensor(pars), i, False)
    pp = pt.atleast_2d(pt.mul(pp, press.P500[i]))
    int_prof = int_func_1(shared(szr), shared(szrd), pp, shared(sza), shared(szi), shared(szf), shared(szc), 
                          shared(szl), shared(szs), shared(dm), shared(output))
    int_prof = int_prof+ped
    map_prof = int_func_2(int_prof, shared(szrv), shared(szfl))
    chisq = pt.sum([pt.sum(((szfl[1].value-map_prof)/szfl[2].value)**2, axis=-1)], axis=0)
    log_lik = -chisq/2+p_pr
    return log_lik, pp, int_prof, slope

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
    chisq = np.sum([np.sum(((fl[1]-g(fl[0])[i]*fl[1].unit)/fl[2].value)**2, axis=-1) 
                    for i, fl in enumerate(sz.flux_data)], axis=0)
    wid1 = len(max(prs, key=len))
    wid2 = max(list(map(lambda x: len(format(x, '.2e')), pmed)))
    wid3 = max(list(map(lambda x: len(format(x, '.2e')), pstd)))
    print(('{:>%i}' % (wid1+2)).format('|')+
          ('{:>%i} Median |' % max(wid2-6,0)).format('')+
          ('{:>%i} Sd |' % max(wid3-2,0)).format('')+
          '\n'+'-'*(wid1+16+max(wid2-6,0)+max(wid3-2,0)))
    for i in range(len(prs)):
        print(('{:>%i}' % (wid1+2)).format('%s |' %prs[i])+
              ('{:>%i}' % max(wid2+3, 9)).format(' %s |' %format(pmed[i], '.2e'))+
              ('{:>%i}' % max(wid3+3, 6)).format(' %s |' %format(pstd[i], '.2e')))#+
    print('-'*(wid1+16+max(wid2-6,0)+max(wid3-2,0))+
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
    np.savetxt('%s.log' % filename, [pmed, pstd], fmt='%.8e', delimiter='\t', header='This file summarizes MCMC results\n'+
               'Posterior distribution medians + uncertainties (%s%% CI)\n' %ci + ' -- '.join(prs))
