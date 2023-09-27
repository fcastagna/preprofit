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
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from pytensor.tensor.linalg import solve
from pytensor import shared
from pytensor.link.c.type import Generic

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
    def __init__(self, eq_kpc_as, slope_prior=True, r_out=[1e3]*u.kpc, max_slopeout=-2.):
        Pressure.__init__(self, eq_kpc_as)
        self.slope_prior = slope_prior
        self.r_out = r_out.to(u.kpc, equivalencies=eq_kpc_as)
        self.max_slopeout = max_slopeout

    def functional_form(self, r_kpc, pars, i=None, logder=False):
        '''
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        P_0, a, b, c, r_p = set of pressure parameters
        logder = if True returns first order log derivative of pressure, if False returns pressure profile (default is False)
        '''
        P_0, a, b, c, r_p = pt.as_tensor([10**p for p in pars[:5]])
        if not logder:
            den1 = pt.outer(r_kpc, 1/r_p)**c
            den2 = (1+pt.outer(r_kpc, 1/r_p)**a)**((b-c)/a)
            return pt.transpose(P_0/(den1*den2))
        else:
            den = 1+pt.outer(r_kpc, 1/r_p)**a
            return pt.transpose((b-c)/den-b)

    def prior(self, pars, i):
        '''
        Checks accordance with prior constrains
        ---------------------------------------
        pars = set of pressure parameters
        '''
        if self.slope_prior:
            slope_out = self.functional_form(shared(self.r_out.value), pars, logder=True)
            return pt.switch(pt.gt(slope_out, self.max_slopeout), -np.inf, 0.)
        return pt.as_tensor([0.])

    def get_universal_params(self, cosmo, z, r500=None, M500=None, c500=1.177, a=1.051, b=5.4905, c=0.3081, P0=None):
        '''
        '''
        hz = cosmo.H(z)/cosmo.H0
        h70 = cosmo.H0/(70*cosmo.H0.unit)
        if M500 is None:
            # Compute M500 from definition in terms of density and volume
            M500 = (4/3*np.pi*cosmo.critical_density(z)*500*r500.to(u.cm)**3).to(u.Msun)
        else:
            r500 = ((3/4*M500/(500.*cosmo.critical_density(z)*np.pi))**(1/3)).to(u.kpc)
        # Compute P500 according to the definition in Equation (5) from Arnaud's paper
        mu, mu_e, f_b = .59, 1.14, .175
        pnorm = mu/mu_e*f_b*3/8/np.pi*(const.G.value**(-1/3)*u.kg/u.m/u.s**2).to(u.keV/u.cm**3)/((u.kg/250**2/cosmo.H0**4/u.s**4/3e14/u.Msun).to(''))**(2/3)
        P500 = pnorm*hz**(8/3)*(M500/3e14/u.Msun)**(2/3)
        P0 = 8.403*h70**(-3/2)*P500 if P0 is None else P0*P500
        logunivpars = [np.log10([P0.value[i], a, b, c, (r500.to(u.kpc, equivalencies=self.eq_kpc_as).value/c500)[i]]) for i in range(len(P500))]
        return logunivpars

class Press_rcs(Pressure):

    def __init__(self, knots, eq_kpc_as, slope_prior=True, r_out=1e3*u.kpc, max_slopeout=-2.):
        self.knots = knots.to(u.kpc, equivalencies=eq_kpc_as)
        Pressure.__init__(self, eq_kpc_as)
        self.slope_prior = slope_prior
        self.r_out = r_out.to(u.kpc, equivalencies=eq_kpc_as)
        self.max_slopeout = max_slopeout
        self.N = [len(k)-2 for k in self.knots]
        self.betas = None

    def prior(self, pars, i):
        pars = pt.as_tensor(pars)
        if self.slope_prior == True:
            if self.r_out[i] < self.knots[i][-1]:
                raise RuntimeError("Outer radius should be larger than the outermost knot")
            slopes_out = self.functional_form(shared(self.r_out), pars, i, True)
            return pt.switch(pt.gt(pt.gt(slopes_out, self.max_slopeout).sum(), 0), -np.inf, 0.), slopes_out[0]
        return pt.as_tensor([0.]), None
        
    def functional_form(self, r_kpc, pars, i, logder=False):
        kn = pt.log10(self.knots[i])
        if self.betas is None:
            sv = [(kn > kn[_])*(kn-kn[_])**3-(kn > kn[-2])*(kn-kn[_])*(kn-kn[-2])**2 for _ in range(self.N[i])]
            X = pt.concatenate((pt.atleast_2d(pt.ones(5)), pt.atleast_2d(kn), pt.as_tensor(sv))).T
            self.betas = solve(X, pars)
        if not logder:
            x = pt.log10(r_kpc)
            svr = [(x > kn[_])*(x-kn[_])**3-1/(kn[-1]-kn[-2])*
                   ((kn[-1]-kn[_])*(x > kn[-2])*(x-kn[-2])**3 
                    -(kn[-2]-kn[_])*(x > kn[-1])*(x-kn[-1])**3) 
                    for _ in range(self.N[i])]
            return 10**(self.betas[0]+self.betas[1]*x+pt.sum([self.betas[2+_]*svr[_] for _ in range(self.N[i])], axis=0))
        return pt.as_tensor([
            self.betas[1]+3*(
                pt.sum([self.betas[2+_]*kn[_]**2 for _ in range(self.N[i])], axis=0)
                -kn[-2:].sum()*
                pt.sum([self.betas[2+_]*kn[_] for _ in range(self.N[i])], axis=0)+kn[-2]*kn[-1]*self.betas[2:].sum())])

    def get_universal_params(self, cosmo, z, r500=None, M500=None, c500=1.177, a=1.051, b=5.4905, c=0.3081, P0=None):#, sz=None):
        new_press = Press_gNFW(self.eq_kpc_as)
        gnfw_pars = new_press.get_universal_params(cosmo, z, r500=r500, M500=M500, c500=c500, a=a, b=b, c=c, P0=P0)
        logunivpars = [np.squeeze(np.log10(new_press.functional_form(shared(self.knots[i]), gnfw_pars[i], i).eval())) for i in range(len(gnfw_pars))]
        return logunivpars

class Press_nonparam_plaw(Pressure):
    '''
    Class to parametrize the pressure profile with a non parametric power-law model
    -------------------------------------------------------------------------------
    rbins = radial bins
    slope_prior = apply a prior constrain on outer slope (boolean, default is True)
    max_slopeout = maximum allowed value for the outer slope
    '''
    def __init__(self, rbins, eq_kpc_as, slope_prior=True, max_slopeout=-2.):
        self.rbins = rbins.to(u.kpc, equivalencies=eq_kpc_as)
        self.slope_prior = slope_prior
        self.max_slopeout = max_slopeout
        Pressure.__init__(self, eq_kpc_as)
        self.alpha = pt.ones_like(self.rbins)
        self.alpha_den = [pt.log10(r[1:]/r[:-1]) for r in self.rbins] # denominator for alpha

    def functional_form(self, r_kpc, pars, i, logder=False):
        '''
        Functional form expression for pressure calculation
        ---------------------------------------------------
        r_kpc = radius (kpc)
        pars = set of pressure parameters
        '''
        pars = pt.as_tensor([10**p for p in pars])
        self.alpha = (pt.log10(pt.mul(pars[1:], 1/pars[:-1]))/self.alpha_den[i])[self.alpha_ind[i]]
        self.q = pt.log10(pars[:-1][self.alpha_ind[i]])-pt.mul(self.alpha, pt.log10(self.rbins[i][self.alpha_ind[i]]))#self.r_low[i]))
        out = 10**(pt.mul(self.alpha, pt.log10(r_kpc))+pt.as_tensor([self.q]))
        return out

    def prior(self, pars, i, decr_prior=False):
        '''
        Checks accordance with prior constrains
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
            return pt.switch(pt.gt(slope_out, self.max_slopeout), -np.inf, 0.), slope_out
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
        logunivpars = [np.squeeze(np.log10(new_press.functional_form(shared(self.rbins[i]), gnfw_pars[i], i).eval())) for i in range(len(gnfw_pars))]
        return logunivpars
    
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
        self.d_mat = [centdistmat(np.array([r.to(u.kpc, equivalencies=eq_kpc_as) for r in radius]).T[i]*u.kpc) for i in 
                      range(len(u.arcsec.to(u.kpc, equivalencies=eq_kpc_as)))] # matrix of distances (radially symmetric)
        self.indices = np.tril_indices(sep+1) # position indices of unique values within the matrix of distances
        self.d_arr = [d[sep:,sep:][self.indices] for d in self.d_mat] # array of unique values within the matrix of distances
        self.labels = [np.rint(self.d_mat[i].value*self.d_mat[i].unit.to(step.unit, equivalencies=eq_kpc_as)[i]/step).astype(int) for i in 
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

@as_op(itypes=[pt.dvector, pt.drow, Generic(), pt.zmatrix, pt.dscalar, pt.lmatrix, 
               pt.lscalar, pt.dmatrix, Generic()], otypes=[pt.dvector])
def int_func_1(r, pp, sza, szf, szc, szl, szs, dm, output):
    '''
    First intermediate likelihood function
    --------------------------------------
    r = array of radii
    pp = pressure profile
    sz = class of SZ data
    output = desired output
    '''
    # abel transform
    ab = calc_abel(pp, r=r, abel_data=sza)
    # Compton parameter
    y = (const.sigma_T/(const.m_e*const.c**2)).to('cm3 keV-1 kpc-1').value*ab
    f = interp1d(np.append(-r, r), np.append(y, y, axis=-1), 'cubic', bounds_error=False, fill_value=(0., 0.), axis=-1)
    # Compton parameter 2D image
    y_2d = f(dm)#.value)
    # Convolution with the beam and the transfer function at the same time
    map_out = np.real(fftshift(ifft2(fft2(y_2d)*szf), axes=(-2, -1)))
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
    return g(szfl[0])

def whole_lik(pars, press, szr, sza, szf, szc, szl, szs, dm, szrv, szfl, i, output):
    ped = pt.as_tensor(pars[-1])
    pars = pars[:-1]
    p_pr, slope = press.prior(pars, i)
    if np.isinf(p_pr.eval()):
        return p_pr, pt.zeros_like(szfl[0]), pt.zeros_like(szfl[0]), slope
    pp = press.functional_form(shared(szr), pt.as_tensor(pars), i, False)
    pp = pt.atleast_2d(pp)
    int_prof = int_func_1(shared(szr), pp, shared(sza), shared(szf), shared(szc), 
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
