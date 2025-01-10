import numpy as np
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
from pytensor.link.c.type import Generic
from scipy.interpolate import interp1d
from scipy.fftpack import ifft2, fft2
from scipy.ndimage import mean
from pytensor import shared

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

@as_op(itypes=[pt.dvector, pt.dvector, pt.drow, Generic(), pt.dmatrix, pt.dscalar, pt.lmatrix, 
               pt.lscalar, pt.dmatrix, Generic()], otypes=[pt.dvector])
def int_func_1(r, szrd, pp, sza, szf, szc, szl, szs, dm, output):
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
    # y = (const.sigma_T/(const.m_e*const.c**2)).to('cm3 keV-1 kpc-1').value*ab
    y = 4.0171007732191115e-06*ab
    f = interp1d(np.append(-r, r), np.append(y, y, axis=-1), 'cubic', bounds_error=False, fill_value=(0., 0.), axis=-1)
    # Compton parameter 2D image
    y_2d = f(dm)
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
    return g(szfl[0])

def whole_lik(lgP_ki, ped_i, press, szr, szrd, sza, szf, szc, szl, szs, dm, szrv, szfl, i, output):
    p_pr, slope = press.prior(lgP_ki, szr, i)
    if np.isinf(p_pr.eval()):
        return p_pr, pt.sum(lgP_ki)*pt.atleast_2d(pt.zeros_like(szr)), pt.sum(lgP_ki+ped_i)*pt.zeros(szs+1), slope
    pp = press.functional_form(shared(szr), pt.as_tensor(lgP_ki), i, False)
    pp = pt.atleast_2d(pt.mul(pp, press.P500[i]))
    int_prof = int_func_1(shared(szr), shared(szrd), pp, shared(sza), shared(szf), shared(szc), 
                          shared(szl), shared(szs), shared(dm), shared(output))
    int_prof = int_prof+ped_i
    map_prof = int_func_2(int_prof, shared(szrv), shared(szfl))
    return map_prof, pp, int_prof, slope
