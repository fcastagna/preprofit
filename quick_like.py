import cloudpickle
import numpy as np
import preprofit_funcs as pfuncs

def calc_lik():
    
    savedir = './data/'
    with open('%s/szdata_obj.pickle' % savedir, 'rb') as f:
        sz = cloudpickle.load(f)
    with open('%s/press_obj.pickle' % savedir, 'rb') as f:
        press = cloudpickle.load(f)

    with pm.Model() as model:
        ped = pm.Uniform("ped", lower=-1, upper=1, testval=0.)
        P_0 = pm.Uniform("p0", lower=0, upper=1, testval=.15)
        a = pm.Uniform('a', lower=0.5, upper=5., testval=2.81)
        b = pm.Uniform('b', lower=3, upper=7, testval=6.29)
        c = .014
        r_p = pm.Uniform('r_p', lower=100., upper=1000., testval=380)
        mypar = np.array([[-1.20161504e-01,  8.87863338e-02,  2.38608789e+00, 1.45155478e+01,  6.34006287e+02],
                          [-1.20403334e-01,  9.35682058e-02,  2.27575994e+00, 1.36006212e+01,  6.33433044e+02]])
        ped, P_0, a, b, r_p = mypar.T
        ped = np.atleast_2d(ped)
        return pfuncs.log_lik(sz.r_pp, P_0, a, b, c, r_p, ped, press, sz)[:,0]

    # multiple set of parameters
    # 2 likelihoods
    params_2d = np.array([params_1d, np.array(params_1d)*.9])
    # 200 likelihoods
    # params_2d = np.tile(params_2d, (100,1))
    return pfuncs.log_lik(params_2d, press, sz)[:,0]

calc_lik()
