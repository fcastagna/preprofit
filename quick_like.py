import cloudpickle
import numpy as np
import preprofit_funcs as pfuncs

def calc_lik():
    
    savedir = './data/'
    with open('%s/szdata_obj.pickle' % savedir, 'rb') as f:
        sz = cloudpickle.load(f)
    with open('%s/press_obj.pickle' % savedir, 'rb') as f:
        press = cloudpickle.load(f)
    
    # single set of paramters
    ped = pm.Uniform("ped", lower=-1, upper=1, testval=0.)
    P_0 = pm.Uniform("p0", lower=0, upper=1, testval=.15)
    a = pm.Uniform('a', lower=0.5, upper=5., testval=2.81)
    b = pm.Uniform('b', lower=3, upper=7, testval=6.29)
    c = .014
    r_p = pm.Uniform('r_p', lower=100., upper=1000., testval=380)
    # return pfuncs.log_lik(params_1d, press, sz)[:,0]

    # multiple set of parameters
    # 2 likelihoods
    params_2d = np.array([params_1d, np.array(params_1d)*.9])
    # 200 likelihoods
    # params_2d = np.tile(params_2d, (100,1))
    return pfuncs.log_lik(params_2d, press, sz)[:,0]

calc_lik()
