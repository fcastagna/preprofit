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
    params_1d = [press.pars[x].val for x in press.fit_pars]
    # return pfuncs.log_lik(params_1d, press, sz)[:,0]

    # multiple set of parameters
    # 2 likelihoods
    params_2d = np.array([params_1d, np.array(params_1d)*.9])
    # 200 likelihoods
    # params_2d = np.tile(params_2d, (100,1))
    return pfuncs.log_lik(params_2d, press, sz)[:,0]

calc_lik()
