import cloudpickle
import numpy as np
import preprofit_funcs as pfuncs
import pymc3 as pm

def calc_lik():

    savedir = './data/'
    with open('%s/szdata_obj.pickle' % savedir, 'rb') as f:
        sz = cloudpickle.load(f)
    with open('%s/press_obj.pickle' % savedir, 'rb') as f:
        press = cloudpickle.load(f)

    # define how many set of paramters
    shape = 2
    with pm.Model() as model:
        ped = pm.Uniform("ped", lower=-1, upper=1, shape=shape).random()
        P_0 = pm.Uniform("p0", lower=0, upper=1, shape=shape).random()
        a = pm.Uniform('a', lower=0.5, upper=5., shape=shape).random()
        b = pm.Uniform('b', lower=1, upper=17, shape=shape).random()
        c = .014
        r_p = pm.Uniform('r_p', lower=100., upper=1000., shape=shape).random()
        return pfuncs.log_lik(P_0, a, b, c, r_p, ped, press, sz)[:,0]

print(calc_lik())
