import cloudpickle
import numpy as np
import preprofit_funcs as pfuncs
import pymc3 as pm
import pymc3_ext as pmx

with pm.Model() as model:
    pass

def calc_lik():

    savedir = './data/'
    with open('%s/szdata_obj.pickle' % savedir, 'rb') as f:
        sz = cloudpickle.load(f)
    with open('%s/press_obj.pickle' % savedir, 'rb') as f:
        press = cloudpickle.load(f)

    testval = np.array([
        [0., .5], # ped
        [.15, .5], # P_0
        [2.81, 4], # a
        [6.29, 3.5], # b
        [380, 700] # r_p
        ])
#    testval = np.array([[-1.20161504e-01,  8.87863338e-02,  2.38608789e+00, 1.45155478e+01,  6.34006287e+02], 
#                        [-1.20403334e-01,  9.35682058e-02,  2.27575994e+00, 1.36006212e+01,  6.33433044e+02]]).T

    # define how many set of parameters
    shape = 2

    with model:
        ped = pm.Uniform("ped", lower=-1, upper=1, shape=shape, testval=testval[0,:shape])
        P_0 = pm.Uniform("p0", lower=0, upper=1, shape=shape, testval=testval[1,:shape])
        a = pm.Uniform('a', lower=0.5, upper=5., shape=shape, testval=testval[2,:shape])
        b = pm.Uniform('b', lower=1, upper=17, shape=shape, testval=testval[3,:shape])
        c = .014
        r_p = pm.Uniform('r_p', lower=100., upper=1000., shape=shape, testval=testval[4,:shape])
        return pfuncs.log_lik(P_0, a, b, c, r_p, ped, press, sz)

with model:
    print(pmx.eval_in_model(calc_lik()))
