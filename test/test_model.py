import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sb

import numpy as np
import pymc as P

from pymc_dram import DRAM

NS = int(1e5)

K = 20.5
C0 = 1.5
CMIN, CMAX = 0.5, 2.5

SIGMA = 0.1

NT = 500
TMIN, TMAX = 0, 5
T = np.linspace(TMIN, TMAX, NT)

def model(C):
    return 2. * np.exp(-C * T / 2.) * np.cos(np.sqrt(K - (C ** 2) / 4.) * T) 

nominal = model(C0)
data = nominal + np.random.normal(0, SIGMA, len(T))

plt.figure()
plt.plot(nominal, label="Nominal response")
plt.plot(data, '.', label="Simulated data")
plt.legend()
plt.show()

def model_factory():
    C = P.Uniform("C", value=C0, lower=CMIN, upper=CMAX)
    
    @P.deterministic(plot=False)
    def response(C=C):
        return model(C)

    Y = P.MvNormalCov(
        'Y',
        response,
        (SIGMA ** 2) * np.eye(NT),
        observed=True,
        plot=False,
        value=data,
    )

    return locals()

mvars = model_factory()
M = P.MCMC(mvars)

M.use_step_method(
    # P.AdaptiveMetropolis,
    DRAM,
    [mvars["C"]],
    # verbose=3,
) 

M.sample(NS, burn=NS / 2)

# plt.figure()
P.Matplot.plot(M)
plt.show()
