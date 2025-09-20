from sympy.core.numbers import Infinity

from delta.propagator import run_all_tests
from scipy import integrate
import numpy as np

def num_integration_check(mu):
    fx = lambda xi: (xi * np.square(np.sinc(xi/2)))/(xi*xi + mu*mu)
    res = integrate.quad(fx, 100, np.inf)
    print(res)

def num_integration_check3(mu, ll):
    fx = lambda xi: (xi * np.square(np.sinc(xi/2)))/(xi*xi + np.square(mu*ll))
    res = integrate.quad(fx, 0., 200)
    print(res)

def num_integration_check2():
    fx = lambda xi: np.square(np.sinc(xi/2))/xi
    res = integrate.quad(fx, 200, np.inf)
    print(res)

if __name__ == '__main__':
    num_integration_check3(1., 2.)
    num_integration_check2()