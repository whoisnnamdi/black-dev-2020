from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import StrVector
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
import numpy as np

def analyze(Y, X, D):
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    
    if not rpackages.isinstalled("hdm"):
        utils.install_packages("hdm")
    
    base = importr("base")
    stats = importr("stats")
    hdm = importr("hdm")
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        X_r = ro.conversion.py2rpy(X)
        Y_r = ro.conversion.py2rpy(Y)

    D_full = [col for col in X.columns if D in col]
    D_r = StrVector(D_full)

    results = hdm.rlassoEffects(X_r, Y_r, index=D_r)

    with localconverter(ro.default_converter + pandas2ri.converter):
        coefficients = ro.conversion.rpy2py(results.rx2("coefficients"))
        se = ro.conversion.rpy2py(results.rx2("se"))
        t = ro.conversion.rpy2py(results.rx2("t"))
        p = ro.conversion.rpy2py(results.rx2("pval"))
        conf = ro.conversion.rpy2py(stats.confint(results))

    return pd.DataFrame(zip(coefficients, se, t, p, *conf.T), columns=["coefficients", "se", "t", "p", "lower", "upper"], index=D_full)