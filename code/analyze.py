from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import StrVector
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
import numpy as np

def analyze(Y: pd.Series, X: pd.DataFrame, D: str):
    """
    Conduct inference via partialling out as described in Chernozhukov, Hansen, and Spindler, “HIGH-DIMENSIONAL METRICS IN R.”

    Y: Pandas Series with the endogenous variable

    X: Pandas DataFrame with all controls including those 

    D: String name of variable to conduct inference on
    """

    # Import some necessary packages, install if necessary
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    
    if not rpackages.isinstalled("hdm"):
        utils.install_packages("hdm")
    
    stats = importr("stats")
    hdm = importr("hdm")
    
    # Convert pandas dataframes to R dataframes
    with localconverter(ro.default_converter + pandas2ri.converter):
        X_r = ro.conversion.py2rpy(X)
        Y_r = ro.conversion.py2rpy(Y)

    # Get variables for inference
    D_full = [col for col in X.columns if D in col]
    D_r = StrVector(D_full)

    # Conduct inference via partialling out
    results = hdm.rlassoEffects(X_r, Y_r, index=D_r, method="partialling out")

    # Collect the results, convert back to pandas dataframe, and return
    with localconverter(ro.default_converter + pandas2ri.converter):
        coefficients = ro.conversion.rpy2py(results.rx2("coefficients"))
        se = ro.conversion.rpy2py(results.rx2("se"))
        t = ro.conversion.rpy2py(results.rx2("t"))
        p = ro.conversion.rpy2py(results.rx2("pval"))
        conf = ro.conversion.rpy2py(stats.confint(results))

    return pd.DataFrame(zip(coefficients, se, t, p, *conf.T), columns=["coefficients", "se", "t", "p", "lower", "upper"], index=D_full)