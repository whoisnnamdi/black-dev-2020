from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import StrVector
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import statsmodels.api as sm
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
        coef_ds = ro.conversion.rpy2py(results.rx2("coefficients"))
        se_ds = ro.conversion.rpy2py(results.rx2("se"))
        t_ds = ro.conversion.rpy2py(results.rx2("t"))
        p_ds = ro.conversion.rpy2py(results.rx2("pval"))
        conf_ds = ro.conversion.rpy2py(stats.confint(results))

    ols = sm.OLS(endog=Y, exog=X[[col for col in X.columns if D in col]].assign(const=1)).fit()

    return pd.DataFrame(zip(coef_ds, se_ds, t_ds, p_ds, *conf_ds.T, ols.params, ols.bse, ols.tvalues, ols.pvalues, *ols.conf_int().values.T), 
                        columns=["coef_ds", "se_ds", "t_ds", "p_ds", "lower_ds", "upper_ds", "coef_ols", "se_ols", "t_ols", "p_ols", "lower_ols", "upper_ols"], 
                        index=D_full)