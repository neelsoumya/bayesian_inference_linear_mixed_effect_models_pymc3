####################################################################################################
# Example code to perform
#	multi-level linear mixed effects regression
#	in a Bayesian framework using PyMc3
# 
# Adapted from:
#	https://docs.pymc.io/notebooks/getting_started
#	https://github.com/fonnesbeck/PyMC3_Oslo/blob/master/notebooks/b.%20Multilevel%20Modeling.ipynb
#	https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/data/cty.dat
#
# Usage:
#   python3 pooled_model_example.py
#
####################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymc3 import Model, sample, Normal, HalfCauchy, Uniform
from pymc3 import forestplot, summary
import pymc3 as pm

import pdb

########################################
# MCMC parameters
########################################
i_num_samples = 2000
i_burnin      = 1000

########################################
# Import radon data
########################################
srrs2 = pd.read_csv('radon.csv')
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state=='MN']

########################################
# Next, obtain the county-level
# predictor, uranium, by combining two
# variables.
########################################
srrs_mn['fips'] = srrs_mn.stfips*1000 + srrs_mn.cntyfips
cty = pd.read_csv('cty.dat')
cty_mn = cty[cty.st=='MN'].copy()
cty_mn[ 'fips'] = 1000*cty_mn.stfips + cty_mn.ctfips

########################################
# Use the merge method to combine home-
#  and county-level information in a
#  single DataFrame.
########################################
srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
# srrs_mn = srrs_mn.drop_duplicates(subset='idnum')
# u = np.log(srrs_mn.Uppm)
n = len(srrs_mn)
srrs_mn.head()


########################################
# Also need a lookup table (dict) for
# each unique county, for indexing.
########################################
srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))

########################################
# Finally, create local copies of
# variables.
########################################
county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
floor_measure = srrs_mn.floor.values


floor = srrs_mn.floor.values
log_radon = srrs_mn.log_radon.values

########################################
# Creaet pooled model
########################################
with Model() as pooled_model:
    
    beta = Normal('beta', 0, sd=1e5, shape=2)
    sigma = HalfCauchy('sigma', 6)
    
    theta = beta[0] + beta[1]*floor
    
    y = Normal('y', theta, sd=sigma, observed=log_radon)

with pooled_model:
    pooled_trace = sample(i_num_samples)


########################################
# Plots
########################################
b0, m0 = pooled_trace['beta', i_burnin:].mean(axis=0)

plt.figure()
plt.scatter(srrs_mn.floor, np.log(srrs_mn.activity+0.1))
xvals = np.linspace(-0.2, 1.2)
plt.plot(xvals, m0*xvals+b0, 'r--')
plt.savefig('scatter_plot.png', dpi=300)

# pdb.set_trace()

pm.traceplot(pooled_trace)
plt.savefig('trace_plot.png', dpi=300)

pm.summary(pooled_trace).round(2)

