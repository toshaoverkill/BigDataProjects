import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

races = ['asian', 'black', 'hispanic', 'white', 'other']
voter_race = np.random.choice(a=races, p=[0.05, 0.15, 0.25, 0.05, 0.5], size=1000)
voter_age = sts.poisson.rvs(loc=18, mu=30, size=1000)
print(type(voter_age), type(voter_race))
