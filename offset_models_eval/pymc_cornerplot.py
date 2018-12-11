## this code makes a corner plot with the pymc posterior model parameter constraints.
## it makes a figure that looks like Figure 4.

import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

if __name__ == "__main__":


 mcmc_file='/home/s1/ynzhang/redmapper_centering/model_fc_June21/temp_combined/dep_lmdexp_exp_newsdss_nobins_noastro_ge20_lt300'
 rho_1=np.genfromtxt(mcmc_file+'/Chain_0/Rho0.txt')
 r0=np.genfromtxt(mcmc_file+'/Chain_0/R0.txt')
 tau=np.genfromtxt(mcmc_file+'/Chain_0/tau.txt')

 c = ChainConsumer()
 data=np.vstack( (rho_1, r0, tau) ).T
 c.add_chain( data, parameters=[r"$\rho$", r"$\sigma$", r"$\tau$"], name='SDSS')

 mcmc_file='/home/s1/ynzhang/redmapper_centering/model_fc_June21/temp_combined/dep_lmdexp_exp_newy1a1_nobins_noastro_ge20_lt300'
 rho_1=np.genfromtxt(mcmc_file+'/Chain_0/Rho0.txt')
 r0=np.genfromtxt(mcmc_file+'/Chain_0/R0.txt')
 tau=np.genfromtxt(mcmc_file+'/Chain_0/tau.txt')

 data=np.vstack( (rho_1, r0, tau) ).T
 c.add_chain( data, parameters=[r"$\rho$", r"$\sigma$", r"$\tau$"], name='DES')

 c.configure(colors=['k', 'b'], linestyles=["-", "--"], shade=[True, False], shade_alpha=[0.5, 0.0])
 c.plot(filename="cornerplot.png", figsize="column")
