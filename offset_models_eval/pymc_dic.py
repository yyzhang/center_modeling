# this code basically serves as a demonstration on how to calculate DIC values using the pymc deviance.txt
# ## the posteior distribution has to be sampled fine enough for this to work. please check to make sure that params[ind_argmin] is a very small number, compared to the average values of rho_1, r0, tau
# alternatively, one can just code up the DIC reusing the likelihood code

import numpy as np

if __name__ == "__main__":

 ### sorry, hard-wired file name again
 mcmc_file='/home/s1/ynzhang/redmapper_centering/model_fc_June21/temp_combined/dep_lmdexp_exp_newsdss_nobins_noastro_ge20_lt300'
 ###
 rho_1=np.genfromtxt(mcmc_file+'/Chain_0/Rho0.txt')
 r0=np.genfromtxt(mcmc_file+'/Chain_0/R0.txt')
 tau=np.genfromtxt(mcmc_file+'/Chain_0/tau.txt')
 dev=np.genfromtxt(mcmc_file+'/Chain_0/deviance.txt')
 params=((rho_1-np.mean(rho_1) )**2+ (r0-np.mean(r0) )**2 + (tau-np.mean(tau) )**2 )
 ind_argmin=np.argmin(params) ## the posteior distribution has to be sampled fine enough for this to work
 print 'DIC:', mcmc_file, 2*np.mean(dev) - dev[ind_argmin]
