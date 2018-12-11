## This code fits the fiducial exponential + gamma(1) centering+miscentering offset model
##
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatwCDM
from pymc import *
import astropy.io.fits as pyfits

cosmo = FlatwCDM(H0=70, Om0=0.3)

def make_model(r_offset, rlambda):
       # remember to adjust the prior ranges if the posterior values are out of range.
       rho_0=Uniform('Rho0', lower=0.3, upper=1 )
       r0=Uniform('sigma0', lower=0.0001, upper=0.1 )
       tau=Uniform('tau', lower=0.04, upper=0.5 )

       r_rlam=r_offset/rlambda
       @pymc.stochastic(observed=True, plot=False)
       def log_prob(value=0, rho_0=rho_0, r0=r0, tau=tau):
           pr_cor=rho_0*( 1.0/r0*np.exp(-r_rlam/r0) )   
           pr_mis=(1-rho_0)*(r_rlam)*( np.exp(-r_rlam/tau) )/tau**2
           pr=pr_cor+pr_mis
           logpr1=np.log(pr)

           tot_logprob=np.sum(logpr1)
           return tot_logprob
       return locals()


if __name__ == "__main__":

 # read in files
 # I hardwired the column names, file paths, so remember to change them when using your own catalogs.
 ######################3
 cluster= pyfits.open('/home/s1/ynzhang/redmapper_centering/data_dump_June21/RM_SDSS_YUANYUAN-oct11.fits')[1].data

 lambdas=cluster['LAMBDA_OPTICAL']
 zs=cluster['Z_LAMBDA_OPTICAL']
 rdmp_ra=cluster['RM ra']
 rdmp_dec=cluster['RM dec']
 xray_ra=cluster['peak ra']
 xray_dec=cluster['peak dec']
 #####################


 # calculate separations
 centr=SkyCoord(rdmp_ra, rdmp_dec, frame='icrs', unit='deg')
 coord=SkyCoord(xray_ra, xray_dec, frame='icrs', unit='deg')
 sep=centr.separation(coord).arcminute*u.arcmin
 psep=sep*cosmo.kpc_proper_per_arcmin(cluster['Z_LAMBDA_OPTICAL']).to(u.Mpc/u.arcmin)
 roffs=psep.value
 rlmds=(0.01*lambdas)**0.2/0.7

 lmd_bins=[20]
 lmd_bins_max=[1000]
 for ii in np.arange(0, len(lmd_bins)):
    # some additional purging and binning
    lmd_min=lmd_bins[ii]
    lmd_max=lmd_bins_max[ii]
    ind, =np.where( (rlmds>0) & (cluster['Z_LAMBDA_OPTICAL']< 0.35) & (cluster['Z_LAMBDA_OPTICAL'] > 0.1) & (lambdas >= lmd_min) & (lambdas < lmd_max) )
    r_offset=roffs[ind]
    lmd=lambdas[ind]
    rlmd=rlmds[ind]

    # now start running the chain and print out the posterior mean and stdev at the end
    mcmc_file='mcmc_output/XCS_exp_gamma1_lmd%i_lt%i'%(lmd_min, lmd_max)
    M=pymc.Model(make_model(r_offset, rlmd))
    mc=MCMC(M, db='txt', dbname=mcmc_file)
    num=1000 # these numbers should be high enough that the mcmc chains converge
    n_iter=num*1000*10
    n_burn=num*1000*4
    n_thin=num
    mc.sample(iter=n_iter, burn=n_burn, thin=n_thin)
     
    rho_0=np.loadtxt(mcmc_file+'/Chain_0/Rho0.txt')
    r0=np.loadtxt(mcmc_file+'/Chain_0/sigma0.txt')
    tau=np.loadtxt(mcmc_file+'/Chain_0/tau.txt')
    print 'Chain Results rho, sigma, tau: %0.3f +- %0.3f, %0.3f+-%0.3f, %0.4f+-%0.4f'%(np.mean(rho_0), np.std(rho_0), np.mean(r0), np.std(r0), np.mean(tau), np.std(tau))
 
