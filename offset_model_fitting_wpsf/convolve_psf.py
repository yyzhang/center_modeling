#############
# this code was developed as a concept to take into account x-ray resolution, so as to further account for different resolutions in different data sets if doing joint analyses.
## In the end, it has been ran, but never made further than just as a prototype.
## some further developments will likely be needed
## the code is much slower than the other ones
############
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatwCDM
from pymc import *
import astropy.io.fits as pyfits
cosmo = FlatwCDM(H0=70, Om0=0.3)

def prob(roffs, rlam, rho_0, r0, tau):
           rda = np.linspace(0, 0.5, 100)
           theta = np.linspace(0, 2.0*np.pi, 50)
           d_rda=rda[1]-rda[0]
           d_theta=theta[1]-theta[0]

           rv, tv, roff=np.meshgrid(rda, theta, roffs)
           # the real mis-centering model
           rconv=np.sqrt(rv**2+roff**2-2.0*roff*rv*np.cos(tv) )

           r_rlam=rconv/rlam
           pr_cor=rho_0*( 1.0/r0*np.exp(-r_rlam/r0) )
           pr_mis=(1-rho_0)*(r_rlam/tau**2)*( np.exp(-r_rlam/tau) )
           pr=pr_cor+pr_mis
           #convolutoin to R_delta, X-ray uncertainties
           p_conv = np.exp(-rv/0.01*1000.0)
           res=np.sum(pr*p_conv)*d_rda*d_theta
           return res

def make_model(r_offset, rlambda):
       rho_0=Uniform('Rho0', lower=0.5, upper=1 )
       r0=Uniform('R0', lower=0.0001, upper=0.1 )
       tau=Uniform('tau', lower=0.1, upper=1.0 )

       r_rlam=r_offset/rlambda
       @pymc.stochastic(observed=True, plot=True)
       def log_prob(value=0, rho_0=rho_0, r0=r0, tau=tau):
           pr=np.zeros(len(r_offset))
           for ii in range(len(r_offset)):
               offsets=np.linspace(0, 10, 800)
               d_off=offsets[1]-offsets[0]
               prs=prob(offsets, rlambda[ii], rho_0, r0, tau)
               tot_prs=np.sum(prs)*d_off
               pr[ii]=prob(r_offset[ii], rlambda[ii], rho_0, r0, tau)/tot_prs
           logpr=np.log(pr)
           tot_logprob=np.sum(logpr)
           return tot_logprob
       return locals()

if __name__ == "__main__":

 cluster=pyfits.open('/home/s1/ynzhang/redmapper_centering/data_dump_June21/SDSS-June-2-2017-peak-sample-only_wpeak_offset_crop_forYY.fits')[1].data
 lambdas=cluster['lambda']
 rdmp_ra=cluster['redMaPPer_ra']
 rdmp_dec=cluster['redMaPPer_dec']
 xray_ra=cluster['x_ray_peak_ra']
 xray_dec=cluster['x_ray_peak_dec']

 centr=SkyCoord(rdmp_ra, rdmp_dec, frame='icrs', unit='deg')
 coord=SkyCoord(xray_ra, xray_dec, frame='icrs', unit='deg')
 sep=centr.separation(coord).arcminute*u.arcmin
 psep=sep*cosmo.kpc_proper_per_arcmin(cluster['Redshift']).to(u.Mpc/u.arcmin)
 roffs=psep.value

 rlmds=(0.01*lambdas)**0.2/0.7
 lmd_bins=[20]
 lmd_bins_max=[300]
 for ii in range(1):
    lmd_min=lmd_bins[ii]
    lmd_max=lmd_bins_max[ii]
    ind, =np.where( (lambdas<lmd_max) & (lambdas>=lmd_min) & (rlmds>0))

    r_offset=roffs[ind]
    lmd=lambdas[ind]
    rlmd=rlmds[ind]

    mcmc_file='mcmc_output/psfconvolve_ge%i_lt%i'%(lmd_min, lmd_max)
    M=pymc.Model(make_model(r_offset, rlmd))
    mc=MCMC(M, db='txt', dbname=mcmc_file)
    num=5000
    n_iter=num*1000*5
    n_burn=num*1000*1
    n_thin=num
    mc.sample(iter=n_iter, burn=n_burn, thin=n_thin)
    
    rho_0=np.loadtxt(mcmc_file+'/Chain_0/Rho0.txt')
    r0=np.loadtxt(mcmc_file+'/Chain_0/R0.txt')
    tau=np.loadtxt(mcmc_file+'/Chain_0/tau.txt')
    print '%0.3f +- %0.3f, %0.3f+-%0.3f, %0.4f+-%0.4f'%(np.mean(rho_0), np.std(rho_0), np.mean(r0), np.std(r0), np.mean(tau), np.std(tau))
