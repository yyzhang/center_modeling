## this code performs a posterior P-value check and makes a plot similar to Figure 5
## this code has incorporated ideas from Adam Mantz in the repo below
## https://github.com/KIPAC/StatisticalMethods/problems/model_evaluation.ipynb
##
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatwCDM
import numpy.random
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import scipy.stats as st
cosmo = FlatwCDM(H0=70, Om0=0.3)

def func4(nn, rho_0, r0, tau):
    dd_mis=st.gamma.rvs(2, scale=tau, size=nn)
    dd_cen=st.gamma.rvs(1, scale=r0, size=nn)
    r = np.random.sample(nn)
    return np.concatenate((dd_cen[np.where(r <= rho_0)], dd_mis[np.where(r > rho_0)]))

def Test(yy, yycut=0.1):
    return np.float(np.where(yy > yycut)[0].shape[0])/np.float(len(yy))

def p_value_2sided(yy_pp, yy_obs):
    ind1, = np.where(yy_pp > yy_obs)
    ind2, = np.where(yy_pp < yy_obs)
    return np.min([ np.float(len(ind1))/np.float(len(yy_pp)), np.float(len(ind2))/np.float(len(yy_pp))])

if __name__ == "__main__":

 fig, (ax1) = plt.subplots(1, 1)
 
 ### hard wired column names, file paths. Remember to modify   
 cluster=pyfits.open('/home/s1/ynzhang/redmapper_centering/data_dump_June21/SDSS-aug-30-2017-highsnr-peak-sample-noastro.fits')[1].data
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
 ###

 lmd_bins=[20]
 lmd_bins_max=[300]
 for ii in np.arange(0, len(lmd_bins)):
    lmd_min=lmd_bins[ii]
    lmd_max=lmd_bins_max[ii]
    ind, =np.where( (lambdas<lmd_max) & (lambdas>=lmd_min) & (rlmds>0))

    r_offset=roffs[ind]
    lmd=lambdas[ind]
    rlmd=rlmds[ind]

    ## again, hard wired file name. remember to modify
    mcmc_file='/home/s1/ynzhang/redmapper_centering/model_fc_June21/temp_combined/dep_lmdexp_exp_newsdss_nobins_noastro_ge%i_lt%i'%(lmd_min, lmd_max)
    rho_0=np.loadtxt(mcmc_file+'/Chain_0/Rho0.txt')
    r0=np.loadtxt(mcmc_file+'/Chain_0/R0.txt')
    tau=np.loadtxt(mcmc_file+'/Chain_0/tau.txt')

    yy_pp=np.zeros(len(rho_0))
    for jj in range(len(rho_0)):
        res=func4(len(r_offset), rho_0[jj], r0[jj], tau[jj])
        yy_pp[jj]=Test(res, 0.1)
    ax1.hist(yy_pp, color='k', normed=True, alpha=0.2, label=r'Posterior Prediction')
    yy_obs=Test(r_offset/rlmd, 0.1)
    ax1.plot([yy_obs, yy_obs],  [1, 1], 'ks', label=r'Data')
    print 'p-value at 0.1:', p_value_2sided(yy_pp, yy_obs)
    
    yy_pp=np.zeros(len(rho_0))
    for jj in range(len(rho_0)):
        res=func4(len(r_offset), rho_0[jj], r0[jj], tau[jj])
        yy_pp[jj]=Test(res, 0.5)
    ax1.hist(yy_pp, color='b', normed=True, alpha=0.2)
    yy_obs=Test(r_offset/rlmd, 0.5)
    ax1.plot([yy_obs, yy_obs],  [1, 1], 'bs')
    print 'p-value at 0.5:', p_value_2sided(yy_pp, yy_obs)

    yy_pp=np.zeros(len(rho_0))
    for jj in range(len(rho_0)):
        res=func4(len(r_offset), rho_0[jj], r0[jj], tau[jj])
        yy_pp[jj]=Test(res, 1.0)
    ax1.hist(yy_pp, color='r', normed=True, alpha=0.2)
    yy_obs=Test(r_offset/rlmd, 1)
    ax1.plot([yy_obs, yy_obs],  [1, 1], 'rs')
    print 'p-value at 1.0:', p_value_2sided(yy_pp, yy_obs)

 ax1.text(0.02, 60, r'$x_0=1.0$', fontsize=15)
 ax1.text(0.05, 20, r'$x_0=0.5$', fontsize=15)
 ax1.text(0.25, 10, r'$x_0=0.1$', fontsize=15)
 ax1.text(0.5, 20, r'SDSS', fontsize=20)
 ax1.set_xlabel(r'$P(x>x_0)$', fontsize=20)
 fig.text(0.05, 0.5,'Normed Dist.', fontsize=20, va='center', rotation='vertical')
 ax1.set_xlim([0, 0.6])
 ax1.set_ylim([0, 90])
 plt.savefig('PP_check.png')
 plt.show()


