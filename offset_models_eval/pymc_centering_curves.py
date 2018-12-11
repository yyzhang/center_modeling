## this code plots the offset distribution, and overplots the posterior models.
## it makes a Figure 3 like plot
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatwCDM
import astropy.io.fits as pyfits
cosmo = FlatwCDM(H0=70, Om0=0.3)

## some of the models were coded up for testing purposes, but only func4 was used in the end.
def func(xx, rho_0, r0, tau):
    # the gaussian+Rayleigh cent/miscent model
    pr_cor=rho_0*( 1.0/r0/np.sqrt(2.0*np.pi)*np.exp(-(xx/r0)**2*0.5 ) )
    pr_mis=(1-rho_0)*(xx/tau**2)*( np.exp(-0.5*xx**2/tau**2) )
    pr=pr_cor+pr_mis
    return pr, pr_cor, pr_mis

def func2(xx, rho_0, r0, tau):
    # the two gaussians model with the second width fixed 
    pr_cor=rho_0*( 1.0/r0/np.sqrt(2.0*np.pi)*np.exp(-(xx/r0)**2*0.5 ) )
    pr_mis=(1-rho_0)*( 1.0/0.329/np.sqrt(2.0*np.pi)*np.exp(-(xx/0.329)**2*0.5 ) )
    pr=pr_cor+pr_mis
    return pr, pr_cor, pr_mis

def func3(xx, rho_0, r0, tau):
    ## gaussian + gaussian
    pr_cor=rho_0*( 1.0/r0/np.sqrt(2.0*np.pi)*np.exp(-(xx/r0)**2*0.5 ) )
    pr_mis=(1-rho_0)*( 1.0/tau/np.sqrt(2.0*np.pi)*np.exp(-(xx/tau)**2*0.5 ) )
    pr=pr_cor+pr_mis
    return pr, pr_cor, pr_mis

def func4(xx, rho_0, r0, tau):
    ## the fiducial exp+gamma(1) model
    pr_cor=rho_0*( 1.0/r0*np.exp(-xx/r0) )
    pr_mis=(1-rho_0)*(xx)*( np.exp(-xx/tau) )/tau**2
    pr=pr_mis+pr_cor
    return pr, pr_cor, pr_mis

def func5(xx, rho_0, r0, tau):
    ## Rayleigh+Rayleigh model
    pr_mis=(1-rho_0)*(xx/tau**2)*( np.exp(-0.5*xx**2/tau**2) )
    pr_cor=rho_0*(xx/r0**2)*( np.exp(-0.5*xx**2/r0**2) )
    pr=pr_mis+pr_cor
    return pr, pr_cor, pr_mis

if __name__ == "__main__":

 # setting up the "canvas" for the plot, including a inset figure.
 fig, ax1 = plt.subplots()
 # These are in unitless percentages of the figure size. (0,0 is bottom left)
 left, bottom, width, height = [0.32, 0.3, 0.5, 0.3]
 ax2 = fig.add_axes([left, bottom, width, height])

 ## hardwired column names and file paths here. Remember to change.
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
 ######################

 lmd_bins=[20]
 lmd_bins_max=[300]
 for ii in np.arange(0, len(lmd_bins)):
    lmd_min=lmd_bins[ii]
    lmd_max=lmd_bins_max[ii]
    ind, =np.where( (lambdas<lmd_max) & (lambdas>=lmd_min) & (rlmds>0))

    r_offset=roffs[ind]
    lmd=lambdas[ind]
    rlmd=rlmds[ind]

    # overplotting the posterior model constraints, and sample the model uncertainties
    mcmc_file='/home/s1/ynzhang/redmapper_centering/model_fc_June21/temp_combined/dep_lmdexp_exp_newsdss_nobins_noastro_ge%i_lt%i'%(lmd_min, lmd_max) ## again hardwired file name
    rho_0=np.loadtxt(mcmc_file+'/Chain_0/Rho0.txt')
    r0=np.loadtxt(mcmc_file+'/Chain_0/R0.txt')
    tau=np.loadtxt(mcmc_file+'/Chain_0/tau.txt')

    xx=np.arange(0.02, 1.5, 0.001)
    yy_arr=np.zeros([len(xx), len(rho_0)])
    yy_mea=np.zeros(len(xx))
    yy_std=np.zeros(len(xx))
    yy_arr1=np.zeros([len(xx), len(rho_0)])
    yy_mea1=np.zeros(len(xx))
    yy_std1=np.zeros(len(xx))
    yy_arr2=np.zeros([len(xx), len(rho_0)])
    yy_mea2=np.zeros(len(xx))
    yy_std2=np.zeros(len(xx))
    for jj in range(len(rho_0)):
        res, res1, res2=func4(xx, rho_0[jj], r0[jj], tau[jj])
        yy_arr[:, jj]=res
        yy_arr1[:, jj]=res1
        yy_arr2[:, jj]=res2
    for kk in range(len(xx)):
        yy_mea[kk]=np.mean(yy_arr[kk, :])
        yy_std[kk]=np.std(yy_arr[kk, :])
        yy_mea1[kk]=np.mean(yy_arr1[kk, :])
        yy_std1[kk]=np.std(yy_arr1[kk, :])
        yy_mea2[kk]=np.mean(yy_arr2[kk, :])
        yy_std2[kk]=np.std(yy_arr2[kk, :])
    
    #plotting stuff
    ax1.hist(r_offset/rlmd, bins=20, range=[0, 1.5], color='k', normed=True, alpha=0.2, label=r'SDSS')
    ax2.hist(r_offset/rlmd, bins=20, range=[0, 1.5], color='k', normed=True, alpha=0.2)
 

    ax1.plot(xx, yy_mea1, 'k', label='Well-Centered Model')
    ax1.fill_between(xx, yy_mea1-yy_std1, yy_mea1+yy_std1, color='k', alpha=0.5)
    ax1.plot(xx, yy_mea2, 'r:', label='Mis-Centered Model')
    ax1.fill_between(xx, yy_mea2-yy_std2, yy_mea2+yy_std2, color='r', alpha=0.3)
    ax2.plot(xx, yy_mea1, 'k')
    ax2.fill_between(xx, yy_mea1-yy_std1, yy_mea1+yy_std1, color='k', alpha=0.5)
    ax2.plot(xx, yy_mea2, 'r:')
    ax2.fill_between(xx, yy_mea2-yy_std2, yy_mea2+yy_std2, color='r', alpha=0.3)


 ax2.set_xlim([0.05, 1.4])
 ax2.set_ylim([0, 2])
 ax1.legend(loc=1)
 ax1.set_xlabel(r'$r_\mathrm{offset}/R_\lambda$', fontsize=25)
 ax1.set_ylabel(r'Normed Dist.', fontsize=20)
 plt.savefig('fit_curves.png')
 plt.show()


