#!/usr/bin/env python

import pylab as plt
import numpy as np

iipspecfile = 'SDSS-000018.SED'
iinspecfile = 'SDSS-012842.SED'
iilspecfile = 'Nugent+Scolnic_IIL.SED'
ibspecfile = 'CSP-2004gv.SED'
icspecfile = 'SNLS-04D4jv.SED'

iiplcfile = 'SDSS-000018.DAT'
iinlcfile = 'SDSS-012842.DAT'
iillcfile = 'Nugent+Scolnic_IIL.DAT'
iblcfile = 'CSP-2004gv.DAT'
iclcfile = 'SNLS-04D4jv.DAT'


def main():
    plt.rcParams['figure.figsize'] = (10,12)
    plt.clf()
    axiiplc = plt.axes([0.05,0.81,0.3,0.13])
    axiipspec = plt.axes([0.375,0.81,0.3,0.13])
    axiipmures = plt.axes([0.75,0.81,0.2,0.13])
    axiillc = plt.axes([0.05,0.62,0.3,0.13])
    axiilspec = plt.axes([0.375,0.62,0.3,0.13])
    axiilmures = plt.axes([0.75,0.62,0.2,0.13])
    axiinlc = plt.axes([0.05,0.43,0.3,0.13])
    axiinspec = plt.axes([0.375,0.43,0.3,0.13])
    axiinmures = plt.axes([0.75,0.43,0.2,0.13])
    axiblc = plt.axes([0.05,0.24,0.3,0.13])
    axibspec = plt.axes([0.375,0.24,0.3,0.13])
    axibmures = plt.axes([0.75,0.24,0.2,0.13])
    axiclc = plt.axes([0.05,0.05,0.3,0.13])
    axicspec = plt.axes([0.375,0.05,0.3,0.13])
    axicmures = plt.axes([0.75,0.05,0.2,0.13])

    pltspec(axiipspec,iipspecfile)
    pltspec(axiilspec,iilspecfile)
    pltspec(axiinspec,iinspecfile)
    pltspec(axibspec,ibspecfile)
    pltspec(axicspec,icspecfile)

    axiipspec.set_title('SDSS-000018, type II-P')
    axiilspec.set_title('Nugent Template, type II-L')
    axiinspec.set_title('SDSS-012842, type IIn')
    axibspec.set_title('CSP-2004gv, type Ib')
    axicspec.set_title('SNLS-04D4jv, type Ic')
    
    for ax,filename in zip([axiiplc,axiillc,axiinlc,axiblc,axiclc],
                           [iiplcfile,iillcfile,iinlcfile,iblcfile,iclcfile]):
        pltlc(ax,filename)

    for ax,cctype in zip([axiipmures,axiilmures,axiinmures,axibmures,axicmures],
                         [20,21,22,32,33]):
        pltmures(ax,'../ps1phot_prob.fitres',
                 '../ps1sim_default.fitres',cctype=cctype)
    
    plt.savefig('cctmpl.png')

def pltmures(ax,datafile,simfile,histmin=-2,histmax=4,
             nbins=20,beta=3.13,alpha=0.147,sigint=0.1,
             dataM=-19.33,simM=-19.36,cctype=23,histvar='MURES'):
    import os
    from ovdatamc import txtobj
    os.chdir('../')
    from ovdataIIP import salt2mu
    os.chdir('manglefig')

    data = txtobj(datafile)
    sim = txtobj(simfile)

    data.MU,data.MUERR = salt2mu(x1=data.x1,x1err=data.x1ERR,c=data.c,cerr=data.cERR,mb=data.mB,mberr=data.mBERR,
                                 cov_x1_c=data.COV_x1_c,cov_x1_x0=data.COV_x1_x0,cov_c_x0=data.COV_c_x0,
                                 alpha=alpha,beta=beta,
                                 x0=data.x0,sigint=sigint,z=data.zHD,M=dataM)
    from astropy.cosmology import Planck13 as cosmo
    if not data.__dict__.has_key('MURES'):
        data.MURES = data.MU - cosmo.distmod(data.zHD).value

    sim.MU,sim.MUERR = salt2mu(x1=sim.x1,x1err=sim.x1ERR,c=sim.c,cerr=sim.cERR,mb=sim.mB,mberr=sim.mBERR,
                               cov_x1_c=sim.COV_x1_c,cov_x1_x0=sim.COV_x1_x0,cov_c_x0=sim.COV_c_x0,
                               alpha=alpha,beta=beta,
                               x0=sim.x0,sigint=sigint,z=sim.zHD,M=simM)
    from astropy.cosmology import Planck13 as cosmo
    if not sim.__dict__.has_key('MURES'):
        sim.MURES = sim.MU - cosmo.distmod(sim.zHD).value

    sim = mkcuts(sim)
    data = mkcuts(data)

    cols_CC = np.where((sim.SIM_TYPE_INDEX == cctype) & 
                       (sim.__dict__[histvar] >= histmin) &
                       (sim.__dict__[histvar] <= histmax))[0]
    cols_Ia = np.where((sim.SIM_TYPE_INDEX == 1) & 
                       (sim.__dict__[histvar] >= histmin) &
                       (sim.__dict__[histvar] <= histmax))[0]
    lenCC = float(len(cols_CC))
    lenIa = float(len(cols_Ia))


    histint = (histmax - histmin)/nbins
    histlen = float(len(np.where((data.__dict__[histvar] > histmin) &
                                 (data.__dict__[histvar] < histmax))[0]))
    n_nz = np.histogram(data.__dict__[histvar],bins=np.linspace(histmin,histmax,nbins))
    import scipy.stats
    errl,erru = scipy.stats.poisson.interval(0.68,n_nz[0])
    ax.plot(n_nz[1][:-1]+(n_nz[1][1]-n_nz[1][0])/2.,n_nz[0],'o',color='k',lw=2,label='data')
    ax.errorbar(n_nz[1][:-1]+(n_nz[1][1]-n_nz[1][0])/2.,n_nz[0],yerr=[n_nz[0]-errl,erru-n_nz[0]],color='k',fmt=' ',lw=2)

    n_nz = np.histogram(sim.__dict__[histvar],bins=np.linspace(histmin,histmax,nbins))
    ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
            color='k',drawstyle='steps-mid',lw=2,label='Sim. SNe',ls='-.')
    n_nz = np.histogram(sim.__dict__[histvar][cols_CC],bins=np.linspace(histmin,histmax,nbins))
    if cctype == 20: cclabel = 'II-P'
    elif cctype == 21: cclabel = 'IIn'
    elif cctype == 22: cclabel = 'II-L'
    elif cctype == 32: cclabel = 'Ib'
    elif cctype == 33: cclabel = 'Ic'
    ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
            color='b',drawstyle='steps-mid',lw=2,label=cclabel,ls='--')
    n_nz = np.histogram(sim.__dict__[histvar][cols_Ia],bins=np.linspace(histmin,histmax,nbins))
    ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
            color='r',drawstyle='steps-mid',lw=2,label='Sim. Ia')

    ax.set_yscale('log')
    ax.set_ylim([0.1,1000])
    ax.set_ylabel('#')
    ax.set_xlabel('$\mu - \mu_{\Lambda CDM}$',fontsize='large')
#    ax.legend(numpoints=1,prop={'size':9},loc='upper center',
#              ncol=2)
    
def mkcuts(fr,alpha=0.147,beta=3.13,sigint=0.1):
    # uncertainties
    sf = -2.5/(fr.x0*np.log(10.0))
    cov_mb_c = fr.COV_c_x0*sf
    cov_mb_x1 = fr.COV_x1_x0*sf
    
    invvars = 1.0 / (fr.mBERR**2.+ alpha**2. * fr.x1ERR**2. + beta**2. * fr.cERR**2. + \
                     2.0 * alpha * (fr.COV_x1_x0*sf) - 2.0 * beta * (fr.COV_c_x0*sf) - \
                     2.0 * alpha*beta * (fr.COV_x1_c) )
    
    cols = np.where((fr.x1 > -3.0) & (fr.x1 < 3.0) &
                    (fr.c > -0.3) & (fr.c < 0.3) &
                    (fr.x1ERR < 1) & (fr.PKMJDERR < 2*(1+fr.zHD)) &
                    (fr.FITPROB >= 0.001) & (invvars > 0))

    for k in fr.__dict__.keys():
        fr.__dict__[k] = fr.__dict__[k][cols]
    return(fr)
    
def pltspec(ax,specfile):
    from txtobj import txtobj
    print specfile
    spec = txtobj(specfile)

    colspec = np.where(np.abs(spec.phase) == min(np.abs(spec.phase)))[0]
    colspec = colspec[spec.phase[colspec] == np.unique(spec.phase[colspec])[0]]
    wave,flux = spec.wavelength[colspec],spec.flux[colspec]
    flux /= np.max(flux)
    ax.plot(wave,flux,color='k',label='original spectrum')

    ax.set_xlim([3000,9000])
    ax.set_ylim([0,1.2])
    ax.set_xlabel('wavelength',labelpad=0)
    ax.set_ylabel('flux')
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([4000,6000,8000])
    
    return()

def pltlc(ax,lcfile):
    from mangle import SuperNova
    from scipy import interpolate 
    from mangle import savitzky_golay
    
    ph,u,g,r,i,z = np.loadtxt(lcfile,unpack=True,usecols=[1,2,3,4,5,6],skiprows=7)
    
    for f,flt,c in zip([u,g,r,i,z],['u','g','r','i','z'],
                       ['b','g','r','darkorange','cyan']):
        ax.plot(ph,
                10**(-0.4*(f-27.5)),
                'o',
                color=c,label=flt)
    ax.set_xlim([-20,50])
    ax.legend(numpoints=1,prop={'size':10})
    ax.set_xlabel('phase',labelpad=0)
    ax.set_ylabel('flux')

    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([-20,0,20,40])    
    
if __name__ == "__main__":
    main()
