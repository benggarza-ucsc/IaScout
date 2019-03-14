#!/usr/bin/env python

import pylab as plt
import numpy as np

bgspecfile = 'sn91bg-nugent.SED'
bgmanglespecfile = '1999by.SED'
iibspecfile = '2008bo_interpolatedSpectra.txt'
iibmanglespecfile = '2008bo.SED'

def main():
    plt.rcParams['figure.figsize'] = (12,8)
    plt.clf()
    axbglc = plt.axes([0.05,0.55,0.3,0.4])
    axbgspec = plt.axes([0.375,0.55,0.3,0.4])
    axbgmures = plt.axes([0.75,0.55,0.2,0.4])
    axiiblc = plt.axes([0.05,0.07,0.3,0.4])
    axiibspec = plt.axes([0.375,0.07,0.3,0.4])
    axiibmures = plt.axes([0.75,0.07,0.2,0.4])

    pltspec(axbgspec,bgspecfile,bgmanglespecfile,
            filtfiles=['U.dat','B.dat','V.dat','R.dat','I.dat'])
    pltspec(axiibspec,iibspecfile,iibmanglespecfile,
            filtfiles=['keplercam_B.dat','keplercam_V.dat',
                       'keplercam_r.dat','keplercam_i.dat'])
    axiibspec.set_title('SN 2008bo, type IIb')
    axbgspec.set_title('SN 1999by, type Ia-91bg')
    
    pltlc(axbglc,'LOWZ_JRK07_1999by.DAT',snid='1999by')
    pltlc(axiiblc,'2008bo.snana.dat',snid='2008bo')

    pltmures(axbgmures,'ps1phot_prob.fitres',
             'ps1sim_iapec_default.fitres',cctype=42)
    pltmures(axiibmures,'ps1phot_prob.fitres',
             'ps1sim_iapec_default.fitres',cctype=23)
    
    plt.savefig('mangle.png')

def pltmures(ax,datafile,simfile,histmin=-2,histmax=4,
             nbins=20,beta=3.13,alpha=0.147,sigint=0.1,
             dataM=-19.33,simM=-19.36,cctype=23,histvar='MURES'):
    import os
    os.chdir('../snanautil')
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
            color='k',drawstyle='steps-mid',lw=2,label='All Sim. SNe',ls='-.')
    n_nz = np.histogram(sim.__dict__[histvar][cols_CC],bins=np.linspace(histmin,histmax,nbins))
    if cctype == 42: cclabel = 'Ia-91bg'
    elif cctype == 23: cclabel = 'IIb'
    ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
            color='b',drawstyle='steps-mid',lw=2,label=cclabel,ls='--')
    n_nz = np.histogram(sim.__dict__[histvar][cols_Ia],bins=np.linspace(histmin,histmax,nbins))
    ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
            color='r',drawstyle='steps-mid',lw=2,label='Sim. SNe Ia')

    ax.set_yscale('log')
    ax.set_ylim([0.1,1000])
    ax.set_ylabel('#')
    ax.set_xlabel('$\mu - \mu_{\Lambda CDM}$',fontsize='large')
    #ax.legend(numpoints=1,prop={'size':20})
    
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
    
def pltspec(ax,specfile,manglespecfile,filtfiles=[]):
    from txtobj import txtobj
    spec = txtobj(specfile)
    mspec = txtobj(manglespecfile)

    colspec = [np.abs(spec.phase) == min(np.abs(spec.phase))]
    mcolspec = [np.abs(mspec.phase) == min(np.abs(mspec.phase))] 
    wave,flux = spec.wavelength[colspec],spec.flux[colspec]
    flux /= np.max(flux)

    mwave,mflux = mspec.wavelength[mcolspec],mspec.flux[mcolspec]
    mflux /= np.max(mflux)
                   
    ax.plot(wave,flux,color='k',label='original spectrum')
    ax.plot(mwave,mflux,color='b',ls='--',label='warped spectrum')
    ax.legend(prop={'size':12})
    for f,c in zip(filtfiles,['b','g','r','darkorange','cyan']):
        w,tp = np.loadtxt(f,unpack=True)
        ax.plot(w,tp*0.2,color=c)
    ax.set_xlim([3000,9000])
    ax.set_ylim([0,1.2])
    ax.set_xlabel('wavelength',labelpad=0)
    ax.set_ylabel('flux')
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([4000,6000,8000])
    
    return()

def pltlc(ax,lcfile,snid=''):
    from mangle import SuperNova
    from scipy import interpolate 
    from mangle import savitzky_golay
    
    sn = SuperNova(lcfile)
    import copy
    sns = copy.deepcopy(sn)
    sns.tobs = sns.MJD - sns.PEAKMJD
    fluxcalout,tobsout,fltout = \
        np.array([]),np.array([]),np.array([])
    if snid == '1999by': tmin = 5
    elif snid == '2008bo': tmin = 15
    t_interp = np.arange(min(np.unique(sns.tobs))-tmin-5,
                         max(np.unique(sns.tobs))+10,1)
    for f in sns.FILTERS:
        fnc = interpolate.InterpolatedUnivariateSpline(
            sns.tobs[sns.FLT == f],sns.FLUXCAL[sns.FLT == f],
            k=1)
        fluxtmp = fnc(t_interp)
        fluxtmp[(t_interp < min(sns.tobs[sns.FLT == f])-tmin) | 
                (t_interp > max(sns.tobs[sns.FLT == f])+10)] = 0.0
        fnc = interpolate.InterpolatedUnivariateSpline(
            t_interp[(t_interp < min(sns.tobs[sns.FLT == f])-tmin) |
                     (t_interp > min(sns.tobs[sns.FLT == f]))],
            fluxtmp[(t_interp < min(sns.tobs[sns.FLT == f])-tmin) |
                    (t_interp > min(sns.tobs[sns.FLT == f]))],
            k=1)
        fluxtmp = fnc(t_interp)
        fluxtmp = savitzky_golay(fluxtmp,window_size=11)
        fluxtmp[fluxtmp < 0] = 0.0
        fluxcalout = np.append(fluxcalout,fluxtmp)
        tobsout = np.append(tobsout,t_interp)
        fltout = np.append(fltout,[f]*len(t_interp))
    sns.FLUXCAL = fluxcalout[:]
    sns.tobs = tobsout[:]
    sns.FLT = fltout[:]
    
    for f,c in zip(sn.FILTERS,['b','g','r','darkorange','cyan']):
        ax.plot(sns.tobs[sns.FLT == f],
                sns.FLUXCAL[sns.FLT == f],
                'o',
                color=c,alpha=0.2)
        ax.errorbar(sn.MJD[sn.FLT == f]-sn.PEAKMJD,
                    sn.FLUXCAL[sn.FLT == f],
                    yerr=sn.FLUXCALERR[sn.FLT == f],fmt='o',
                    color=c,label=f)
    ax.set_xlim([-20,50])
    ax.legend(numpoints=1,prop={'size':10})
    ax.set_xlabel('phase',labelpad=0)
    ax.set_ylabel('flux')
    if snid == '1999by':
        mures,mureserr,c,cerr,x1,x1err = 0.760,0.392,0.3777,0.0267,-2.553,0.0458
        xtext,ytext = -18,6e5
        ax.text(xtext,ytext,r"""$\mu - \mu_{\Lambda CDM}$ = %s$\pm$%.3f
$C$ = %.3f$\pm$%.3f
$X_1$ = %.3f$\pm$%.3f"""%(mures,mureserr,c,cerr,x1,x1err),bbox={'color':'1.0','alpha':0.75},fontsize=15)

    elif snid == '2008bo':
        mures,mureserr,c,cerr,x1,x1err = 2.872,0.230,0.3277,0.0272,-0.643,0.0555
        mures = '...'
        xtext,ytext = -18,3e4
        ax.text(xtext,ytext,r"""$\mu - \mu_{\Lambda CDM}$ = %s$\pm$%.3f
$C$ = %.3f$\pm$%.3f
$X_1$ = %.3f$\pm$%.3f"""%(mures,mureserr,c,cerr,x1,x1err),bbox={'color':'1.0','alpha':0.75},fontsize=15)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([-20,0,20,40])    
    
if __name__ == "__main__":
    main()
