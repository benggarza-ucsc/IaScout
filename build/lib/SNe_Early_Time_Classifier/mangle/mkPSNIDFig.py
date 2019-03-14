#!/usr/bin/env python

def main(datafile='../ps1phot_prob.fitres'):
    import pylab as plt
    import numpy as np
    from ovdatamc import txtobj
    from ovdataIIP import salt2mu
    
    plt.clf()
    ax3 = plt.axes([0.1,0.07,0.8,0.25])
    ax2 = plt.axes([0.1,0.39,0.8,0.25])
    ax1 = plt.axes([0.1,0.70,0.8,0.25])
    
    data = txtobj(datafile)
    data.MU,data.MUERR = salt2mu(x1=data.x1,x1err=data.x1ERR,c=data.c,cerr=data.cERR,mb=data.mB,mberr=data.mBERR,
                                 cov_x1_c=data.COV_x1_c,cov_x1_x0=data.COV_x1_x0,cov_c_x0=data.COV_c_x0,
                                 alpha=0.147,beta=3.13,
                                 x0=data.x0,sigint=0.1,z=data.zHD,M=-19.36)
    from astropy.cosmology import Planck13 as cosmo
    if not data.__dict__.has_key('MURES'):
        data.MURES = data.MU - cosmo.distmod(data.zHD).value

    cols = np.where((data.x1 > -3.0) & (data.x1 < 3.0) &
                    (data.c > -0.3) & (data.c < 0.3) &
                    (data.x1ERR < 1) & (data.PKMJDERR < 2*(1+data.zHD)) &
                    (data.FITPROB >= 0.001))
    #for k in data.__dict__.keys():
    #    data.__dict__[k] = data.__dict__[k][cols]
    
    for ax,title,color in zip([ax1,ax2,ax3],['P(SN Ia) > 0.95','P(SN Ib/c) > 0.95','P(SN II) > 0.95'],['r','g','b']):
        if 'Ia' in title: cols = [(data.PBAYES_Ia > 0.95)]
        if 'Ib' in title: cols = [(data.PBAYES_Ibc > 0.95)]
        if 'II' in title: cols = [(data.PBAYES_II > 0.95)]
        #cols = range(len(data.CID))
        
        ax.set_ylabel('$\mu$',fontsize=15)
        z = np.arange(0,1,0.001)
        ax.plot(z,cosmo.distmod(z).value,color='k')
        ax.errorbar(data.zHD[cols],data.MU[cols],
                    yerr=data.MUERR[cols],fmt='o',color=color)
        ax.set_title(title)
        ax.set_xlim([0,0.7])
        ax.set_ylim([36,45])
        
    ax3.set_xlabel('$z$',fontsize=20)
        
    return()
