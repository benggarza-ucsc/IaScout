#!/usr/bin/env python
#import pysynphot
import numpy as np
from SNe_Early_Time_Classifier.templates import vegafile,abfile

def synphot(wave,flux,magtype=None,zpoff=0,filtfile=None,
			primarywave=[],primaryflux=[],filtwave=[],filttp=[],
			plot=False,oplot=False,allowneg=False):

	if filtfile:
		mag = zpoff - 2.5 * np.log10( synflux(wave,flux,pb=filtfile,plot=plot,oplot=oplot,
									   allowneg=allowneg))
	elif len(filtwave) and len(filttp):
		mag = zpoff - 2.5 * np.log10( synflux(wave,flux,pbx=filtwave,pby=filttp,plot=plot,oplot=oplot,
											  allowneg=allowneg))
	else:
		raise RuntimeError("filter file or throughput must be defined")

	if magtype:
		if magtype.lower() == 'vega':
			swave,sflux = np.genfromtxt(vegafile,unpack=True)
		elif magtype.lower() == 'ab':
			swave,sflux = np.genfromtxt(abfile,unpack=True)
			
		if len(filtwave) and len(filttp):
			zpt = - 2.5 * np.log10( synflux(swave,sflux,pbx=filtwave,pby=filttp))
		elif filtfile:
			zpt = - 2.5 * np.log10( synflux(swave,sflux,pb=filtfile))
		else:
			raise RuntimeError("filter file or throughput must be defined")

		mag -= zpt

	return(mag)

def synflux(x,spc,pb=None,plot=False,oplot=False,allowneg=False,pbx=[],pby=[]):
	import numpy as np

	nx = len(x)
	pbphot = 1
	if pb:
		pbx,pby = np.loadtxt(pb,unpack=True)
	elif not len(pbx) or not len(pby):
		raise RuntimeError("filter file or throughput must be defined")
		
	npbx = len(pbx)
	if (len(pby) != npbx):
		print(' pbs.wavelength and pbs.response have different sizes')

	if nx == 1 or npbx == 1:
		print('warning! 1-element array passed, returning 0')
		return(spc[0]-spc[0])

	diffx = x[1:nx]-x[0:nx-1]
	diffp = pbx[1:npbx]-pbx[0:npbx-1]

	if (np.min(diffx) <= 0) or (np.min(diffp) <= 0):
		print('passed non-increasing wavelength array')

	g = np.where((x >= pbx[0]) & (x <= pbx[npbx-1]))  # overlap range

	pbspl = np.interp(x[g],pbx,pby)#,kind='cubic')

	if not allowneg:
		pbspl = pbspl
		col = np.where(pbspl < 0)[0]
		if len(col): pbspl[col] = 0

	if (pbphot): pbspl *= x[g]


	res = np.trapz(pbspl*spc[g],x[g])/np.trapz(pbspl,x[g])

	return(res)
