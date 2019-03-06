#!/usr/bin/env python
import json
import numpy as np

peakmjddict = {'SN1993J':49095}
filtdict = {}#'SN2005hk':'ugriz'}

def mkAllFiles():
	import glob
	jsonfiles = glob.glob('*/*.json')
	for f in jsonfiles:
		mkSpec(fname=f,outfile=f.replace('.json','.sed'))
		#mkPhotFile(fname=f,outfile=f.replace('.json','.snana.dat'))
		
def mkSpec(fname='TDe/ASASSN-14ae.json',
		   outfile='TDe/ASASSN-14ae.sed',
		   interp_time = 1,
		   wavelen_space = 5,
		   lamrange = [3000,10000],
		   minlamrange = [4000,8500],
		   mjdrange = None,
		   wavepad=500,peakmjd=None):
	
	with open(fname) as data_file:
		data = json.load(data_file)

	# figure out which epochs have good spectra
	time = []
	for i,j in zip(data[fname.split('.')[0].split('/')[-1]]['spectra'],
				   range(len(data[fname.split('.')[0].split('/')[-1]]['spectra']))):
		w,f = np.array(i['data']).transpose().astype(float)[0:2,:]
		if w[0] > minlamrange[0] or w[-1] < minlamrange[1]: continue 
		time += [float(i['time'])]

	time = np.array(time)

	tstart = min(time)
	tend =	max(time)
	if mjdrange:
		if tstart < mjdrange[0]: tstart = mjdrange[0]
		if tend > mjdrange[1]: tend = mjdrange[1]
	
	wstart = lamrange[0]; wend = lamrange[-1]

	timearray = np.arange(float(tstart),float(tend),interp_time)
	wavearray = np.arange(float(wstart)-wavepad,float(wend)+wavepad,wavelen_space)
	
	# first interpolate in wavelength space
	outspec = np.zeros([len(wavearray),
						len(time)])#data[fname.split('.')[0]]['spectra'])])
	peakflux,peaktime = 0,0
	jcount = 0
	snid = fname.split('.')[0].split('/')[-1]
	if snid in peakmjddict.keys():
		peaktime = peakmjddict[snid]

	for i,j in zip(data[fname.split('.')[0].split('/')[-1]]['spectra'],
				   range(len(data[fname.split('.')[0].split('/')[-1]]['spectra']))):
		if j == 0: mintime = float(i['time'])
		w,f = np.array(i['data']).transpose().astype(float)[0:2,:]
#		if np.max(f) > 1e-15: continue
		
		# hack for SN 2011dh
#		 if max(f) > 1: f *= 1e-15
#		 f *= 1e22
		if w[0] > minlamrange[0] or w[-1] < minlamrange[1]: continue 
		outspec[:,jcount] = np.interp(wavearray,w,f)
		istartwave = np.where(np.abs(wavearray - w[0]) == min(np.abs(wavearray - w[0])))[0]
		iendwave = np.where(np.abs(wavearray - w[-1]) == min(np.abs(wavearray - w[-1])))[0]
		if len(istartwave) > 1: istartwave = [istartwave[0]]
		if len(iendwave) > 1: iendwave = [iendwave[0]]

		if w[0] > 0:
			outspec[0:istartwave[0],jcount] = (f[0]/(w[0] - wavearray[0]))*wavearray[0:istartwave[0]] - f[0]/(w[0] - wavearray[0])*wavearray[0]
		else:
			outspec[0:istartwave[0],jcount] = 0
		if w[-1] > 0:
			outspec[iendwave[0]:,jcount] = -(f[-1]/(wavearray[-1]-w[-1]))*wavearray[iendwave[0]:] + f[-1]/(wavearray[-1]-w[-1])*wavearray[-1]
		else:
			outspec[iendwave[0]:,jcount] = 0
		#outspec[0:wavepad] = 0; outspec[-wavepad:] = 0
		if np.sum(f[(w > 4500) & (w < 6500)]) > peakflux and float(i['time'])-tstart < 100 and snid not in peakmjddict.keys():
			peakflux = np.sum(f[(w > 4500) & (w < 6500)])
			peaktime = float(i['time'])
			
		jcount += 1
		
	# then interpolate in time
	outspect = np.zeros([len(wavearray),len(timearray)+2])
	for h in range(len(outspec[:,0])):
		outspect[h,1:-1] = np.interp(timearray,
									 time,
									 outspec[h,:])
		outspect[h,0] = 0
		outspect[h,-1] = 0
	timearray = np.concatenate((np.array([np.min(timearray)-interp_time]),
								timearray,np.array([np.max(timearray)+interp_time]),))
		
	# print the output
	if not peakmjd: peakmjd = peaktime

	fout = open(outfile,'w')
	print >> fout, '# phase wavelength flux'
	for i in range(np.shape(outspect)[1]):
		for j in range(np.shape(outspect)[0]):
			if timearray[i]-peakmjd < 300:
				print >> fout, '%i %.1f	 %8.5e'%(
					timearray[i]-peakmjd,wavearray[j],outspect[j,i])
	fout.close()

	return()

def mkPhotFile(fname='SLSN/SN2015bh.json',outfile='SLSN/SN2015bh.snana.dat',
			   surveyname='NULL',pkmjd=None,filtlist=None):

	with open(fname) as data_file:
		data = json.load(data_file)

	snid = fname.split('.')[0].split('/')[-1]
	nobs = len(data[snid]['photometry'])
	z = float(data[snid]['redshift'][0]['value'])
	peakdate = data[snid]['maxdate'][0]['value']

	if not pkmjd:
		from astropy.time import Time
		times = ['%sT00:00:00'%peakdate.replace('/','-')]
		t = Time(times,format='isot',scale='utc')
		pkmjd = t.mjd[0]

	if not filtlist and snid not in filtdict.keys():
		filters = ''
		for phot in data[snid]['photometry']:
			if 'band' in phot.keys():
				if phot['band'] not in filters and phot['band'].replace("'","") not in filters and \
				   phot['band'].replace("'","") in ['U','B','V','R','I','u','g','r','i','z']:
					filters += phot['band'].replace("'","")
	elif snid in filtdict.keys():
		filters = filtdict[snid]
	else:
		filters = filtlist
		
	snanahdr = """SNID: %s
SURVEY: %s
FILTERS: %s
REDSHIFT_FINAL: %.4f +- 0.0010
PEAKMJD: %s
	
NOBS: %i
NVAR: 7
VARLIST:  MJD  FLT FIELD   FLUXCAL	 FLUXCALERR	   MAG	   MAGERR
"""%(snid,surveyname,filters,z,pkmjd,nobs)

	fout = open(outfile,'w')
	print >> fout, snanahdr

	for phot in data[snid]['photometry']:
		linefmt = 'OBS: %.1f  %s  NULL	%.3f  %.3f	%.3f  %.3f'
		if 'band' in phot.keys() and 'time' in phot.keys() and phot['band'].replace("'","") in filters:
			if 'system' in phot.keys() and phot['band'] == phot['band'].upper() and len(phot['band']) == 1 and phot['system'] == 'AB':
				print('AHHHHH AB MAG FOR FILTER %s'%phot['band'])
			print >> fout, linefmt%(float(phot['time']),phot['band'].replace("'",""),10**(-0.4*(float(phot['magnitude'])-27.5)),
									0.0,float(phot['magnitude']),0.0)
	print >> fout, 'END: '
	fout.close()

def mkPhotFile(fname='SLSN/SN2015bn.json',outfile='SLSN/SN2015bn.snana.dat',
			   surveyname='NULL',pkmjd=None,filtlist=None):

	with open(fname) as data_file:
		data = json.load(data_file)

	snid = fname.split('.')[0].split('/')[-1]
	nobs = len(data[snid]['photometry'])
	z = float(data[snid]['redshift'][0]['value'])
	peakdate = data[snid]['maxdate'][0]['value']

	if not pkmjd:
		from astropy.time import Time
		times = ['%sT00:00:00'%peakdate.replace('/','-')]
		t = Time(times,format='isot',scale='utc')
		pkmjd = t.mjd[0]

	if not filtlist and snid not in filtdict.keys():
		filters = ''
		for phot in data[snid]['photometry']:
			if 'band' in phot.keys():
				if phot['band'] not in filters and phot['band'].replace("'","") not in filters and \
				   phot['band'].replace("'","") in ['U','B','V','R','I','u','g','r','i','z']:
					filters += phot['band'].replace("'","")
	elif snid in filtdict.keys():
		filters = filtdict[snid]
	else:
		filters = filtlist
		
	snanahdr = """SNID: %s
SURVEY: %s
FILTERS: %s
REDSHIFT_FINAL: %.4f +- 0.0010
PEAKMJD: %s
	
NOBS: %i
NVAR: 7
VARLIST:  MJD  FLT FIELD   FLUXCAL	 FLUXCALERR	   MAG	   MAGERR
"""%(snid,surveyname,filters,z,pkmjd,nobs)

	fout = open(outfile,'w')
	print >> fout, snanahdr

	for phot in data[snid]['photometry']:
		linefmt = 'OBS: %.1f  %s  NULL	%.3f  %.3f	%.3f  %.3f'
		if 'band' in phot.keys() and 'time' in phot.keys() and phot['band'].replace("'","") in filters:
			if 'system' in phot.keys() and phot['band'] == phot['band'].upper() and len(phot['band']) == 1 and phot['system'] == 'AB':
				print('AHHHHH AB MAG FOR FILTER %s'%phot['band'])
			print >> fout, linefmt%(float(phot['time']),phot['band'].replace("'",""),10**(-0.4*(float(phot['magnitude'])-27.5)),
									0.0,float(phot['magnitude']),0.0)
	print >> fout, 'END: '
	fout.close()
	
def mkPhotFile_Err(fname='Ia/SN1572A.json',outfile='/Users/David/Dropbox/research/Tycho/SN1572A.snana.dat',
				   surveyname='NULL',pkmjd=None,filtlist=None):

	with open(fname) as data_file:
		data = json.load(data_file)

	snid = fname.split('.')[0].split('/')[-1]
	nobs = len(data[snid]['photometry'])
	try:
		z = float(data[snid]['redshift'][0]['value'])
	except:
		z = 0
		print('warning : no redshift')
	peakdate = data[snid]['maxdate'][0]['value']

	if not pkmjd:
		from astropy.time import Time
		times = ['%sT00:00:00'%peakdate.replace('/','-')]
		t = Time(times,format='isot',scale='utc')
		pkmjd = t.mjd[0]

	if not filtlist and snid not in filtdict.keys():
		filters = ''
		for phot in data[snid]['photometry']:
			if 'band' in phot.keys():
				if phot['band'] not in filters and phot['band'].replace("'","") not in filters and \
				   phot['band'].replace("'","") in ['U','B','V','R','I','u','g','r','i','z']:
					filters += phot['band'].replace("'","")
	elif snid in filtdict.keys():
		filters = filtdict[snid]
	else:
		filters = filtlist
		
	snanahdr = """SNID: %s
SURVEY: %s
FILTERS: %s
REDSHIFT_FINAL: %.4f +- 0.0010
PEAKMJD: %s
	
NOBS: %i
NVAR: 7
VARLIST:  MJD  MJDERR  FLT FIELD   FLUXCAL	 FLUXCALERR	   MAG	   MAGERR
"""%(snid,surveyname,filters,z,pkmjd,nobs)

	fout = open(outfile,'w')
	print >> fout, snanahdr

	for phot in data[snid]['photometry']:
		linefmt = 'OBS: %.1f  %.1f  %s  NULL	%.3f  %.3f	%.3f  %.3f'
		if 'band' in phot.keys() and 'time' in phot.keys() and phot['band'].replace("'","") in filters:
			if 'system' in phot.keys() and phot['band'] == phot['band'].upper() and len(phot['band']) == 1 and phot['system'] == 'AB':
				print('AHHHHH AB MAG FOR FILTER %s'%phot['band'])
			if 'e_time' in phot.keys() and 'e_magnitude' in phot.keys():
				print >> fout, linefmt%(float(phot['time']),float(phot['e_time']),phot['band'].replace("'",""),10**(-0.4*(float(phot['magnitude'])-27.5)),
										np.abs(float(phot['e_magnitude'])*0.4*np.log(10)*10**(-0.4*(float(phot['magnitude'])-27.5))),float(phot['magnitude']),float(phot['e_magnitude']))
			elif 'e_magnitude' in phot.keys():
				print >> fout, linefmt%(float(phot['time']),0.0,phot['band'].replace("'",""),10**(-0.4*(float(phot['magnitude'])-27.5)),
										np.abs(float(phot['e_magnitude'])*0.4*np.log(10)*10**(-0.4*(float(phot['magnitude'])-27.5))),
										float(phot['magnitude']),float(phot['e_magnitude']))

			elif phot['upperlimit'] == True:
				flux_upperlim = 10**(-0.4*(float(phot['magnitude'])-27.5))
				fluxerr = flux_upperlim/3.
				flux = 0
				
				print >> fout, linefmt%(float(phot['time']),0.0,phot['band'].replace("'",""),flux,fluxerr,
										float(phot['magnitude']),0)


	print >> fout, 'END: '
	fout.close()
	
		
if __name__ == "__main__":
	#mkAllFiles()
	mkSpec()
	#mkPhotFile()
	#mkPhotFile_Err()
