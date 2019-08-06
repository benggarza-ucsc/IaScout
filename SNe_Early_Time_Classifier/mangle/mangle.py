#!/usr/bin/env python
# D. Jones - 2/25/16
import os
import numpy as np
from scipy.optimize import least_squares
import george
from george import kernels
from scipy.optimize import minimize
import emcee

class txtobj:
	def __init__(self,filename):
		import numpy as np
		fin = open(filename,'r')
		lines = fin.readlines()
		for l in lines:
			if l.startswith('#'):
				l = l.replace('\n','')
				coldefs = l.replace('#','').split()
				break

		with open(filename) as f:
			reader = [x.split() for x in f if not x.startswith('#')]

		i = 0
		for column in zip(*reader):
			try:
				self.__dict__[coldefs[i]] = np.array(column[:]).astype(float)
			except:
				self.__dict__[coldefs[i]] = np.array(column[:])
			i += 1
	def writeto(self,outfilename,verbose=False):
		fout = open(outfilename,'w')
		headerstr = '# '
		for k in list(self.__dict__.keys()):
			headerstr += '%s '%k
		print(headerstr[:-1], file=fout)

		for i in range(len(self.__dict__[k])):
			line = ''
			for k in list(self.__dict__.keys()):
				line += '%s '%self.__dict__[k][i]
			print(line[:-1], file=fout)
		fout.close()
		if verbose: print(('Wrote object to file %s'%outfilename))
		return(outfilename)

class SuperNova( object ) :
	""" object class for a single SN extracted from SNANA sim tables
		or from a SNANA-style .DAT file.  From S. Rodney.
	"""

	def __init__( self, datfile=None, simname=None, snid=None, verbose=False ) :
		""" Read in header info (z,type,etc) and full light curve data.
		For simulated SNe stored in fits tables, user must provide the simname and snid,
		and the data are collected from binary fits tables, assumed to exist
		within the $SNDATA_ROOT/SIM/ directory tree.
		For observed or simulated SNe stored in ascii .dat files, user must provide
		the full path to the datfile.
		"""
		if not (datfile or (snid and simname)) :
			if verbose:	 print("No datfile or simname provided. Returning an empty SuperNova object.")

		if simname and snid :
			if verbose : print(("Reading in data from binary fits tables for %s %s"%(simname, str(snid))))
			self.simname = simname
			self.snid = snid
			# read in header and light curve data from binary fits tables
			gothd = self.getheadfits( )
			gotphot = self.getphotfits( )
			if not (gothd and gotphot) :
				gotgrid = self.getgridfits()
				if not gotgrid :
					print(("Unable to read in data for %s %s.  No sim product .fits files found."%(simname, str(snid))))
		elif datfile :
			if verbose : print(("Reading in data from light curve file %s"%(datfile)))
			self.readdatfile( datfile )

		# Load in the survey data from constants.py	 (default to HST)
		if 'SURVEY' not in self.__dict__ :
			print('Warning : Survey information is required!')

	@property
	def name(self):
		if 'NAME' in self.__dict__ :
			return( self.NAME )
		elif 'SNID' in self.__dict__ :
			return( self.SNID )
		elif 'NICKNAME' in self.__dict__ :
			return( self.NICKNAME )
		elif 'CID' in self.__dict__ :
			return( self.CID )
		elif 'IAUNAME' in self.__dict__ :
			return( self.IAUNAME )
		else :
			return( '' )

	@property
	def nickname(self):
		if 'NICKNAME' in self.__dict__ :
			return( self.NICKNAME )
		elif 'NAME' in self.__dict__ :
			return( self.NAME )
		elif 'IAUNAME' in self.__dict__ :
			return( self.IAUNAME )
		elif 'SNID' in self.__dict__ :
			return( self.SNID )
		elif 'CID' in self.__dict__ :
			return( self.CID )
		else :
			return( '' )

	@property
	def bandlist(self):
		if 'FLT' in self.__dict__ :
			return( np.unique( self.FLT ) )
		else :
			return( np.array([]))

	@property
	def bands(self):
		return( ''.join(self.bandlist) )

	@property
	def BANDORDER(self):
		return( self.SURVEYDATA.BANDORDER )

	@property
	def bandorder(self):
		return( self.SURVEYDATA.BANDORDER )

	@property
	def signoise(self):
		""" compute the signal to noise curve"""
		if( 'FLUXCALERR' in self.__dict__ and
			'FLUXCAL' in self.__dict__	) :
			return( self.FLUXCAL / np.abs(self.FLUXCALERR) )
		else:
			return( None)

	def pkmjd(self):
		if 'PEAKMJD' in list(self.__dict__.keys()) :
			return( self.PEAKMJD )
		elif 'SIM_PEAKMJD' in list(self.__dict__.keys()) :
			return( self.SIM_PEAKMJD )
		else :
			return( self.pkmjdobs )

	@property
	def pkmjderr(self):
		if 'PEAKMJDERR' in list(self.__dict__.keys()) :
			return( self.PEAKMJDERR )
		elif 'SIM_PEAKMJDERR' in list(self.__dict__.keys()) :
			return( self.SIM_PEAKMJDERR )
		elif 'SIM_PEAKMJD_ERR' in list(self.__dict__.keys()) :
			return( self.SIM_PEAKMJD_ERR )
		else :
			return( max( self.pkmjdobserr, 1.2*abs(self.pkmjdobs-self.pkmjd)  ) )

	@property
	def pkmjdobs(self):
		if 'SEARCH_PEAKMJD' in list(self.__dict__.keys()) :
			return( self.SEARCH_PEAKMJD )
		elif 'SIM_PEAKMJD' in list(self.__dict__.keys()) :
			return( self.SIM_PEAKMJD )
		else :
			# crude guess at the peak mjd as the date of highest S/N
			return( self.MJD[ self.signoise.argmax() ] )

	@property
	def pkmjdobserr(self):
		if 'SEARCH_PEAKMJDERR' in list(self.__dict__.keys()) :
			return( self.SEARCH_PEAKMJDERR )
		elif 'SEARCH_PEAKMJD_ERR' in list(self.__dict__.keys()) :
			return( self.SEARCH_PEAKMJD_ERR )
		else :
			# determine the peak mjd uncertainty
			ipk = self.signoise.argmax()
			pkband = self.FLT[ ipk ]
			ipkband = np.where(self.FLT==pkband)[0]
			mjdpkband = np.array( sorted( self.MJD[ ipkband ] ) )
			if len(ipkband)<2 : return( 30 )
			ipkidx = ipkband.tolist().index( ipk )
			if ipkidx == 0 :
				return( 0.7*(mjdpkband[1]-mjdpkband[0]) )
			elif ipkidx == len(ipkband)-1 :
				return( 0.7*(mjdpkband[-1]-mjdpkband[-2]) )
			else :
				return( 0.7*0.5*(mjdpkband[ipkidx+1]-mjdpkband[ipkidx-1]) )

	@property
	def mjdpk(self):
		return( self.pkmjd )

	@property
	def mjdpkerr(self):
		return( self.pkmjderr )

	@property
	def mjdpkobs(self):
		return( self.pkmjdobs )

	@property
	def mjdpkobserr(self):
		return( self.pkmjdobserr )

	@property
	def isdecliner(self):
		if 'DECLINER' in self.__dict__ :
			if self.DECLINER in ['True','TRUE',1] : return( True )
			else : return( False )
		if self.pkmjd < self.MJD.min() : return( True )
		else : return( False )

	@property
	def zphot(self):
		zphot = None
		for key in ['HOST_GALAXY_PHOTO-Z','ZPHOT']:
			if key in list(self.__dict__.keys()) :
				hostphotz = self.__dict__[key]
				if type( hostphotz ) == str :
					zphot = float(hostphotz.split()[0])
					break
				else :
					zphot = float(hostphotz)
					break
		if zphot>0 : return( zphot )
		else : return( 0 )

	@property
	def zphoterr(self):
		zphoterr=None
		for key in ['HOST_GALAXY_PHOTO-Z_ERR','ZPHOTERR']:
			if key in list(self.__dict__.keys()) :
				zphoterr = float(self.__dict__[key])
				break
		if not zphoterr :
			for key in ['HOST_GALAXY_PHOTO-Z',]:
				if key in list(self.__dict__.keys()) :
					hostphotz = self.__dict__[key]
					if type( hostphotz ) == str :
						zphoterr = float(hostphotz.split()[2])
						break
		if zphoterr>0 : return( zphoterr )
		else : return( 0 )

	@property
	def zspec(self):
		zspec=None
		for key in ['HOST_GALAXY_SPEC-Z','SN_SPEC-Z','ZSPEC']:
			if key in list(self.__dict__.keys()) :
				hostspecz = self.__dict__[key]
				if type( hostspecz ) == str :
					zspec = float(hostspecz.split()[0])
					break
				else :
					zspec = float(hostspecz)
					break
		if zspec>0 : return( zspec )
		else : return( 0 )

	def zspecerr(self):
		zspecerr=None
		for key in ['HOST_GALAXY_SPEC-Z_ERR','SN_SPEC-Z_ERR','ZSPECERR']:
			if key in list(self.__dict__.keys()) :
				zspecerr = float(self.__dict__[key])
				break
		if not zspecerr :
			for key in ['HOST_GALAXY_SPEC-Z','SN_SPEC-Z']:
				if key in list(self.__dict__.keys()) :
					specz = self.__dict__[key]
					if type( specz ) == str :
						zspecerr = float(specz.split()[2])
						break
		if zspecerr>0 : return( zspecerr )
		else : return( 0 )

	@property
	def z(self):
		zfin = None
		if 'REDSHIFT_FINAL' in self.__dict__ :
			if type( self.REDSHIFT_FINAL ) == str :
				zfin = float(self.REDSHIFT_FINAL.split()[0])
			else :
				zfin = float(self.REDSHIFT_FINAL)
			if zfin > 0 : return( zfin )
		elif 'REDSHIFT' in self.__dict__ : return( self.REDSHIFT )
		elif self.zspec > 0 : return( self.zspec )
		elif self.zphot > 0 : return( self.zphot )
		elif 'SIM_REDSHIFT' in self.__dict__ : return( self.SIM_REDSHIFT )
		else : return( 0 )

	@property
	def zerr(self):
		# TODO : better discrimination of possible redshift labels
		if ( 'REDSHIFT_FINAL' in list(self.__dict__.keys()) and
			 type( self.REDSHIFT_FINAL ) == str ):
			return( float(self.REDSHIFT_FINAL.split()[2]))
		elif ( 'REDSHIFT_ERR' in list(self.__dict__.keys()) ):
			if type( self.REDSHIFT_ERR ) == str :
				return( float(self.REDSHIFT_ERR.split()[0]) )
			else :
				return(self.REDSHIFT_ERR)
		if self.zspecerr > 0 : return( self.zspecerr )
		elif self.zphoterr > 0 : return( self.zphoterr )
		else : return( 0 )

	def nobs(self):
		return( len(self.FLUXCAL) )

	@property
	def chi2_ndof(self):
		""" The reduced chi2.
		!! valid only for models that have been fit to observed data !!
		"""
		if 'CHI2VEC' in self.__dict__ and 'NDOF' in self.__dict__ :
			return( self.CHI2VEC.sum() / self.NDOF )
		elif 'CHI2' in self.__dict__ and 'NDOF' in self.__dict__ :
			return( self.CHI2 / self.NDOF )
		else :
			return( 0 )

	@property
	def chi2(self):
		""" The raw (unreduced) chi2.
		!! valid only for models that have been fit to observed data !!
		"""
		if 'CHI2VEC' in self.__dict__  :
			return( self.CHI2VEC.sum() )
		elif 'CHI2' in self.__dict__  :
			return( self.CHI2 )
		else :
			return( 0 )

	def readdatfile(self, datfile ):
		""" read the light curve data from the SNANA-style .dat file.
		Metadata in the header are in "key: value" pairs
		Observation data lines are marked with OBS:
		and column names are given in the VARLIST: row.
		Comments are marked with #.
		"""
		# TODO : could make the data reading more general: instead of assuming the 6 known
		#	columns, just iterate over the varlist.

		from numpy import array,log10,unique,where

		if not os.path.isfile(datfile): raise RuntimeError( "%s does not exist."%datfile)
		self.datfile = os.path.abspath(datfile)
		fin = open(datfile,'r')
		data = fin.readlines()
		fin.close()
		flt,mjd=[],[]
		fluxcal,fluxcalerr=[],[]
		mag,magerr=[],[]

		# read header data and observation data
		for i in range(len(data)):
			line = data[i]
			if(len(line.strip())==0) : continue
			if line.startswith("#") : continue
			if line.startswith('END:') : break
			if line.startswith('VARLIST:'):
				colnames = line.split()[1:]
				for col in colnames :
					self.__dict__[col] = []
			elif line.startswith('NOBS:'):
				nobslines = int(line.split()[1])
			elif line.startswith('NVAR:'):
				ncol = int(line.split()[1])
			elif 'OBS:' in line :
				obsdat = line.split()[1:]
				for col in colnames :
					icol = colnames.index(col)
					self.__dict__[col].append( str2num(obsdat[icol]) )
			else :
				colon = line.find(':')
				key = line[:colon].strip()
				val = line[colon+1:].strip()
				self.__dict__[ key ] = str2num(val)

		for col in colnames :
			self.__dict__[col] = array( self.__dict__[col] )
		return( None )

	def writedatfile(self, datfile, mag2fluxcal=False, **kwarg ):
		""" write the light curve data into a SNANA-style .dat file.
		Metadata in the header are in "key: value" pairs
		Observation data lines are marked with OBS:
		and column names are given in the VARLIST: row.
		Comments are marked with #.

		mag2fluxcal : convert magnitudes and errors into fluxcal units
		  and update the fluxcal and fluxcalerr arrays before writing
		"""
		from numpy import array,log10,unique,where

		if mag2fluxcal :
			from .. import hstsnphot
			self.FLUXCAL, self.FLUXCALERR = hstsnphot.mag2fluxcal( self.MAG, self.MAGERR )

		fout = open(datfile,'w')
		for key in ['SURVEY','NICKNAME','SNID','IAUC','PHOTOMETRY_VERSION',
					'SNTYPE','FILTERS','MAGTYPE','MAGREF','DECLINER',
					'RA','DECL','MWEBV','REDSHIFT_FINAL',
					'HOST_GALAXY_PHOTO-Z','HOST_GALAXY_SPEC-Z','REDSHIFT_STATUS',
					'SEARCH_PEAKMJD','SEARCH_PEAKMJDERR',
					'PEAKMJD','PEAKMJDERR',
					'SIM_SALT2c','SIM_SALT2x1','SIM_SALT2mB','SIM_SALT2alpha',
					'SIM_SALT2beta','SIM_REDSHIFT','SIM_PEAKMJD'] :
			if key in kwarg :
				print('%s: %s'%(key,str(kwarg[key])), file=fout)
			elif key in self.__dict__ :
				print('%s: %s'%(key,str(self.__dict__[key])), file=fout)
		print('\nNOBS: %i'%len(self.MAG), file=fout)
		print('NVAR: 7', file=fout)
		print('VARLIST:  MJD	FLT FIELD	FLUXCAL	  FLUXCALERR	MAG		MAGERR\n', file=fout)

		for i in range(self.nobs):
			print('OBS: %9.3f  %s  %s %8.3f %8.3f %8.3f %8.3f'%(
				self.MJD[i], self.FLT[i], self.FIELD[i], self.FLUXCAL[i],
				self.FLUXCALERR[i], self.MAG[i], self.MAGERR[i] ), file=fout)
		print('\nEND:', file=fout)
		fout.close()
		return( datfile )

class mangle:
	def __init__(self):
		self.clobber = False
		self.verbose = False

	def add_options(self, parser=None, usage=None, config=None):
		import optparse
		if parser == None:
			parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

		# The basics
		parser.add_option('-v', '--verbose', action="count", dest="verbose",default=1)
		parser.add_option('--clobber', default=False, action="store_true",
						  help='overwrite output file if it exists')

		parser.add_option('--mkplot', default=False, action="store_true",
						  help='make mangling plots')
		parser.add_option('--workdir',type='string',default='workdir',
						  help='parameter file')
		parser.add_option('--outdir',type='string',default='youngsn',
						  help='parameter file')
		parser.add_option('--datadir',type='string',default='kessler/SMOOTH_INTERP_v3',
						  help='parameter file')
		parser.add_option('--filtpath',type='string',default='$SNDATA_ROOT/filters',
						  help='filter set')
		parser.add_option('--sntemp',type='string',default='snIc_flux.v1.3b.txt',
						  help='SN template file')
		parser.add_option('--snid',type='string',default='CSP-2004fe',
						  help='SN ID to mangle (must be in SN info file)')
		parser.add_option('--smoothfluxfile',type='string',default='',
						  help='file with smoothed flux info (if not provided, default = <workdir>/<SN ID>-smooth.flux)')
		parser.add_option('--niter',type='int',default=9,
						  help='number of iterations')
		parser.add_option('--scale',type='float',default=1,
						  help='scale spectrum flux by this to speed up the mangling')
		parser.add_option('--smoothtobsrange',type='float',default=(-30,100),
						  help='min/max tobs to use in generating smoothed template',nargs=2)
		parser.add_option('--tobsrange',type='int',default=(-45,100),
						  help='min/max tobs to use in mangling',nargs=2)
		parser.add_option('--primary',type='string',default='$SNDATA_ROOT/standards/flatnu.dat',
						  help='primary')
		parser.add_option('--tol',type='float',default=10,
						  help='minimization tolerance.	 This needs to be high or it takes forever!')
		parser.add_option('--smoothlc',action='store_true',default=False,
						  help='if set, smooth the light curve before mangling')
		parser.add_option('--smoothfunc',default='mcmc',type='string',
						  help='function for smoothing (mcmc, bazin, george, or georgebazin)')
		parser.add_option('--interplc',action='store_true',default=False,
						  help='if set, interpolate the light curve before mangling')
		parser.add_option('--ab',action='store_true',default=False,
						  help='if set, ab mags for all')
		parser.add_option('--shockbreakout',action='store_true',default=False,
						  help="""if set, before the first observation the light curve is linearly interpolated
to 0 w/i half a day""")
		parser.add_option('--filterfiles',type='string',default=('',''),
						  help="""two args: comma-separated filters, and comma-separated 
files corresponding to each SN filter (In <filtpath>/<survey>.	If not provided, mangle tries <filter>.dat""",nargs=2)

		
		return(parser)

	def main(self,snfile,templatefile):
		import os

		if not os.path.exists(self.options.workdir):
			os.system('mkdir %s'%self.options.workdir)
		if '$' in self.options.filtpath:
			filtpath = os.path.expandvars(self.options.filtpath)
		if '$' in self.options.primary:
			primary = os.path.expandvars(self.options.primary)

		sn = SuperNova(os.path.expandvars(snfile))
		snid = sn.SNID
		if 'TELESCOPE' not in list(sn.__dict__.keys()):
			sn.TELESCOPE = sn.SURVEY
		# see if these already exists
		outsedfile = '%s/%s.SED'%(self.options.outdir,snid)
		outlcfile = '%s/%s.DAT'%(self.options.outdir,snid)
		if (os.path.exists(outsedfile) or os.path.exists(outlcfile)) and \
				not self.options.clobber:
			print(('Output file %s or %s exists!!  Not clobbering...'%(outsedfile,outlcfile)))
			return()

		from astropy.cosmology import Planck13 as cosmo
		sn.FLUXCAL *= 10**(0.4*(cosmo.distmod(sn.z).value-27.5))
		sn.FLUXCALERR *= 10**(0.4*(cosmo.distmod(sn.z).value-27.5))

		if 'PEAKMJD' in sn.__dict__: sn.tobs = sn.MJD - sn.PEAKMJD
		else: sn.tobs = sn.MJD - sn.SEARCH_PEAKMJD
		if self.options.smoothlc and self.options.interplc:
			fluxcalout,tobsout,fltout = np.array([]),np.array([]),np.array([])
			from scipy import interpolate
			t_interp = np.arange(min(np.unique(sn.tobs))-10,max(np.unique(sn.tobs))+10,1)
			for f in np.unique(sn.FLT):
				fnc = interpolate.InterpolatedUnivariateSpline(
					sn.tobs[sn.FLT == f],sn.FLUXCAL[sn.FLT == f],
					k=1)
				fluxtmp = fnc(t_interp)
				fluxtmp = np.interp(t_interp,sn.tobs[sn.FLT == f],
									sn.FLUXCAL[sn.FLT == f])
				fluxtmp[(t_interp < min(sn.tobs[sn.FLT == f])) | 
						(t_interp > max(sn.tobs[sn.FLT == f]))] = 0.0
				fluxtmp = savitzky_golay(fluxtmp,window_size=5)
				fluxtmp[fluxtmp < 0] = 0.0
				fluxcalout = np.append(fluxcalout,fluxtmp)
				tobsout = np.append(tobsout,t_interp)
				fltout = np.append(fltout,[f]*len(t_interp))
			sn.FLUXCAL = fluxcalout[:]
			sn.tobs = tobsout[:]
			sn.FLT = fltout[:]
			if 'PEAKMJD' in sn.__dict__: sn.MJD = sn.tobs + sn.PEAKMJD
			else: sn.MJD = sn.tobs + sn.SEARCH_PEAKMJD
			sn.FLUXCALERR = np.zeros(len(sn.FLUXCAL))
			if self.options.mkplot:
				import matplotlib.pylab as plt
				plt.clf()
				for f in sn.FILTERS:
					plt.plot(sn.tobs[sn.FLT == f],sn.FLUXCAL[sn.FLT == f],'o',label=f)
				if not os.path.exists(self.options.workdir):
					os.makedirs(self.options.workdir)
				plt.legend()
				plt.xlim([-30,100])
				plt.savefig('%s/%s.png'%(self.options.workdir,snid))
				plt.clf()
		if self.options.smoothlc:
			sn = smoothlc(sn,tobsrange=self.options.smoothtobsrange,
						  workdir=self.options.workdir,mkplot=self.options.mkplot,
						  smoothfunc=self.options.smoothfunc,shockbreakout=self.options.shockbreakout)
		if self.options.filterfiles[0]:
			self.options.filterfiles = (np.array(self.options.filterfiles[0].split(',')),
										np.array(self.options.filterfiles[1].split(',')))
			usefiltfiles = True
		else: usefiltfiles = False
		if not self.options.smoothfluxfile:
			self.options.smoothfluxfile = '%s/%s-smooth.flux'%(self.options.workdir,snid)
			fout = open(self.options.smoothfluxfile,'w')
			print('# filtfile zpt phase flux fluxerr', file=fout)
			for flt in sn.FILTERS:
				print(self.options.filterfiles[1], self.options.filterfiles[0], flt)
				if 'PEAKMJD' in sn.__dict__:
					tobs,flux,err = sn.MJD[(sn.FLT == flt[0])]-sn.PEAKMJD,\
						sn.FLUXCAL[(sn.FLT == flt[0])],sn.FLUXCALERR[(sn.FLT == flt[0])]
				else:
					tobs,flux,err = sn.MJD[(sn.FLT == flt[0])]-sn.SEARCH_PEAKMJD,\
						sn.FLUXCAL[(sn.FLT == flt[0])],sn.FLUXCALERR[(sn.FLT == flt[0])]

				for t,fl,e in zip(tobs,flux,err):
					# if e <= 0
					if t > self.options.tobsrange[0] and t < self.options.tobsrange[1]:
						if usefiltfiles:
							print('%s/%s  %.4f	%.4f  %8.5e	 %8.5e'%(
								os.path.expandvars(self.options.filtpath),
								self.options.filterfiles[1][self.options.filterfiles[0] == flt][0],
								27.5,t,fl,e), file=fout)
						else:
							print('%s/%s.dat  %.4f	%.4f  %8.5e	 %8.5e'%(
								os.path.expandvars(self.options.filtpath),flt,
								27.5,t,fl,e), file=fout)
			fout.close()

			sedfile = doMangle(snid,sn.z,niter=self.options.niter,
							   sntemp=templatefile,workdir=self.options.workdir,
							   mkplot=self.options.mkplot,smoothfluxfile=self.options.smoothfluxfile,
							   standard=self.options.primary,ndeg=len(sn.FILTERS),tol=self.options.tol,
							   verbose=self.options.verbose,scale=self.options.scale,ab=self.options.ab)

			self.snanafiles(sn,sedfile,outsedfile,outlcfile,sn.z,
							usefiltfiles=usefiltfiles,tmplfile=templatefile)

	def snanafiles(self,sn,sedfile,outsedfile,outlcfile,z,usefiltfiles=False,tmplfile=''):
		import numpy as np
		from scipy import interpolate
		import time
		from astropy.cosmology import Planck13 as cosmo

		fout = open(outsedfile,'w')
		print("""### %s.SED ###
### Mangled by D. Jones using template %s
### %i-%i-%i"""%(sn.SNID,tmplfile,time.localtime()[0],time.localtime()[1],time.localtime()[2]), file=fout)
		snt = txtobj(sedfile)


		for p in np.unique(snt.phase):
			wavearr = np.arange(2500,11000,10)
			flux = np.interp(wavearr,snt.wavelength[(snt.phase == p) & (snt.wavelength < 9000)]/(1+z),
							 snt.flux[(snt.phase == p) & (snt.wavelength < 9000)],left=0,right=0)
			for w,f in zip(wavearr,flux):
				outline = '%s %s %s'%(
					repr('%.4f'%p).rjust(14),repr('%.2f'%w).rjust(14),repr('%8.4e'%f).rjust(21))
				print(outline.replace("'",""), file=fout)
		fout.close()

		if 'PEAKMJD' in sn.__dict__:
			sn.tobs = sn.MJD-sn.PEAKMJD
		else:
			sn.tobs = sn.MJD-sn.SEARCH_PEAKMJD

		fluxcalout,tobsout,fltout = np.array([]),np.array([]),np.array([])
		for f in sn.FILTERS:
			fnc = interpolate.InterpolatedUnivariateSpline(
				sn.tobs[sn.FLT == f],sn.FLUXCAL[sn.FLT == f],
				k=1)
			fluxcalout = np.append(fluxcalout,fnc(np.unique(sn.tobs)))
			tobsout = np.append(tobsout,np.unique(sn.tobs))
			fltout = np.append(fltout,[f]*len(np.unique(sn.tobs)))

		fout = open(outlcfile,'w')
		print("NEPOCH: %i"%len(np.unique(sn.tobs)), file=fout)
		if 'SNTYPE' in sn.__dict__: sntype = sn.SNTYPE
		else: sntype = ''
		print("SNTYPE: %s"%sntype, file=fout)
		for f,a in zip(sn.FILTERS,'abcdefghijklmnopqrstuvwxyz'):
			if usefiltfiles:
				print("FILTER:	 %s	 %s/%s"%(
					a,self.options.filtpath,self.options.filterfiles[1][self.options.filterfiles[0] == f][0]), file=fout)
			else:
				print("FILTER:	 %s	 %s/%s.dat"%(
					a,self.options.filtpath,f), file=fout)

		for t in np.unique(sn.tobs):
			line = "EPOCH: %s"%repr('%.4f'%t).rjust(14)
			for f in sn.FILTERS:
				flux = fluxcalout[(fltout == f) & (tobsout == t)]
				mag = -2.5*np.log10(flux) + 27.5# - cosmo.distmod(sn.z).value
				line += "%s "%repr('%.3f'%mag).rjust(14)
			print(line[:-1].replace("'",''), file=fout)
		print('END: ', file=fout)
		fout.close()


class lightcurve_fit_mcmc:
	def __init__(self):
		self.parlist = ['A','t0','tfall','trise']

		self.npar = 4
		self.stepsize_t = 0.2
		self.stepsize_A = 0.03
		#self.stepsize_B = 0.02
		
	def adjust_model(self,X):
		
		X2 = np.zeros(self.npar)
		for i,par in zip(range(self.npar),self.parlist):
			if par == 'A': X2[i] = X[i]*10**(0.4*np.random.normal(scale=self.stepsize_A))#*stepfactor))
			elif par.startswith('t'):
				X2[i] = X[i] + np.random.normal(scale=self.stepsize_t)
			else: raise RuntimeError('parameter %s not found'%par)
		return X2

	def accept(self, last_loglike, this_loglike):
		alpha = np.exp(this_loglike - last_loglike)
		return_bool = False
		if alpha >= 1:
			return_bool = True
		else:
			if np.random.rand() < alpha:
				return_bool = True
		return return_bool

	def bazin(self,time, A, t0, tfall, trise):
		X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
		bazinfunc = A * X
		bazinfunc -= bazinfunc[time == np.min(time)][0]
		
		return bazinfunc

	
	def fit(self, time, flux, fluxerr, nsteps=10000, nburn=9000):
		scaled_time = time - time.min()
		t0 = scaled_time[flux.argmax()]
		guess = (1, t0, 40, -5)
		
		#errfunc = lambda params: 
		def errfunc(params):
			chi = -np.sum(abs(flux - self.bazin(scaled_time, *params))/fluxerr)
			#print(chi)
			return chi/2
			
		loglikes = [errfunc(guess)]
		Xlast = guess[:]
		accept = 0
		nstep = 0
		outpars = [[] for i in range(self.npar)]
		while nstep < nsteps:
			nstep += 1
			X = self.adjust_model(Xlast)
			
			# loglike
			this_loglike = errfunc(X)
			#import pdb; pdb.set_trace()
			
			# accepted?
			accept_bool = self.accept(loglikes[-1],this_loglike)
			if accept_bool:
				for j in range(self.npar):
					outpars[j] += [X[j]]
				loglikes+=[this_loglike]
				#loglike_history+=[this_loglike]
				accept += 1
				Xlast = X[:]
			else:
				for j in range(self.npar):
					outpars[j] += [Xlast[j]]
	
		print('acceptance = %.3f'%(accept/float(nstep)))
		outpars = self.getParsMCMC(np.array(outpars),nburn=nburn)
		#import pdb; pdb.set_trace()
		return outpars

	def getParsMCMC(self,x,nburn=500):

		outpars = np.array([])
		for i in range(len(x)):
			outpars = np.append(outpars,x[i][nburn:].mean())

		return outpars


class lightcurve_fit_george:
	from george.modeling import Model


	def __init__(self):
		self.syserror = 0.02
		self.gp = []

	def fit(self, time, flux, fluxerr):
		scaled_time = time - time.min()
		t0 = scaled_time[flux.argmax()]
		sys_fluxerr = self.syserror*flux
		tot_fluxerr = np.sqrt(sys_fluxerr**2. + fluxerr**2.)

		rise_time = scaled_time[:flux.argmax()+1]
		fall_time = scaled_time[flux.argmax():]

		rise_flux = flux[:flux.argmax()+1]
		fall_flux = flux[flux.argmax():]

		rise_fluxerr = tot_fluxerr[:flux.argmax()+1]
		fall_fluxerr = tot_fluxerr[flux.argmax():]


		### The coefficient of the kernel changes the fit immensly ###
		rise_kernel = np.var(rise_flux) * kernels.ExpKernel(5.0)
		fall_kernel = np.var(fall_flux) * kernels.Matern32Kernel(5.0)
		kernel = rise_kernel + fall_kernel


		
		self.gp = george.GP(kernel, mean=self.GeorgeModel(A=1, beta=0, c=0, tmax=t0, tfall=40, trise=-5))
		self.gp.compute(scaled_time, fluxerr)

		def lnprob(p):
			self.gp.set_parameter_vector(p)
			return self.gp.log_likelihood(flux, quiet=True) + self.gp.log_prior()

		'''def neg_ln_like(p):
			self.gp.set_parameter_vector(p)
			return -self.gp.log_likelihood(flux)

		def grad_neg_ln_like(p):
			self.gp.set_parameter_vector(p)
			return -self.gp.grad_log_likelihood(flux)

		result = minimize(neg_ln_like, self.gp.get_parameter_vector(), jac=grad_neg_ln_like)'''

		init = self.gp.get_parameter_vector()
		ndim = len(init)
		nwalkers = 2*ndim
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

		# 500 iterations takes about 2 mins on ziggy
		print("Running first burn-in...")
		p0 = init + 1e-8 * np.random.randn(nwalkers, ndim)
		p0,lp,_ = sampler.run_mcmc(p0, 200)
		
		print("Running second burn-in...")
		p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
		sampler.reset()
		p0,_,_ = sampler.run_mcmc(p0, 200)
		sampler.reset()

		# 1000 iterations takes about 3 mins
		print("Running production...")
		sampler.run_mcmc(p0, 500)

		samples = sampler.flatchain
		for s in samples[np.random.randint(len(samples), size=24)]:
			self.gp.set_parameter_vector(s)


		#self.gp.set_parameter_vector(result.x)
		

		return self.gp
		

	def george(self, time, flux):
		return self.gp.predict(flux,time,return_cov=False)


	class GeorgeModel(Model):
		parameter_names = ("A", "beta", "c", "tmax", "tfall", "trise")

		def get_value(self, time):

			model = np.zeros(len(time))

			self.trise, self.tfall = np.abs(self.trise), np.abs(self.tfall)
			model[time < self.tmax] = 1
			model[time >= self.tmax] = np.exp(-(time[time >= self.tmax]-self.tmax)/self.tfall)
			model *= (self.A + self.beta*(time-self.tmax))
			model *= 1/(1 + np.exp(-(time-self.tmax)/self.trise))
			model += self.c

			return model


'''
	def lnlike(p, t, y, yerr):
		#return -0.5 * np.sum(((y-model(p,t))/yerr) **2)
		# Change np.var(y) and 5.0 to something better
		a, tau = np.exp(p[:2])
		self.gp = george.GP(a * kernels.Matern32Kernel(tau))
		self.gp.compute(t, yerr)
		return self.gp.lnlikelihood(y - model(p, t))


	def lnprior(p):
		lna, lntau, A, beta, c, tmax, tfall, trise = p

		if (-1 < A < 1):
			return 0.0
		return -np.inf
		'''



		
def lightcurve_fit(time, flux, fluxerr):
	scaled_time = time - time.min()
	t0 = scaled_time[flux.argmax()]
	guess = (0, 0, t0, 40, -5)

	errfunc = lambda params: abs(flux - bazin(scaled_time, *params))/fluxerr

	result = least_squares(errfunc, guess, method='lm')

	return result.x

'''def lightcurve_fit_george(time, flux, fluxerr):
	import george
	from george import kernels
	from scipy.optimize import minimize
	scaled_time = time - time.min()
	t0 = scaled_time[flux.argmax()]
	guess = (0, 0, 0, t0, 40, -5)
	yerr = flux*0.02
		
	kernel = np.var(flux) * kernels.Matern32Kernel(5.0) # What kernel to use?
	gp = george.GP(kernel, white_noise=np.log(0.005*np.max(flux)))
	gp.compute(scaled_time, yerr)
	
	trange = np.arange(-40, 100, 1)
	pred, pred_var = gp.predict(flux, trange, return_var=True)
	
	plt.errorbar(scaled_time, flux, yerr=yerr, fmt='k.')
	plt.plot(trange, pred, 'k')
	plt.fill_between(trange, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var), color='k', alpha = 0.3)
	
	plt.xlabel('Phase')
	plt.ylabel('Flux')
	plt.show()
	
	
	def neg_ln_like(p):
		gp.set_parameter_vector(p)
		return -gp.log_likelihood(flux)
	
	def grad_neg_ln_like(p):
		gp.set_parameter_vector(p)
		return -gp.grad_log_likelihood(flux)
	
	
	# errfunc = lambda params: abs(flux - george(scaled_time, *params))/fluxerr

	result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
	gp.set_parameter_vector(result.x)
	
	pred, pred_var = gp.predict(flux, trange, return_var=True)
	
	plt.errorbar(scaled_time, flux, yerr=yerr, fmt='k.')
	plt.plot(trange, pred, 'k')
	plt.fill_between(trange, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var), color='k', alpha = 0.3)
	
	plt.xlabel('Phase')
	plt.ylabel('Flux')
	plt.show()
	kernel.get_parameter_vector()

	return result.x'''

def lightcurve_fit_georgepbazin(time, flux, fluxerr):
	scaled_time = time - time.min()
	t0 = scaled_time[flux.argmax()]
	guess = (0, 0, 0, t0, 40, -5, 0, 0, t0-10, 40, -5)

	errfunc = lambda params: abs(flux - georgebazin(scaled_time, *params))/fluxerr

	result = least_squares(errfunc, guess, method='lm')

	return result.x



def bazin(time, A, B, t0, tfall, trise):
	X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
	return A * X + B

def bazin_noB(time, A, t0, tfall, trise):
	X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
	bazinfunc = A * X
	bazinfunc -= bazinfunc[time == np.min(time)][0]

	return bazinfunc


'''def george(time, A, beta, c, tmax, tfall, trise):

	model = np.zeros(len(time))

	trise,tfall = np.abs(trise),np.abs(tfall)
	model[time < tmax] = 1
	model[time >= tmax] = np.exp(-(time[time >= tmax]-tmax)/tfall)
	model *= (A + beta*(time-tmax))
	model *= 1/(1 + np.exp(-(time-tmax)/trise))
	model += c

	return model'''

def georgebazin(time, A, beta, c, tmax, tfall, trise,
		   bazinA, bazinB, bazint0, bazintfall, bazintrise):

	model = np.zeros(len(time))

	trise,tfall = np.abs(trise),np.abs(tfall)
	model[time < tmax] = 1
	model[time >= tmax] = np.exp(-(time[time >= tmax]-tmax)/tfall)
	model *= (A + beta*(time-tmax))
	model *= 1/(1 + np.exp(-(time-tmax)/trise))
	model += c
	
	bazinX = np.exp(-(time - bazint0) / bazintfall) / (1 + np.exp((time - bazint0) / bazintrise))
	bazinmodel = bazinA * bazinX + bazinB

	
	return model + bazinmodel


def smoothlc(sn,tobsrange=[-30,90],mkplot=False,workdir='workdir',addpts=True,
			 smoothfunc=None,shockbreakout=False,debug=False):
	from scipy.optimize import minimize
	import numpy as np

	if 'FAKEFLAG' not in list(sn.__dict__.keys()):
		sn.FAKEFLAG = np.zeros(len(sn.tobs))
	sn.FLUXCALERR[:] = 0.1
	for f in sn.FILTERS:
		sn.tobs = np.append(sn.tobs,tobsrange[0])
		sn.FLUXCAL = np.append(sn.FLUXCAL,0)
		sn.FLUXCALERR = np.append(sn.FLUXCALERR,0.01)#np.median(sn.FLUXCALERR))
		sn.FLT = np.append(sn.FLT,f)
		sn.FAKEFLAG = np.append(sn.FAKEFLAG,1)
		
		sn.tobs = np.append(sn.tobs,tobsrange[1])
		sn.FLUXCAL = np.append(sn.FLUXCAL,0)
		sn.FLUXCALERR = np.append(sn.FLUXCALERR,0.01)#np.median(sn.FLUXCALERR))
		sn.FLT = np.append(sn.FLT,f)
		sn.FAKEFLAG = np.append(sn.FAKEFLAG,1)
		
		#if f == 'V':
		#	sn.tobs = np.append(sn.tobs,-23.5)
		#	sn.FLUXCAL = np.append(sn.FLUXCAL,5000)
		#	sn.FLUXCALERR = np.append(sn.FLUXCALERR,0.01)#np.median(sn.FLUXCALERR))
		#	sn.FLT = np.append(sn.FLT,f)
		#if f == 'B':
		#	sn.tobs = np.append(sn.tobs,-23.5)
		#	sn.FLUXCAL = np.append(sn.FLUXCAL,5000)
		#	sn.FLUXCALERR = np.append(sn.FLUXCALERR,0.01)#np.median(sn.FLUXCALERR))
		#	sn.FLT = np.append(sn.FLT,f)
		
#	 sn.FLUXCALERR[:] = 1e-2
	if mkplot:
		import matplotlib.pylab as plt
		plt.clf()

	icol = np.where((sn.tobs/(1+sn.z) >= tobsrange[0]-1) &
					(sn.tobs/(1+sn.z) <= tobsrange[1]+1))
	#guess = []
	#for i in range(len(sn.FILTERS)):
	#	guess += [max(sn.FLUXCAL[sn.FLT == sn.FILTERS[i]])]
	#guess += [0.]*len(sn.FILTERS)*2 + [20.]*len(sn.FILTERS) + [50.]*len(sn.FILTERS) + [0.]
	#for f in sn.FILTERS:
	#md = minimize(smoothfunc,guess,args=(sn.FLUXCAL[icol],sn.FLUXCALERR[icol],sn.tobs[icol],sn.FLT[icol]))
	#if md.message != 'Optimization terminated successfully.':
	#	print('Warning!! : %s'%md.message)

	if smoothfunc == 'mcmc':
		fitfunc = lightcurve_fit_mcmc
		lcfunc = bazin_noB
	if smoothfunc == 'bazin':
		fitfunc = lightcurve_fit
		lcfunc = bazin
	elif smoothfunc == 'george':
		fitfunc = lightcurve_fit_george
	elif smoothfunc == 'georgebazin':
		fitfunc = lightcurve_fit_georgepbazin
		lcfunc = georgebazin
		
		
	tflt = np.array([])
	tsmoothout = np.array([])
	tsmooth = np.arange(tobsrange[0],tobsrange[1],0.25)
	model = np.array([])
	
	# For sigma clipping #
	expectedvalues = []
	FLUXCALERRNEW = np.sqrt(sn.FLUXCALERR**2. + (0.02 * sn.FLUXCAL)**2.)


	for flt in sn.FILTERS:
		print(flt)
		tflt = np.append(tflt,[flt]*len(tsmooth))
		tsmoothout = np.append(tsmoothout,tsmooth)
		icol = np.where((sn.tobs/(1+sn.z) >= tobsrange[0]-1) &
						(sn.tobs/(1+sn.z) <= tobsrange[1]+1) &
						(sn.FLT == flt))

		# Also does sigma clipping #
		if smoothfunc == 'mcmc':
			i_outlier = np.array([])
			fltexpectedvalues = np.array([])
			maxiter = 5
			it = 0

			while it < maxiter:
				it += 1
				
				
				# Do fit with data points within 3*sigma #
				iclipped  = np.setdiff1d(icol, i_outlier)
				
				lcfit = lightcurve_fit_mcmc()
				fit_params = lcfit.fit(sn.tobs[iclipped],sn.FLUXCAL[iclipped]/np.max(sn.FLUXCAL[iclipped]),sn.FLUXCALERR[iclipped])
					
				# Determine data points outside of 3*sigma #
				fltexpectedvalues = lcfunc(sn.tobs[icol]-np.min(sn.tobs[iclipped]),*fit_params)*np.max(sn.FLUXCAL[iclipped])
				
				sigma = np.sqrt(np.sum((sn.FLUXCAL[icol] - fltexpectedvalues)**2.)/len(sn.FLUXCAL[icol]))
				
				i_outlier = np.fromiter((icol[0][i] for i in range(len(icol[0])) if sn.FLUXCAL[icol[0][i]] - FLUXCALERRNEW[icol[0][i]] > fltexpectedvalues[i] + 3*sigma or sn.FLUXCAL[icol[0][i]] + FLUXCALERRNEW[icol[0][i]] < fltexpectedvalues[i] - 3*sigma), int)
				
			expectedvalues.append(fltexpectedvalues)
			
			 
		elif smoothfunc == 'george':
			lcfit = lightcurve_fit_george()
			lcfunc = lcfit.george

			lcfit.fit(sn.tobs[icol], sn.FLUXCAL[icol]/np.max(sn.FLUXCAL[icol]),sn.FLUXCALERR[icol])
			fit_params = [sn.FLUXCAL[icol]/np.max(sn.FLUXCAL[icol])]
		else:
			fit_params = fitfunc(sn.tobs[icol], sn.FLUXCAL[icol]/np.max(sn.FLUXCAL[icol]),sn.FLUXCALERR[icol])
		filtmodel = lcfunc(tsmooth-sn.tobs[icol].min(),*fit_params)*np.max(sn.FLUXCAL[icol])
		#filtmodel -= np.min(filtmodel)
		#import pdb; pdb.set_trace()
		if shockbreakout:
			iMinT = np.min(sn.tobs[sn.FAKEFLAG == 0])+1# - 0.5
			if len(filtmodel[tsmooth < iMinT]):
				filtmodel[tsmooth < iMinT] = 0
				iInterp = (tsmooth < iMinT) & (tsmooth > iMinT + 1)
				iMin = [(tsmooth - iMinT)**2 == np.min((tsmooth-iMinT)**2)]
				from numpy import ones,vstack
				from numpy.linalg import lstsq
				points = [(iMinT-1,0),(tsmooth[iMin],filtmodel[iMin])]
				x_coords, y_coords = list(zip(*points))
				A = vstack([x_coords,ones(len(x_coords))]).T
				m, b = lstsq(A, y_coords)[0]

				filtmodel[iInterp] = m*tsmooth[iInterp] + b
			#model[(sn.tobs[icol] > iMinT) & (sn.tobs[icol] < iMinT+0.5)] = np.interp
		model = np.append(model,filtmodel)
		model[model < 0] = 0
			
		icol2 = np.where((sn.tobs/(1+sn.z) >= tobsrange[0]-1) &
						 (sn.tobs/(1+sn.z) <= tobsrange[1]+1) &
						 (sn.FLT == flt))
		iMinData = np.argsort(sn.tobs[icol2])[1]

		'''print(('%s data at %i days: %.2f model: %.2f'%(
			flt,sn.tobs[icol2][iMinData],
			-2.5*np.log10(sn.FLUXCAL[icol2][iMinData])+27.5,
			-2.5*np.log10(lcfunc(np.array([sn.tobs[icol2][iMinData]-sn.tobs[icol2].min()]),*fit_params)*np.max(sn.FLUXCAL[icol]))+27.5)))'''
		#import pdb; pdb.set_trace()		
	tsmooth = tsmoothout[:]

	#model = smoothfunc_modelout(md.x,t=tsmooth,flt=tflt)
	if mkplot:
		cwheel = {'u':'m','b':'b','v':'c', 'g':'g','r':'r','i':'k'}
		plt.gca().set_prop_cycle(None)
	
	for flt,i in zip(sn.FILTERS,list(range(len(sn.FILTERS)))):
		icol = np.where((sn.tobs/(1+sn.z) >= tobsrange[0]-1) &
						(sn.tobs/(1+sn.z) <= tobsrange[1]+1) &
						(sn.FLT == flt))
		tcol = np.where(tflt == flt)
		if mkplot:
			#sigma = np.sqrt(np.sum((sn.FLUXCAL[icol] - expectedvalues[i])**2.)/len(sn.FLUXCAL[icol]))
			
			# Plot fitted lightcurve #
			
			
			# Add shaded area within 1*sigma of model #
			# plt.fill_between(tsmooth[tcol], model[tcol] - sigma, model[tcol] + sigma,facecolor=cwheel[i],alpha=0.2)
								
			# Plot photometry #
			#i_outlier = np.fromiter((icol[0][j] for j in range(len(icol[0])) if sn.FLUXCAL[icol[0][j]] - FLUXCALERRNEW[icol[0][j]] > expectedvalues[i][j] + 3*sigma or sn.FLUXCAL[icol[0][j]] + FLUXCALERRNEW[icol[0][j]] < expectedvalues[i][j] - 3*sigma), int)
			i_outlier = []
			iclipped = np.setdiff1d(icol, i_outlier)
			if flt.lower() in cwheel:	
				plt.plot(tsmooth[tcol],model[tcol],color=cwheel[flt.lower()])

				plt.errorbar(sn.tobs[iclipped],sn.FLUXCAL[iclipped],
						 yerr=sn.FLUXCALERR[iclipped],fmt='o',color=cwheel[flt.lower()],label=flt)
			
			# Data points removed from fitting via 3*sigma clipping #
				plt.errorbar(sn.tobs[i_outlier],sn.FLUXCAL[i_outlier],
						 yerr=sn.FLUXCALERR[i_outlier],fmt='x',color=cwheel[flt.lower()])
			else:
				plt.plot(tsmooth[tcol],model[tcol])
				plt.errorbar(sn.tobs[iclipped],sn.FLUXCAL[iclipped],
						 yerr=sn.FLUXCALERR[iclipped],fmt='o',label=flt)
			
			# Data points removed from fitting via 3*sigma clipping #
				plt.errorbar(sn.tobs[i_outlier],sn.FLUXCAL[i_outlier],
						 yerr=sn.FLUXCALERR[i_outlier],fmt='x')

			
			# To look at plots of individual filters #
			#icol = np.where((sn.tobs/(1+sn.z) >= tobsrange[0]-1) & (sn.tobs/(1+sn.z) <= tobsrange[1]+1))
			plt.xlim([min(sn.tobs[icol])-5,max(sn.tobs[icol])+5])
			plt.ylim([0,max(sn.FLUXCAL[icol])*4/3.])
			plt.legend(numpoints=1)
			plt.xlabel('Phase')
			plt.ylabel('Flux')
			#plt.show()
			plt.savefig('%s/%s.png'%(workdir, sn.SNID + flt))
			plt.clf()
			
	# youngsn xlim is off
	if mkplot:
		icol = np.where((sn.tobs/(1+sn.z) >= tobsrange[0]-1) &
						(sn.tobs/(1+sn.z) <= tobsrange[1]+1))
		plt.xlim([min(sn.tobs[icol])-5,max(sn.tobs[icol])+5])
		plt.ylim([max(sn.FLUXCAL[icol])*(-1/10.),max(sn.FLUXCAL[icol])*4/3.])
		plt.legend(numpoints=1)
		plt.xlabel('Phase')
		plt.ylabel('Flux')
		plt.title("%s" % sn.SNID,fontsize=15)
		#plt.xlim([-3,3])
		plt.savefig('%s/%s.lcsmooth.png'%(workdir,sn.SNID))

		ycol = np.where((sn.tobs/(1+sn.z) > tobsrange[0]) &
						(sn.tobs/(1+sn.z) < tobsrange[0]+30))
		plt.xlim([min(sn.tobs[ycol])-5, max(sn.tobs[ycol]) + 5])
		plt.ylim([max(sn.FLUXCAL[ycol])*(-1/10.),max(sn.FLUXCAL[ycol])*4./3.])
		#plt.show()
		plt.savefig('%s/%s.lcsmooth.youngsn.png'%(workdir,sn.SNID))
		
	if debug: import pdb; pdb.set_trace()
		
	if 'PEAKMJD' in sn.__dict__: sn.MJD = tsmooth + sn.PEAKMJD
	else: sn.MJD = tsmooth + sn.SEARCH_PEAKMJD
	sn.FLUXCAL = model[:]
	sn.FLUXCALERR = np.zeros(len(model))
	sn.FIELD = np.array('NULL'*len(model))
	sn.FLT = tflt[:]
	sn.MAG = np.array([-99]*len(tflt))
	sn.MAGERR = np.array([-99]*len(tflt))

	return(sn)

def smoothfunc(x,flux=None,fluxerr=None,t=None,flt=None):
	import numpy as np

	filtlen = len(np.unique(flt))
	#Alist = x[0:filtlen]
	#a1,a2,trise,tfall,tmax = x[filtlen:]

	Alist = x[0:filtlen]
	a1 = x[filtlen:filtlen*2]
	a2 = x[filtlen*2:filtlen*3]
	trise = x[filtlen*3:filtlen*4]
	tfall = x[filtlen*4:filtlen*5]
	tmax = x[-1]

	expterm = np.zeros(len(flt))
	polyterm = np.zeros(len(flt))

#	 expterm = np.exp(-(t-tmax)/np.abs(tfall[0]))/(1+np.exp(-(t-tmax)/np.abs(trise[0])))
	for f,i in zip(np.unique(flt),list(range(filtlen))):
		expterm[(flt == f)] = np.exp(-(t[(flt == f)]-tmax)/np.abs(tfall[i]))/(1+np.exp(-(t[(flt == f)]-tmax)/np.abs(trise[i])))
		polyterm[(flt == f)] = Alist[i]*(1 + a1[i]*(t[(flt == f)]-tmax) + a2[i]*(t[(flt == f)]-tmax)**2.)

	model = expterm*polyterm
	model[(model > 1e100)] = 0
	model[(model != model)] = 0
	model[(model < 0)] = 0
	return(np.sum((model-flux)**2./fluxerr**2.))

def smoothfunc_george(x,flux=None,fluxerr=None,t=None,flt=None):
	import numpy as np

	filtlen = len(np.unique(flt))
	#Alist = x[0:filtlen]
	#a1,a2,trise,tfall,tmax = x[filtlen:]

	A = x[0:filtlen]
	beta = x[filtlen:filtlen*2]
	c = x[filtlen*2:filtlen*3]
	trise = x[filtlen*3:filtlen*4]
	tfall = x[filtlen*4:filtlen*5]
	tmax = x[-1]

	model = np.zeros(len(flt))
	for f,i in zip(np.unique(flt),list(range(filtlen))):
		trise[i],tfall[i] = np.abs(trise[i]),np.abs(tfall[i])
		model[((flt == f) & (t < tmax))] = 1
		model[((flt == f) & (t >= tmax))] = \
			np.exp(-(t[(flt == f) & (t >= tmax)]-tmax)/tfall[i])
		model[(flt == f)] *= (A[i] + beta[i]*(t[flt == f]-tmax))
		model[flt == f] *= 1/(1 + np.exp(-(t[flt == f]-tmax)/trise[i]))
		model[(flt == f)] += c[i]

	return(np.sum((model-flux)**2./fluxerr**2.))

def smoothfunc_george_modelout(x,flux=None,fluxerr=None,t=None,flt=None):
	import numpy as np

	filtlen = len(np.unique(flt))
	#Alist = x[0:filtlen]
	#a1,a2,trise,tfall,tmax = x[filtlen:]

	A = x[0:filtlen]
	beta = x[filtlen:filtlen*2]
	c = x[filtlen*2:filtlen*3]
	trise = x[filtlen*3:filtlen*4]
	tfall = x[filtlen*4:filtlen*5]
	tmax = x[-1]

	model = np.zeros(len(flt))
	for f,i in zip(np.unique(flt),list(range(filtlen))):
		trise[i],tfall[i] = np.abs(trise[i]),np.abs(tfall[i])
		model[((flt == f) & (t < tmax))] = 1
		model[((flt == f) & (t >= tmax))] = \
			np.exp(-(t[(flt == f) & (t >= tmax)]-tmax)/tfall[i])
		model[(flt == f)] *= (A[i] + beta[i]*(t[flt == f]-tmax))
		model[flt == f] *= 1/(1 + np.exp(-(t[flt == f]-tmax)/trise[i]))
		model[(flt == f)] += c[i]

	return(model)


def smoothfunc_modelout(x,flux=None,fluxerr=None,t=None,flt=None):
	import numpy as np

	filtlen = len(np.unique(flt))
	#Alist = x[0:filtlen]
	#a1,a2,trise,tfall,tmax = x[filtlen:]

	Alist = x[0:filtlen]
	a1 = x[filtlen:filtlen*2]
	a2 = x[filtlen*2:filtlen*3]
	trise = x[filtlen*3:filtlen*4]
	tfall = x[filtlen*4:filtlen*5]
	tmax = x[-1]

	expterm = np.zeros(len(flt))
	polyterm = np.zeros(len(flt))

#	 expterm = np.exp(-(t-tmax)/np.abs(tfall[0]))/(1+np.exp(-(t-tmax)/np.abs(trise[0])))
	for f,i in zip(np.unique(flt),list(range(filtlen))):
		expterm[(flt == f)] = np.exp(-(t[(flt == f)]-tmax)/np.abs(tfall[i]))/(1+np.exp(-(t[(flt == f)]-tmax)/np.abs(trise[i])))
		polyterm[(flt == f)] = Alist[i]*(1 + a1[i]*(t[(flt == f)]-tmax) + a2[i]*(t[(flt == f)]-tmax)**2.)

	model = expterm*polyterm
	model[(model > 1e100)] = 0
	model[(model != model)] = 0
	return(model)


def doMangle(snid,z,niter=9,sntemp=None,
			 workdir='workdir',mkplot=False,
			 maxphase=0,smoothfluxfile='',
			 standard=None,ndeg=4,tol=10.,
			 verbose=False,scale=1,ab=False):
	from scipy.optimize import minimize

	if not os.path.exists(workdir): os.system('mkdir %s'%workdir)

	if mkplot: from matplotlib.backends.backend_pdf import PdfPages
	for i in range(niter):
		if verbose:
			print(('Starting iteration %i of %i'%(i+1,niter)))
		if i == 0:
			if not sntemp: 
				print('Error : no SN template file given'); return
			snt = txtobj(sntemp)
			minphase = min(snt.phase)
			wavelength = snt.wavelength[np.where(snt.phase == minphase)]
			phase,wavelength_full,flux = np.array([]),np.array([]),np.array([])
			for p in np.unique(snt.phase):
				cols = np.where(snt.phase == p)[0]
				wavelength_full = np.append(wavelength_full,wavelength)
				flux = np.append(flux,np.interp(wavelength,snt.wavelength[cols],snt.flux[cols]))
				phase = np.append(phase,np.array([p]*len(wavelength)))
			snt.phase,snt.wavelength,snt.flux = phase[:],wavelength_full[:],flux[:]
			
			snt.phase -= maxphase
			snt.phase *= (1+z)
			snt.wavelength *= (1+z)
			sedfn = snt.writeto('%s/%s.sed.iter00.txt'%(workdir,snid),verbose=True)

		if mkplot:
			pdf_pages = PdfPages('%s/%s.iter%02i.pdf'%(workdir,snid,i))
			fig = plt.figure()

		snt = txtobj(sedfn)
		nphase = len(np.unique(snt.phase))
		pp = np.concatenate(([np.min(snt.phase)-40],snt.phase,[np.max(snt.phase)+100]))
		snt.phase = pp[:]
		nphase += 2
		nflux = np.zeros([int(len(snt.wavelength)/(nphase-2)),nphase])
		phaseunq = np.unique(snt.phase)

		for j in range(1,nphase-1): 
			nflux[:,j] = \
				snt.flux[np.where(snt.phase[1:-1] == phaseunq[j])]
		snt.flux = nflux[:]

		snt.wavelength = snt.wavelength[0:int(len(snt.wavelength)/(nphase-2))]
		if mkplot and i == 0:
			cols = np.where(np.abs(phaseunq) == min(np.abs(phaseunq)))[0]
			xr=snt.wavelength/(1+z); yr = snt.flux[:,cols]/np.max(snt.flux[:,cols])

		# read in smooth flux file
		sff = txtobj(smoothfluxfile)
		efflam = np.array([])
		for f in sff.filtfile: efflam = np.append(efflam,filt2flam(f))

		# ignore the areas not covered by light curve observations
		filters = np.unique(sff.filtfile)
		qq = np.where((snt.wavelength >= 0.7*np.min(efflam)) * (snt.wavelength <= 1.3*np.max(efflam)))[0]
		snt.wavelength = snt.wavelength[qq]; snt.flux = snt.flux[qq,:]

		# sort the dates, get rid of multiples
		# interpolate
		newflux = np.zeros([np.shape(snt.flux)[0],len(np.unique(sff.phase))])
		for j in range(len(snt.wavelength)):
			newflux[j,:] = np.interp(np.unique(sff.phase),np.unique(snt.phase),snt.flux[j,:])
		snt.flux = newflux[:]
		snt.phase = np.unique(sff.phase)

		x0 = np.min(snt.wavelength); dx = np.max(snt.wavelength) - x0
		lxx = (snt.wavelength - x0)/dx * 2. - 1.
		lx = (efflam - x0)/dx * 2. - 1.

		p0 = min(np.unique(sff.phase)); dp = max(np.unique(sff.phase)) - p0
		pxx = (np.unique(sff.phase) - p0)/dp * 2. - 1.
		px = (sff.phase - p0)/dp * 2. - 1.
		nowp=sff.phase[:] ; nowfilt=sff.filtfile[:] ; nowlam=snt.wavelength[:]
		nowflux=snt.flux[:] ; nowphase=np.unique(snt.phase[:])

		infmod = getfmod(nowp, nowfilt, nowlam, nowflux, 
						 nowphase, standard, sff.zpt, ab=ab)

		if verbose:
			print(('max of flux = %s'%np.max(nowflux[:,10])))

		if i == 0:
			jj = np.where((sff.phase > -5.) & (sff.phase < 5))[0]
			if len(jj) < 5: print('very few points near max?')

			infmod2 = infmod[:]
			infmod2[(infmod <= 0)] = 1e-99
			print('using mean for ratio, not median')
			ratio = np.mean(sff.flux[jj]/infmod2[jj])

			infmod *= ratio
			nowflux *= ratio
			nowflux *= scale

		fmod = getfmod(sff.phase, nowfilt, nowlam, 
					   nowflux, nowphase, standard, sff.zpt,ab=ab)

		if mkplot:
			aplot(sff.phase,sff.flux,fmod,nowfilt,np.unique(nowfilt),title='Starting Point for iter %i'%i)
			pdf_pages.savefig(fig)
			plt.close(fig)
			fig = plt.figure()

		norm = 1
		normscl = 0.3

		pwarp = [0., 1., 0.]		 # quadratic time warp in pxx
		pwscl = [0.2, 0.2, 0.2]	  # scale in pxx = [-1,1]

		coeffs = np.zeros(ndeg*ndeg) + 1
		cscl = coeffs + 0.4

		inp = np.concatenate(([norm],pwarp,coeffs))
		scale = [normscl,pwscl,cscl]

		random = np.random.standard_normal((ndeg,ndeg))*1e-15
		out = minimize(meritfunc,inp,
					   args=(nowp,sff.flux,sff.fluxerr,fmod,px,lx,nowfilt,
							 np.unique(nowfilt),len(np.unique(nowfilt)),
							 ndeg,random),tol=tol,method='Nelder-Mead')

		if out.message != 'Optimization terminated successfully.':
			print(('Warning!! : %s'%out.message))
			
		if verbose:
			print(('nfev = %i merit = %.7f'%(
					out.nfev,meritfunc(
						out.x,nowp,sff.flux,sff.fluxerr,fmod,px,lx,
						nowfilt,np.unique(nowfilt),
						len(np.unique(nowfilt)),ndeg,random))))

		nmod = modelfunc(out.x,fmod,px,lx,nowfilt,
						 np.unique(nowfilt),len(np.unique(nowfilt)),
						 ndeg,random)
		if mkplot:
			aplot(sff.phase,sff.flux,nmod,nowfilt,np.unique(nowfilt),title='mangled spectrum iter %i'%i)
			pdf_pages.savefig(fig)
			plt.close(fig)
			fig = plt.figure()

		nowflux = modelspec(out.x,nowflux,pxx,lxx,pwarp,ndeg,random)
		fadmodel = getfmod(nowp, nowfilt, nowlam, nowflux, nowphase, standard, sff.zpt,ab=ab)

		fmod = fadmodel[:]
		flux = nowflux[:]

		ufilt = np.unique(nowfilt)
		fadjm = meritfunc(inp,nowp,sff.flux,sff.fluxerr,fmod,px,lx,nowfilt,ufilt,len(ufilt),ndeg,random)
		if verbose:
			print(('flux adjusted: %s'%fadjm))

		if mkplot:
			aplot(sff.phase,sff.flux,fmod,nowfilt,np.unique(nowfilt),title='mangled spectrum with synthetic photometry, iter %i'%i)
			pdf_pages.savefig(fig)
			plt.close(fig)
			fig = plt.figure()		  

		col = np.where(fmod <= 0)[0]
		if len(col): fmod[col] = 1e-99
		rat = sff.flux / fmod
		ratmincol = np.where(rat < 0.3)[0]
		ratmaxcol = np.where(rat > 3.)[0]
		if len(ratmincol): rat[ratmincol] = 0.3
		if len(ratmaxcol): rat[ratmaxcol] = 3.0
		
		rat2 = np.append(rat,[1.,1.,1.,1.])
		px2 = np.concatenate((px,[-1.2,-1.2,1.2,1.2]))
		lx2 = np.concatenate((lx,[-1.2,1.2,-1.2,1.2]))

		sres2 = min_curve_surf(rat2,px2,lx2,xout=pxx,yout=lxx,tps=True)
		sresmincol = np.where(sres2 < 0.3)
		sresmaxcol = np.where(sres2 > 3.)
		if len(sresmincol): sres2[sresmincol] = 0.3
		if len(sresmaxcol): sres2[sresmaxcol] = 3.0

		nowflux = flux*np.array(sres2)
		nowflux[(nowflux < 0)] = 0

		csmmodel = getfmod(nowp, nowfilt, nowlam, nowflux, nowphase, standard, sff.zpt,ab=ab)
		fmod = csmmodel[:]
		csm = meritfunc(inp,nowp,sff.flux,sff.fluxerr,fmod,px,lx,nowfilt,np.unique(nowfilt),
						len(np.unique(nowfilt)),ndeg,random)
		if verbose:
			print(('mincurvesurf: %s'%csm))
		if mkplot:
			aplot(sff.phase,sff.flux,fmod,nowfilt,np.unique(nowfilt),title='final clipped model')
			pdf_pages.savefig(fig)
			plt.close(fig)
			fig = plt.figure()


		if csm < fadjm*1.1: snt.flux = nowflux[:]
		sedfn = '%s/%s.sed.iter%02i.txt'%(workdir,snid,i+1)

		fout = open(sedfn,'w')
		print('# phase wavelength flux', file=fout)
		for j in range(len(nowphase)):
			for h in range(len(snt.wavelength)):
				print('%s %s %s'%(nowphase[j],snt.wavelength[h],snt.flux[h,j]), file=fout)
		fout.close()
		if mkplot:
			plt.plot(xr,yr,color='k',label='template at peak')
			cols = np.where(np.abs(nowphase) == min(np.abs(nowphase)))[0]
			plt.plot(snt.wavelength/(1+z),snt.flux[:,cols]/np.max(snt.flux[:,cols]),
					 ls='--',label='Warped template at peak, iter %i'%i)
			
			for f in filters:
				w,tp = np.loadtxt(f,unpack=True)
				plt.plot(w,tp*0.3)#,label='filter %s'%f)
			plt.xlabel('wavelength')
			plt.ylabel('flux')
			plt.xlim([2000,9000])
			plt.legend()
			pdf_pages.savefig(fig)
			plt.close(fig)
			pdf_pages.close()
	return(sedfn)

def modelspec(x,flux,pxx,lxx,pwarp,ndeg,random):
	import numpy as np
	from scipy import interpolate
	# x[0] :  flux normalization
	# x[1:3] : polynomial time warp
	# x[4:ndegr*ndegr+3] : spline table coefficients

	norm = x[0]
	pwarp = x[1:4]

	outflux = flux-flux
	for i in range(np.shape(flux)[0]): 
		f = interpolate.InterpolatedUnivariateSpline(
			poly(pxx,pwarp),flux[i,:],
			k=1)
		outflux[i,:] = f(pxx)
		outflux[i,:][(outflux[i,:] != outflux[i,:])] = 0.0

	xx = np.arange(-1.2,1.21,2.4/(ndeg-1))
	yy = xx[:]
	mm = np.zeros([ndeg,ndeg]) + random 
	#np.random.standard_normal((ndeg,ndeg))*1e-10

	ind = 4
	for i in range(ndeg):
		for j in range(ndeg):
			if x[ind] < 0.2: mm[j,i] += 0.2
			elif x[ind] > 5.: mm[j,i] += 5.
			else:  
				mm[j,i] += x[ind]
			ind += 1
	
	mult = min_curve_surf(mm,xvalues=xx,yvalues=yy,xout=pxx,yout=lxx)

	if len(np.where(mult < 0.2)[0]): mult[np.where(mult < 0.2)] = 0.2
	if len(np.where(mult > 5.)[0]): mult[np.where(mult > 5.)] = 5.

	outvar = (outflux * np.array(mult) / norm)
	if len(np.where(outvar < 0)[0]): outvar[np.where(outvar < 0)] = 0
	return(outvar)

def min_curve_surf(z, x=[], y=[], regular = None, xgrid = [],
				   xvalues = [], ygrid = [], yvalues = [],
				   gs = [], bounds = [], nx0 = [], ny0 = [], xout = None,
				   yout = None, xpout = [], ypout = [], sphere = False,
				   tol = 1.0e-20, tps = False):
	"""
	Copyright (c) 1993, Research Systems, Inc.	All rights reserved.
	Unauthorized reproduction prohibited.
	+
	NAME:
	   MIN_CURVE_SURF

	PURPOSE:
	   Interpolate a regular or irregularly gridded set of points
	   with a minimum curvature spline surface.
	CATEGORY:
	   Interpolation, Surface Fitting
	CALLING SEQUENCE:
	   Result = min_curve_surf(z [, x, y])
	INPUTS:
	   X, Y, Z = arrays containing the x, y, and z locations of the
			   data points on the surface.	Need not necessarily be
			   regularly gridded.  For regularly gridded input data,
			   X and Y are not used, the grid spacing is specified
			   via the XGRID or XVALUES, and YGRID or YVALUES,
			   keywords, and Z must be a two dimensional array.
			   For irregular grids, all three parameters must be present
			   and have the same number of elements.
	KEYWORD PARAMETERS:
	   Input grid description:
		  REGULAR = if set, the Z parameter is a two dimensional array,
					of dimensions (N,M), containing measurements over a
					regular grid.  If any of XGRID, YGRID, XVALUES, YVALUES
					are specified, REGULAR is implied.	REGULAR is also
					implied if there is only one parameter, Z.	If REGULAR is
					set, and no grid (_VALUE or _GRID) specifications are present,
					the respective grid is set to (0, 1, 2, ...).
		  XGRID = contains a two element array, [xstart, xspacing],
				  defining the input grid in the X direction.  Do not specify
				  both XGRID and XVALUES.
		  XVALUES = if present, Xvalues(i) contains the X location
					of Z(i,j).	Xvalues must be dimensioned with N elements.
		  YGRID, YVALUES = same meaning as XGRID, XVALUES except for the
						   Y locations of the input points.
	   Output grid description:

		  GS =	spacing of output grid.
				If present, GS a two-element vector
				[XS, YS], where XS is the horizontal spacing between
				grid points and YS is the vertical spacing. The
				default is based on the extents of X and Y. If the
				grid starts at X value Xmin and ends at Xmax, then the
				default horizontal spacing is  (Xmax - Xmin)/(NX-1).
				YS is computed in the same way. The default grid
				size, if neither NX or NY are specified, is 26 by 26.
		  BOUNDS = a four element array containing the grid limits in X and
				   Y of the output grid:  [ Xmin, Ymin, Xmax, Ymax].
				   If not specified, the grid limits are set to the extend
				   of X and Y.
		  NX = Output grid size in the X direction.	 Default = 26, need
			   not be specified if the size can be inferred by
			   GS and BOUNDS.
		  NY = Output grid size in the Y direction.	 See NX.
		  XOUT = a vector containing the output grid X values.	If this
				 parameter is supplied, GS, BOUNDS, and NX are ignored
				 for the X output grid.	 Use this parameter to specify
				 irregular spaced output grids.
		  YOUT = a vector containing the output grid in the Y direction.
				If this parameter is supplied, GS, BOUNDS, and NY are
				ignored for the Y output grid.
		  XPOUT, YPOUT = arrays containing X and Y values for the output
				 points.  With these keywords, the output grid need not
				 be regular, and all other output grid parameters are
				 ignored.  XPOUT and YPOUT must have the same number of
				 points, which is also the number of points returned in
				 the result.
	OUTPUTS:
	   A two dimensional floating point array containing the interpolated
	   surface, sampled at the grid points.
	COMMON BLOCKS:
	   None.
	SIDE EFFECTS:
	   None.
	RESTRICTIONS:
	   Limited by the single precision floating point accuracy of the
	   machine.
			   SAMPLE EXECUTION TIMES  (measured on a Sun IPX)
	   # of input points	   # of output points	   Seconds
	   16					   676					   0.19
	   32					   676					   0.42
	   64					   676					   1.27
	   128					   676					   4.84
	   256					   676					   24.6
	   64					   256					   1.12
	   64					   1024					   1.50
	   64					   4096					   1.97
	   64					   16384				   3.32

	PROCEDURE:
	   A minimum curvature spline surface is fitted to the data points
	   described by X,Y, and Z.	 The basis function:
			   C(x0,x1, y0,y1) = d^2 log(d),
	   where d is the distance between (x0,y0), (x1,y1), is used,
	   as described by Franke, R., Smooth interpolation of scattered
	   data by local thin plate splines: Computers Math With Applic.,
	   v.8, no. 4, p. 273-281, 1982.  For N data points, a system of N+3
	   simultaneous equations are solved for the coefficients of the
	   surface.	 For any interpolation point, the interpolated value
	   is:
		 F(x,y) = b(0) + b(1)*x + b(2)*y + Sum(a(i)*C(X(i),x,Y(i),y))

	EXAMPLE:  IRREGULARLY GRIDDED CASES
	   Make a random set of points that lie on a gaussian:
	   n = 15		   # random points
	   x = RANDOMU(seed, n)
	   y = RANDOMU(seed, n)
	   z = exp(-2 * ((x-.5)^2 + (y-.5)^2))	# The gaussian

	   # get a 26 by 26 grid over the rectangle bounding x and y:
	   r = min_curve_surf(z, x, y)	   # Get the surface.
	   # Or: get a surface over the unit square, with spacing of 0.05:
	   r = min_curve_surf(z, x, y, GS=[0.05, 0.05], BOUNDS=[0,0,1,1])
	   # Or: get a 10 by 10 surface over the rectangle bounding x and y:
	   r = min_curve_surf(z, x, y, NX=10, NY=10)

	   # REGULARLY GRIDDED CASES
	   z = randomu(seed, 5, 6)		   # Make some random data
	   # interpolate to a 26 x 26 grid:
	   CONTOUR, min_curve_surf(z, /REGULAR)

	MODIFICATION HISTORY:
	   DMS, RSI, March, 1993.  Written.
	   DMS, RSI, July, 1993.   Added XOUT and YOUT.
	   D. Jones, 2016.	Converted to Python
"""
	import numpy as np
	dtor = np.pi/180.
	s = np.shape(z)				#Assume 2D
	if len(s) == 2:
		nx = s[1]
		ny = s[0]
	if tps: k = 1.0
	else: k = 0.5

	def min_curve_sphere_basis_function( d, TPS=False):
		"""The radial basis function, F(d), for either a Thin plate spline, or
		a minimum curvature surface."""
		d2 = d**2.
		good = where(d2)[0]
		if len(good) != 0:
			if TPS: d2[good] = d2[good] * np.log( d2[good]) 
			else: d2[good] = d2[good] * np.log(d[good])
		return(d2)

	reg = False
	if len(xgrid) == 2:
		x = np.arange(nx) * xgrid[1] + xgrid[0]
		reg = True
	elif len(xvalues) > 0:
		if len(xvalues) != nx:
				print(('Xvalues must have %i elements.'%nx))
		x = xvalues[:]
		reg = True

	if len(ygrid) == 2:
		y = np.arange(ny) * ygrid[1] + ygrid[0]
		reg = True
	elif len(yvalues) > 0:
		if len(yvalues) != ny:
				print(('Yvalues must have %i elements.'%ny))
		y = yvalues[:]
		reg = True

	if reg:
		if len(s) != 2: print('Z array must be 2D for regular grids')
		if len(x) != nx: x = np.arange(nx)
		if len(y) != ny: y = np.arange(ny)
		x = np.matrix(np.ones([ny,1]))*x	   #Expand to full arrays.
		y = np.transpose(np.ones([nx,1])*np.matrix(y))

	if len(np.shape(x)) == 2:
		n = np.shape(x)[0]*np.shape(x)[1]
	else: n = len(x)
	if len(np.shape(y)) == 2:
		leny = np.shape(y)[0]*np.shape(y)[1]
	else: leny = len(y)
	if len(np.shape(z)) == 2:
		lenz = np.shape(z)[0]*np.shape(z)[1]
	else: lenz = len(z)

	if n != leny or n != lenz:
		print('x, y, and z must have same number of elements.')

	if len(xpout) > 0: #Explicit output locations?
		if len(ypout) != len(xpout):
			print('XPOUT and YPOUT must have same number of points')
	else:				#Regular grid
		if len(bounds) < 4: #Bounds specified?
			xmin = np.min(x); xmax = np.max(x)
			ymin = np.min(y); ymax = np.max(y)
			bounds = [xmin, ymin, xmax, ymax]

		
		if len(gs) < 2: #GS specified?	No.
			if len(nx0) <= 0: nx = 26 
			else: nx = nx0[:]
			if len(ny0) <= 0: ny = 26
			else: ny = ny0[:]
			gs = [(bounds[2]-bounds[0])/(nx-1.),
				  (bounds[3]-bounds[1])/(ny-1.)]
		else:			 #GS is specified?
			if len(nx0) <= 0:
				nx = ceil((bounds[2]-bounds[0])/gs[0]) + 1
			else: nx = nx0
			if len(ny0) <= 0:
				ny = ceil((bounds[3]-bounds[1])/gs[1]) + 1
			else: ny = ny0

		if len(xout) > 0: #Output grid specified?
			nx = len(xout)
			xpout = xout
		else: xpout = gs[0] * np.arange(nx) + bounds[0]

		if len(yout) > 0:
			ny = len(yout)
			ypout = yout
		else: ypout = gs[1] * np.arange(ny) + bounds[1]

		xpout = np.matrix(np.ones([ny,1]))*xpout
		ypout = np.transpose(np.ones([nx,1])*np.matrix(ypout))
						 #Regular grid


	zmin = np.min(z); zmax = np.max(z)
	zscale = 1.0 / (zmax - zmin)	 #Normalize Z, other coords are already
								#on unit sphere.

	if sphere:		   #Spherical?
		if const: cbase = 1
		else: cbase = 4 #If set, use a constant baseline
		m = n + cbase				# # of eqns to solve
		a = np.zeros([m,m])

		# For thin-plate-splines, terms are r^2 log(r^2).  For min curve surf,
		# terms are r^2 log(r).
		cosz = np.cos(y * dtor)
		p3 = [[np.cos(x * dtor) * cosz], #Dims = [n,3]
			  [np.sin(x * dtor) * temporary(cosz)],
			  [np.sin(y * dtor)]]

		for i in range(n-1):
			for j in range(i+1,n): #Compute matrix
				d = np.arccos(total(p3[:,i] * p3[:,j])) # = angular distance, 0 to !PI
				a[j+cbase,i+cbase] = d
				a[i+cbase,j+cbase] = d

		#  Apply basis function
		a[cbase, cbase] = \
			min_curve_sphere_basis_function(a[cbase:, cbase:], TPS=tps)
		for i in range(cbase, m): a[i,i]=0.0 #Zero the diagonals

		a[cbase:m,0] = 1		  # fill rest of array
		a[0,cbase:m] = 1.
		if const == 0:
			for i in range(3): #Add x, y, and Z terms
				r = p3[i,:]
				a[i+1,cbase] = r
				a[cbase,i+1] = transpose(r)

		b[cbase] = (np.reshape(z,(n,1)) - zmin) * zscale

		atemp = a[:]
		ludc, atemp, Index			#Solve the system of equations
		c = LUSOL(temporary(atemp), index, b)
		cc = c[cbase:]

		# For min_curve_surf, divide c[4:] by 2 cause we use d^2 rather than d.
		# common scale factor
		# if keyword_set(tps) eq 0 then c[4] = c[4:]/2.0

		dims = np.shape(xpout)
		nxny = len(ypout)
		cosz = np.reshape(np.cos(ypout * dtor), nxny)
								# Cartesian coords of output grid [nxny,3]
		p3o = [[cosz * np.cos(xpout * dtor)],
			   [temporary(cosz) * np.sin(xpout * dtor)],
			   [np.reshape(np.sin(ypout * dtor), nxny)]]

		if const == 0:
			s = c[0] + c[1] * p3o[0,:] + c[2] * p3o[1,:] + c[3] * p3o[2,:]
			s = c[0] + c[1] * p3o[0,:] + c[2] * p3o[1,:] + c[3] * p3o[2,:]
			s = np.reshape(s, (dims[1], dims[0]), clobber=True)
		else: s = np.zeros([dims[0],dims[1]])+c[0]

		for i in range(n):		#Angular Distance
			#d = np.arccos(p3o # transpose(p3[:,i]))
			s = s + c[i+cbase] * min_curve_sphere_basis_function(d, TPS=tps)

	else:				#Planar case

		m = n + 3			# # of eqns to solve
		a = np.zeros([m,m])

		x0 = np.min(x); xmax = np.max(x)
		y0 = np.min(y); ymax = np.max(y)
		denom = xmax - x0
		cols = np.where(denom < ymax - y0)[0]
		if len(cols): denom[cols] = ymax - y0
		scale = 1./ denom #Scale factor for unit square
		xs = np.array((x - x0) / scale)		  #Scale into 0-1 rectangle to enhance accuracy
		ys = np.array((y - y0) / scale)

		xsshape = np.shape(xs); ysshape = np.shape(ys)
		if len(xsshape) > 1:
			xs = np.reshape(xs,xsshape[0]*xsshape[1])
		if len(ysshape) > 1:
			ys = np.reshape(ys,ysshape[0]*ysshape[1])
		for i in range(n-1):
			# For each point, find the distance to all other points.
			d = (xs[i]-xs[i+1:n])**2 + (ys[i]-ys[i+1:n])**2 #Distance^2
			cols = np.where(d < tol)[0]
			if len(cols): d[cols] = tol
			d = d * np.log(d)* k	  #TPS: d^2 * alog(d^2)), MCS: d^2 * alog(d))
			a[i+1:n,i+3] = d[:]
			a[i,i+4:n+3] = d[:]

		a[0:n,0] = 1			  # fill rest of array
		a[0:n,1] = np.reshape(xs,(1,n))
		a[0:n,2] = np.reshape(ys,(1,n))

		a[n,3:m] = 1.
		a[n+1,3:] = xs #np.reshape(xs, (n, 1))
		a[n+2,3:] = ys #np.reshape(ys, (n, 1))

		b = np.zeros(m)
		b[0:n] = ((np.reshape(z,(1,n)) - zmin) * zscale)

		import scipy.linalg
		lu_and_piv = scipy.linalg.lu_factor(a)				#solution using LU decomposition
		c = scipy.linalg.lu_solve(lu_and_piv, b)

		# For min_curve_surf, divide c[3:] by 2 cause we use d rather than
		# the d^2 we use for TPS.
		if not tps: c[3:] = c[3:]/2.0

		xpouts = (xpout - x0) / scale #reapply scale factor to output grid
		ypouts = (ypout - y0) / scale
		s = c[0] + c[1] * xpouts + c[2] * ypouts #First terms

		for i in range(n):		#This loop takes all the time.
			d = (np.array(xpouts)-xs[i])**2 + (np.array(ypouts)-ys[i])**2 #Distance ^2
			cols = np.where(d < tol)[0]
			if len(cols): d[cols] = tol
			s = s + d * np.log(d)* c[i+3]

	return(s / zscale + zmin)	  #Rescale data

def getfmod(nowp, nowfilt, nowlam, 
			nowflux, nowphase, standard, 
			zp2,ab=False):
	import numpy as np
	from scipy import interpolate
	import os
	n = len(nowp)
	fmod = np.zeros(n)

	nph = len(nowphase)

	#interpfunc = interpolate.interp1d(nowphase,np.arange(nph),kind='linear')
	#ind = interpfunc(nowp)
	f = interpolate.InterpolatedUnivariateSpline(
		nowphase,np.arange(nph),
		k=1)
	ind = f(nowp)
	nfu=np.unique(nowfilt)
	zp=np.zeros(len(nfu))

	for i in range(len(nfu)):

		if 'Bessell' in nfu[i]:
			standard =  '../templates/flatnu.dat' # '$SNDATA_ROOT/standards/vegased_2004_stis.txt'
		else:
			standard = '../templates/flatnu.dat' #'$SNDATA_ROOT/standards/flatnu.dat'
		if ab:
			print('HAAAAAACK AB MAGS FOR 2012fr!!!')
			standard = '../templates/flatnu.dat' # '$SNDATA_ROOT/standards/flatnu.dat'
			
		fnu1,fnu2 = np.loadtxt(os.path.expandvars(standard),unpack=True)
		filt1,filt2 = np.loadtxt(nfu[i],unpack=True)

		#interpfunc = interpolate.interp1d(filt1,filt2,kind='linear')
		#xy = interpfunc(fnu1)
		xy = np.interp(filt1,fnu1,fnu2)
		zp[i]=2.5*np.log10(idl_tabulate(filt1,xy*filt1*filt2)/idl_tabulate(filt1,filt1*filt2))

	for i in range(n):
		j = int(ind[i])
		dj = ind[i] - j
		fj = nowflux[:,j]
		if dj > 0: dfj = nowflux[:,j+1] - fj
		else: dfj = 0.

		spc = fj + dj*dfj
		col = np.where(spc <= 0)
		if len(col[0]): spc[col] = 1e-99

		xx=np.where(nowfilt[i] == nfu)

		fmod[i] = 10.0**(-0.4*(synphot(nowlam,spc,nowfilt[i],zp[xx])+zp2[i] - 27.5))

	return(fmod)

def idl_tabulate(x, f, p=5) :
	import scipy.integrate
	import numpy as np
	def newton_cotes(x, f) :
		if x.shape[0] < 2 :
			return 0
		rn = (x.shape[0] - 1) * (x - x[0]) / (x[-1] - x[0])
		weights = scipy.integrate.newton_cotes(rn)[0]
		return (x[-1] - x[0]) / (x.shape[0] - 1) * np.dot(weights, f)
	ret = 0
	for idx in range(0, x.shape[0], p - 1) :
		ret += newton_cotes(x[idx:idx + p], f[idx:idx + p])
	return ret

def synphot(x,spc,pb,zp,plot=False,oplot=False,allowneg=False):
	import numpy as np
	x1=pb.split('/')
	nx=len(x1)
	xfin=x1[nx-1]
	fp='/'
	for i in range(nx-1): fp=fp+x1[i]+'/'


	import pysynphot.spectrum as S
	sp = S.Vega
	mag = zp - 2.5 * np.log10( synflux(x,spc,pb,plot=plot,oplot=oplot,
										 allowneg=allowneg))
	vegamag = zp - 2.5 * np.log10( synflux(x,sp(x),pb,plot=plot,oplot=oplot,
										 allowneg=allowneg))

	return(mag)

def synflux(x,spc,pb,plot=False,oplot=False,allowneg=False):
	import numpy as np

	nx = len(x)
	pbphot = 1
	pbx,pby = np.loadtxt(pb,unpack=True)

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

	if x[0] > pbx[0]:
		print("spectrum doesn''t go blue enough for passband!")

	if x[nx-1] < pbx[npbx-1]:
		print("spectrum doesn''t go red enough for passband!")

	g = np.where((x >= pbx[0]) & (x <= pbx[npbx-1]))  # overlap range

	pbspl = np.interp(x[g],pbx,pby)#,kind='cubic')

	if not allowneg: 
		pbspl = pbspl
		col = np.where(pbspl < 0)[0]
		if len(col): pbspl[col] = 0

	if (pbphot): pbspl *= x[g]


	res = np.trapz(pbspl*spc[g],x[g])/np.trapz(pbspl,x[g])

	return(res)

def meritfunc(x,p=None,f=None,fluxerr=None,fmod=None,px=None,lx=None,
			  filt=None,ufilt=None,nufilt=None,ndegr=None,random=None,
			  data=None):
	if data:
		p,f,fluxerr,fmod,px,lx,filt,ufilt,nufilt,ndegr = data
	import numpy as np
	efrac = (np.abs(p)**0.1)/7. - 0.1
	col = np.where(efrac > 0.12)[0]
	if len(col): efrac[col] = 0.12
	col = np.where(efrac < 0.03)[0]
	if len(col): efrac[col] = 0.03

	err = np.sqrt(fluxerr**2. + (efrac*f)**2)
	col = np.where(err < 0.003*np.max(f))[0]
	if len(col): err[col] = 0.003*np.max(f)

	model = modelfunc(x,fmod,px,lx,filt,ufilt,nufilt,ndegr,random)
	model[np.where(model != model)] = 0
	cols = np.where(model >= 0)

	merit = np.sum(((f[cols] - model[cols])/err[cols])**2.)
	print(merit)
	return(merit)

def poly(x,c):
	n = len(c) - 1
	if n == 0: return(x*0 + c[0])
	
	y = c[n]
	for i in range(n-1,-1,-1): 
		y = y*x + c[i]
	return(y)

def modelfunc(x, fmod, px, lx, filt, ufilt, nufilt, ndegr,random):
	import numpy as np
	from scipy import interpolate

	norm = x[0]
	pwarp = x[1:4]

	nmod = fmod-fmod
	for i in range(nufilt):
		qq = np.where(filt == ufilt[i])
		f = interpolate.InterpolatedUnivariateSpline(
			poly(px[qq],pwarp),fmod[qq],
			k=1)
		nmod[qq] = f(px[qq])
		#nmod[qq] = np.interp(px[qq],poly(px[qq],pwarp),fmod[qq])
		# IS THIS RIGHT????

	xx = np.arange(-1.2,1.21,2.4/(ndegr-1))

	yy = xx[:]
	mm = np.zeros([ndegr,ndegr]) + random #\
#		 np.random.standard_normal((ndegr,ndegr))*1e-10

	ind = 4
	for i in range(ndegr):
		for j in range(ndegr):
			if x[ind] > 0.2 and x[ind] < 5.:
				mm[j,i] += x[ind]
			elif x[ind] < 0.2: mm[j,i] += 0.2
			else: mm[j,i] += 5.
			ind += 1

	mult = min_curve_surf(mm,xvalues=xx,yvalues=yy,xpout=px,ypout=lx)
	mincol = np.where(mult < 0.2)[0]
	maxcol = np.where(mult > 5)[0]
	if len(mincol): mult[mincol] = 0.2
	if len(maxcol): mult[maxcol] = 5.

	mvar = nmod * mult / norm
	col = np.where(mvar < 0)[0]
	if len(col): mvar[col] = 0

	return(mvar)

def filt2flam(filtfile):
	"""filter response to effective wavelength"""
	import numpy as np
	wavelength, response = np.loadtxt(filtfile,unpack=True)
	efflam = np.sum(response*wavelength)/np.sum(response)
	return(efflam)

def amoeba(var,scale,func,ftolerance=1.e-5,
		   xtolerance=1.e-4,itmax=5000,data=None):
	'''Use the simplex method to maximize a function of 1 or more variables.
	
	   Input:
			  var = the initial guess, a list with one element for each variable
			  scale = the search scale for each variable, a list with one
					  element for each variable.
			  func = the function to maximize.
			  
	   Optional Input:
			  ftolerance = convergence criterion on the function values (default = 1.e-4)
			  xtolerance = convergence criterion on the variable values (default = 1.e-4)
			  itmax = maximum number of iterations allowed (default = 500).
			  data = data to be passed to func (default = None).
			  
	   Output:
			  (varbest,funcvalue,iterations)
			  varbest = a list of the variables at the maximum.
			  funcvalue = the function value at the maximum.
			  iterations = the number of iterations used.

	   - Setting itmax to zero disables the itmax check and the routine will run
		 until convergence, even if it takes forever.
	   - Setting ftolerance or xtolerance to 0.0 turns that convergence criterion
		 off.  But do not set both ftolerance and xtolerance to zero or the routine
		 will exit immediately without finding the maximum.
	   - To check for convergence, check if (iterations < itmax).
			  
	   The function should be defined like func(var,data) where
	   data is optional data to pass to the function.

	   Example:
	   
		   import amoeba
		   def afunc(var,data=None): return 1.0-var[0]*var[0]-var[1]*var[1]
		   print amoeba.amoeba([0.25,0.25],[0.5,0.5],afunc)

	   Version 1.0 2005-March-28 T. Metcalf
			   1.1 2005-March-29 T. Metcalf - Use scale in simsize calculation.
											- Use func convergence *and* x convergence
											  rather than func convergence *or* x
											  convergence.
			   1.2 2005-April-03 T. Metcalf - When contracting, contract the whole
											  simplex.
	   '''

	nvar = len(var)		  # number of variables in the minimization
	nsimplex = nvar + 1	  # number of vertices in the simplex
	
	# first set up the simplex

	simplex = [0]*(nvar+1)	# set the initial simplex
	simplex[0] = var[:]
	for i in range(nvar):
		simplex[i+1] = var[:]
		simplex[i+1][i] += scale[i]

	fvalue = []
	for i in range(nsimplex):  # set the function values for the simplex
		fvalue.append(func(simplex[i],data=data))

	# Ooze the simplex to the maximum

	iteration = 0
	
	while 1:
		# find the index of the best and worst vertices in the simplex
		ssworst = 0
		ssbest	= 0
		for i in range(nsimplex):
			if fvalue[i] > fvalue[ssbest]:
				ssbest = i
			if fvalue[i] < fvalue[ssworst]:
				ssworst = i
				
		# get the average of the nsimplex-1 best vertices in the simplex
		pavg = [0.0]*nvar
		for i in range(nsimplex):
			if i != ssworst:
				for j in range(nvar): pavg[j] += simplex[i][j]
		for j in range(nvar): pavg[j] = pavg[j]/nvar # nvar is nsimplex-1
		simscale = 0.0
		for i in range(nvar):
			simscale += abs(pavg[i]-simplex[ssworst][i])/scale[i]
		simscale = simscale/nvar

		# find the range of the function values
		fscale = (abs(fvalue[ssbest])+abs(fvalue[ssworst]))/2.0
		if fscale != 0.0:
			frange = abs(fvalue[ssbest]-fvalue[ssworst])/fscale
		else:
			frange = 0.0  # all the fvalues are zero in this case
			
		# have we converged?
		if (((ftolerance <= 0.0 or frange < ftolerance) and	   # converged to maximum
			 (xtolerance <= 0.0 or simscale < xtolerance)) or  # simplex contracted enough
			(itmax and iteration >= itmax)):			 # ran out of iterations
			return simplex[ssbest],fvalue[ssbest],iteration

		# reflect the worst vertex
		pnew = [0.0]*nvar
		for i in range(nvar):
			pnew[i] = 2.0*pavg[i] - simplex[ssworst][i]
		fnew = func(pnew,data=data)
		if fnew <= fvalue[ssworst]:
			# the new vertex is worse than the worst so shrink
			# the simplex.
			for i in range(nsimplex):
				if i != ssbest and i != ssworst:
					for j in range(nvar):
						simplex[i][j] = 0.5*simplex[ssbest][j] + 0.5*simplex[i][j]
					fvalue[i] = func(simplex[i],data=data)
			for j in range(nvar):
				pnew[j] = 0.5*simplex[ssbest][j] + 0.5*simplex[ssworst][j]
			fnew = func(pnew,data=data)
		elif fnew >= fvalue[ssbest]:
			# the new vertex is better than the best so expand
			# the simplex.
			pnew2 = [0.0]*nvar
			for i in range(nvar):
				pnew2[i] = 3.0*pavg[i] - 2.0*simplex[ssworst][i]
			fnew2 = func(pnew2,data=data)
			if fnew2 > fnew:
				# accept the new vertex in the simplex
				pnew = pnew2
				fnew = fnew2
		# replace the worst vertex with the new vertex
		for i in range(nvar):
			simplex[ssworst][i] = pnew[i]
		fvalue[ssworst] = fnew
		iteration += 1
		#if __debug__: print ssbest,fvalue[ssbest]

def aplot(p,f,fmod,filt,ufilt,title=''):
	import matplotlib.pylab as plt
	nufilt = len(ufilt)

	#!p.multi = [0,1,nufilt]
	ax,leff = [],[]
	for j in range(nufilt):
		ax += [plt.subplot(nufilt,1,j+1)]
		leff += [filt2flam(ufilt[j])]
	
	ax,ufilt = np.array(ax),np.array(ufilt)[np.argsort(leff)]
	for j in range(nufilt):
		zz = (filt == ufilt[j])
		if j == 0:
			ax[j].plot(p[zz],f[zz],'^',label='Light Curve')
			ax[j].plot(p[zz],fmod[zz],'x',label='Synthetic photometry \n from mangled spectrum')
			ax[j].legend(numpoints=1,loc='upper right')
		else:
			ax[j].plot(p[zz],f[zz],'^')
			ax[j].plot(p[zz],fmod[zz],'x')

		if j == 0 and title:
			ax[j].set_title("""%s
%s"""%(title,ufilt[j].split('/')[-1]),fontsize=12)
		else:
			ax[j].set_title(ufilt[j].split('/')[-1],fontsize=12)			
		if j != nufilt-1:
			ax[j].set_xticklabels([])
		else:
			ax[j].set_xlabel('Phase')
		ax[j].set_ylabel('Flux')

	return()

def savitzky_golay(y, window_size=5, order=3, deriv=0):
	r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
	The Savitzky-Golay filter removes high frequency noise from data.
	It has the advantage of preserving the original shape and
	features of the signal better than other types of filtering
	approaches, such as moving averages techhniques.
	Parameters
	----------
	y : array_like, shape (N,)
		the values of the time history of the signal.
	window_size : int
		the length of the window. Must be an odd integer number.
	order : int
		the order of the polynomial used in the filtering.
		Must be less then `window_size` - 1.
	deriv: int
		the order of the derivative to compute (default = 0 means only smoothing)
	Returns
	-------
	ys : ndarray, shape (N)
		the smoothed signal (or it's n-th derivative).
	Notes
	-----
	The Savitzky-Golay is a type of low-pass filter, particularly
	suited for smoothing noisy data. The main idea behind this
	approach is to make for each point a least-square fit with a
	polynomial of high order over a odd-sized window centered at
	the point.
	Examples
	--------
	t = np.linspace(-4, 4, 500)
	y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
	ysg = savitzky_golay(y, window_size=31, order=4)
	import matplotlib.pyplot as plt
	plt.plot(t, y, label='Noisy signal')
	plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
	plt.plot(t, ysg, 'r', label='Filtered signal')
	plt.legend()
	plt.show()
	References
	----------
	.. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
	   Data by Simplified Least Squares Procedures. Analytical
	   Chemistry, 1964, 36 (8), pp 1627-1639.
	.. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
	   W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
	   Cambridge University Press ISBN-13: 9780521880688
	"""
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError as msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = list(range(order+1))
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv]
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m, y, mode='valid')

def str2num(s) :
	""" convert a string to an int or float, as appropriate.
	If neither works, return the string"""
	try: return int(s)
	except ValueError:
		try: return float(s)
		except ValueError: return( s )

if __name__ == "__main__":
	usagestring="""
mangle.py <snfile> <templatefile> [options]

snfile: SNANA-formatted light curve file
template file: Template SED, with 3 header:
			   # phase wavelength flux

examples: 

./mangle.py CSP-2004fe.snana.dat snIc_flux.v1.3b.txt
./mangle.py 2008bo.snana.dat 2008bo_interpolatedSpectra.txt --mkplot --filterfiles B,V,r,i keplercam_B.dat,keplercam_V.dat,r.dat,i.dat --smoothlc --clobber --filtpath IIBjson/filters
./mangle.py IIBjson/SN2011dh.snana.dat IIBjson/SN2011dh.sed --mkplot --filterfiles U,B,V,R,I Bessell90_U.dat,Bessell90_B.dat,Bessell90_V.dat,Bessell90_R.dat,Bessell90_I.dat --filtpath IIBjson/filters
./mangle.py IIBjson/SN2008ax.snana.dat IIBjson/SN2008ax.sed --mkplot --filterfiles U,B,V,R,I,u,g,r,i,z keplercam_U.dat,keplercam_B.dat,keplercam_V.dat,SNLS3_4shooter2_R.dat,SNLS3_4shooter2_I.dat,u.dat,g.dat,r.dat,i.dat,z.dat --filtpath IIBjson/filters
./mangle.py IIBjson/SN2008ax.snana.dat IIBjson/SN2008ax.sed --mkplot --filterfiles U,B,V,R,I Bessell90_U.dat,Bessell90_B.dat,Bessell90_V.dat,Bessell90_R.dat,Bessell90_I.dat --filtpath IIBjson/filters
./mangle.py IIBjson/SN1993J.snana.dat IIBjson/SN1993J.sed --mkplot --filterfiles U,B,V,R,I Bessell90_U.dat,Bessell90_B.dat,Bessell90_V.dat,Bessell90_R.dat,Bessell90_I.dat --filtpath IIBjson/filters

./mangle.py IIBjson/SN1991bg.snana.dat sn91bg-nugent.SED --mkplot --filterfiles U,B,V,R,I Bessell90_U.dat,Bessell90_B.dat,Bessell90_V.dat,Bessell90_R.dat,Bessell90_I.dat --smoothlc --clobber --filtpath IIBjson/filters --tol 1 --niter 15
./mangle.py sn1999by.snana.dat sn91bg-nugent.SED --mkplot --filterfiles U,B,V,R,I Bessell90_U.dat,Bessell90_B.dat,Bessell90_V.dat,Bessell90_R.dat,Bessell90_I.dat --smoothlc --clobber --filtpath IIBjson/filters --tol 1 --niter 15
./mangle.py IIBjson/SN1998de.snana.dat sn91bg-nugent.SED --mkplot --filterfiles U,B,V,R,I Bessell90_U.dat,Bessell90_B.dat,Bessell90_V.dat,Bessell90_R.dat,Bessell90_I.dat --smoothlc --clobber --filtpath IIBjson/filters --scale 1e-25 --tol 1 --niter 15
./mangle.py IIBjson/SN2000dk.snana.dat sn91bg-nugent.SED --mkplot --filterfiles U,B,V,R,I Bessell90_U.dat,Bessell90_B.dat,Bessell90_V.dat,Bessell90_R.dat,Bessell90_I.dat --smoothlc --clobber --filtpath IIBjson/filters --scale 1e-25 --tol 1 --niter 15
./mangle.py IIBjson/SN2000cn.snana.dat sn91bg-nugent.SED --mkplot --filterfiles U,B,V,R,I Bessell90_U.dat,Bessell90_B.dat,Bessell90_V.dat,Bessell90_R.dat,Bessell90_I.dat --smoothlc --clobber --filtpath IIBjson/filters --scale 1e-25 --tol 1 --niter 15

"""

	import os
	import optparse

	mg = mangle()

	# read in the options from the param file and the command line
	# some convoluted syntax here, making it so param file is not required
	parser = mg.add_options(usage=usagestring)
	options,  args = parser.parse_args()

	try:
		snfile,tmplfile = args[0:2]
	except:
		import sys
		print(usagestring)
		print('Error : incorrect or wrong number of arguments')
		sys.exit(1)
		
	import numpy as np
	import matplotlib.pylab as plt

	mg.options = options
	mg.verbose = options.verbose
	mg.clobber = options.clobber

	mg.main(snfile,tmplfile)
