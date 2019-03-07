*********************************************************
Creating New SN Templates from the Open Supernova Catalog
*********************************************************

Downloading and Converting the Data
===================================

Download a SN from the Open Supernova Catalog (https://sne.space/)
by clicking `Download All Data`.  From this data, make a light curve
file and a spectrum.

usage::

  import mkInterpSpec
  mkInterpSpec.mkSpec()
  mkInterpSpec.mkPhotFile()
  
This is not command-line executable yet, but can be easily.
The spectra and lightcurve output files are then fed into
the mangling code.

Mangling the Spectrum to Match the LC Data
==========================================

usage::

  python mangle.py <lcfile> <sedfile>

