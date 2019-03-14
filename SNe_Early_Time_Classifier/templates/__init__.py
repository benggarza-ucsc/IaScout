import os

tmplbase = os.path.dirname(globals()['__file__'])
if tmplbase == '':  tmplbase = '.'

vegafile = os.path.join(tmplbase,'synphot_vega.dat')
abfile = os.path.join(tmplbase,'flatnu.dat')
