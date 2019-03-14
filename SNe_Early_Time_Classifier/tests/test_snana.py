import unittest
from SNe_Early_Time_Classifier.util import snana
from SNe_Early_Time_Classifier.mangle.LightCurves import lcbase

class SNANA_Test(unittest.TestCase):

	def test_read_ascii(self):
		
		sn = snana.SuperNova(os.path.join(lcbase,'Ia/SN2011fe.snana.dat')
		self.assertTrue('FLUXCAL' in sn.__dict__.keys())
		self.assertTrue(len(sn.FLUXCAL) > 1)
		self.assertTrue('FLUXCALERR' in sn.__dict__.keys())
		self.assertTrue(len(sn.FLUXCALERR) > 1)
		
if __name__ == "__main__":
	unittest.main()
