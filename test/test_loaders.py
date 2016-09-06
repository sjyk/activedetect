import unittest
from activedetect.loaders.csv_loader import CSVLoader

class TestLoad(unittest.TestCase):
	
	def testCSVLoader(self):
		"""
		Tests the CSV Loading primatives
		"""
		c = CSVLoader()
		l = c.loadFile('test/resources/f1.csv')
		assert(c.delim == (',', '"'))

		c = CSVLoader()
                l = c.loadFile('test/resources/f2.csv')
		assert(c.delim == (':', '"'))

		c = CSVLoader()
                l = c.loadFile('test/resources/f3.csv')
		assert(c.delim == (',', '"'))
		assert(len(l[0]) == 2)

		c = CSVLoader()
		l = c.loadFile('test/resources/f4.csv')
		assert(c.delim == (',', '"'))

		c = CSVLoader()
                l = c.loadFile('test/resources/f5.csv')
                assert(c.delim == ('\t', '"'))

if __name__ == '__main__':
    unittest.main()

