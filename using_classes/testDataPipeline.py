import unittest
import pandas as pd
from pipelines import DataPipeline

class TestaDataPipeline(unittest.TestCase):

    def setUp(self):
        self.datapath = "../data/amostra10.csv"
        self.raw_data = pd.read_csv(self.datapath)
        self.pipeline = DataPipeline(self.datapath)

    def test_last_procedure(self):
        newPipeline = DataPipeline(self.datapath)
        self.assertEqual(newPipeline.last_procedure, "Leitura da Base")
    
    def test_procedures(self):
        newPipeline = DataPipeline(self.datapath)
        self.assertEqual(newPipeline.last_procedure, ['Leitura do Arquivo: ../data/amostra10.csv'])

    def test_removeColumns(self):
        testpath = "testeRemoveColunas.csv"
        newPipeline = DataPipeline(testpath)
        newPipeline.step00_remove_columns()
        expected = pd.DataFrame({"OK": [3, 6, 9]})
        self.assertTrue( all(expected == newPipeline.out_data) )

if __name__ == "__main__":
    unittest.main()