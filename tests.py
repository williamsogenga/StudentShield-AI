"""Simple automated checks; run python tests.py."""
import unittest
import numpy as np
import pandas as pd
from core import RAW_FEATURES, ALL_FEATURES, prepare_features, risk_band, support_suggestions
from generate_demo_data import generate

class StudentShieldTests(unittest.TestCase):
    def setUp(self):
        self.data=generate(n=120,seed=123)

    def test_feature_shape(self):
        x=prepare_features(self.data)
        self.assertEqual(x.shape,(120,len(ALL_FEATURES)))
        self.assertEqual(x.isna().sum().sum(),0)

    def test_rates_bounded(self):
        x=prepare_features(self.data)
        self.assertTrue(x["Sem 2 approval rate"].between(0,1).all())

    def test_missing_column_rejected(self):
        with self.assertRaises(ValueError):prepare_features(self.data.drop(columns=[RAW_FEATURES[0]]))

    def test_model_bands(self):
        self.assertEqual(risk_band(.8),"High review priority")
        self.assertEqual(risk_band(.55),"Moderate review priority")
        self.assertEqual(risk_band(.1),"Routine monitoring")

    def test_support_rule(self):
        r=self.data.iloc[0].to_dict()
        r["Tuition fees up to date"]=0
        self.assertTrue(any("bursaries" in x for x in support_suggestions(r)))

if __name__=="__main__":unittest.main(verbosity=2)
