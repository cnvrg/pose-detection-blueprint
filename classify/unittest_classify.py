import unittest
import pose_classify
import yaml
from yaml.loader import SafeLoader
import os
import numpy as np
import pandas as pd

YAML_ARG_TO_TEST = "test_arguments"

class TestLoadPoseLandmarks(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Parse the pose classify  arguments 
        cfg_path = os.path.dirname(os.path.abspath(__file__))
        cfg_file = cfg_path + "/" + "test_config.yaml"
        self.test_cfg = {}
        with open(cfg_file) as c_info_file:
           self.test_cfg = yaml.load(c_info_file, Loader=SafeLoader)
        self.test_cfg = self.test_cfg[YAML_ARG_TO_TEST]
        self.train_data = self.test_cfg['train_data']
        self.test_data = self.test_cfg['test_data']
        self.binary_y_true = self.test_cfg['binary_y_true']

    #test pose landmarks data
    def test_load_pose_landmarks(self):
        X, y, class_names, dataframe = pose_classify.load_pose_landmarks(self.train_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(class_names, np.ndarray)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(dataframe, pd.DataFrame)

    #test classification report with data
    def test_classification_report(self):
        X, y, class_names, _ = pose_classify.load_pose_landmarks(self.train_data)
        X_test, y_test, _, df_test = pose_classify.load_pose_landmarks(self.test_data)
        report = pose_classify.classification_Report(X_test, y_test, class_names)    
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1-score', report)
        self.assertIn('support', report)
        self.assertIn('accuracy', report)
        self.assertIn('macro avg', report)
        self.assertIn('weighted avg', report)


    # def test_binary_classification_report(self):

    #     y_true = self.test_cfg['binary_y_true']
    #     y_pred = self.test_cfg['binary_y_pred']
    #     report = pose_classify.classification_Report(y_true, y_pred)

    #     self.assertIn('0', report)
    #     self.assertIn('1', report)
    #    self.assertIn('accuracy', report)
    
    # test with empty data set with y_true and y_pred    
    def test_empty_input_data(self):
        y_true = self.test_cfg['test_y_true']
        y_pred = self.test_cfg['test_y_pred']
        with self.assertRaises(KeyError) as context:
            report = pose_classify.classification_Report(y_true, y_pred)
        self.assertEqual(str(context.exception), "'pop from an empty set'")


        
              
if __name__ == '__main__':
    unittest.main()