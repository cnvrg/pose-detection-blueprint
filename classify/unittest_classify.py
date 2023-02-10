import unittest
import pose_classify
import yaml
from yaml.loader import SafeLoader
import os
import PIL
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

YAML_ARG_TO_TEST = "test_arguments"

class TestClassify(unittest.TestCase):
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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
        X_test, y_test, _, df_test = pose_classify.load_pose_landmarks(self.test_data)
        report = pose_classify.model_init(X_train,y_train,X_val,y_val,class_names,output_format=None)   
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1-score', report)
        self.assertIn('support', report)
        self.assertIn('accuracy', report)
        self.assertIn('macro avg', report)
        self.assertIn('weighted avg', report)

    #test classification report values
    def test_classification_report1(self):
        X, y, class_names, _ = pose_classify.load_pose_landmarks(self.train_data)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
        X_test, y_test, _, df_test = pose_classify.load_pose_landmarks(self.test_data)
        report = pose_classify.model_init(X_train,y_train,X_val,y_val,class_names,output_format=True)   
        for i in class_names:
            self.assertGreaterEqual(report[i]['precision'],0.80)  
            self.assertGreaterEqual(report[i]['recall'],0.80)
            self.assertGreaterEqual(report[i]['f1-score'],0.90)
            self.assertGreaterEqual(report[i]['support'],0.60)
            
    #test classification report with empty data  
    def test_empty_data(self):
          y_true = []
          y_pred = []
          target_names = ['class_1', 'class_2', 'class_3']
          with self.assertRaises(ValueError) as context:
            classification_report(y_true, y_pred, target_names=target_names)
          self.assertEqual(str(context.exception), "Number of classes, 0, does not match size of target_names, 3. Try specifying the labels parameter")

                    
                    
if __name__ == '__main__':
    unittest.main()
