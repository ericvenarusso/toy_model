import pytest

import os

from src.ml import train
from src.ml import scoring
    
def test_train_file_existence():
     assert os.path.isfile('src/ml/train.py')

def test_scoring_file_existence():
     assert os.path.isfile('src/ml/scoring.py')

def test_gitignore_file_existence():
    assert os.path.isfile('.gitignore')

def test_requirements_file_existence():
    assert os.path.isfile('requirements.txt')

def test_dockerfile_file_existence():
    assert os.path.isfile('Dockerfile')

def test_train_function_existence():
    assert "train" in list(train.__dict__.keys())

def test_predict_function_existence():
    assert "predict" in list(scoring.__dict__.keys())
