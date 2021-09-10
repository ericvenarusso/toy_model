import pytest

import os

from app.ml import trainer
from app.ml import predictor


def test_train_file_existence():
     assert os.path.isfile('app/ml/trainer.py')

def test_scoring_file_existence():
     assert os.path.isfile('app/ml/predictor.py')

def test_producer_file_existence():
    assert os.path.isfile('producer/produce.py')

def test_gitignore_file_existence():
    assert os.path.isfile('.gitignore')

def test_requirements_file_existence():
    assert os.path.isfile('requirements.txt')

def test_dockerfile_file_existence():
    assert os.path.isfile('docker/api/Dockerfile')
    assert os.path.isfile('docker/producer/Dockerfile')
    
def test_train_function_existence():
    assert "train" in list(trainer.__dict__.keys())

def test_predict_function_existence():
    assert "predict" in list(predictor.__dict__.keys())
