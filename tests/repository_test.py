import pytest

import os

from app.core.machine_learning import trainer
from app.core.machine_learning import predictor


def test_train_file_existence():
     assert os.path.isfile('app/core/machine_learning/trainer.py')

def test_scoring_file_existence():
     assert os.path.isfile('app/core/machine_learning/predictor.py')

def test_gitignore_file_existence():
    assert os.path.isfile('.gitignore')

def test_requirements_file_existence():
    assert os.path.isfile('requirements.txt')

def test_dockerfile_file_existence():
    assert os.path.isfile('Dockerfile')

def test_train_function_existence():
    assert "train" in list(trainer.__dict__.keys())

def test_predict_function_existence():
    assert "predict" in list(predictor.__dict__.keys())
