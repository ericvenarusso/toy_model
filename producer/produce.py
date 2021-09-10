import random
import time

import requests


def random_wait():
    time.sleep(random.randint(5, 10))

def generate_random_age():
    return random.randint(1, 90)

def generate_random_sex():
    return random.choice(["male", "female"])

def generate_random_pclass():
    return random.randint(1, 3)

def generate_random_data():
    random_data = {
        "age": generate_random_age(),
        "sex": generate_random_sex(),
        "pclass": generate_random_pclass()
    }
    return random_data

def produce():
    while True:
        random_wait()
        payload = generate_random_data()
        requests.post("http://host.docker.internal:8000/predict", json=payload)


produce()
