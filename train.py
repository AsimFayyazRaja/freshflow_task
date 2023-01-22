import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import utils
import modelling
import pickle


filename=sys.argv[1]

model_name=sys.argv[2]

features,labels=utils.create_dataset(filename)

if model_name=='rnd':
    model=modelling.random_forest_model(features,labels)

    with open('random_forest.pkl', 'wb') as handle:
        pickle.dump(model, handle)

    print("the model is saved as random_forest.pkl")

elif model_name=='grb':
    model=modelling.gradient_boosting_model(features,labels)

    with open('gradient_boosting.pkl', 'wb') as handle:
        pickle.dump(model, handle)

    print("the model is saved as gradient_boosting.pkl")