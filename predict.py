import pickle
import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('-d','--day', help='day', required=True, type=float)
parser.add_argument('-it','--item', help='item', required=True, type=float)
parser.add_argument('-pp','--purchase_price', help='purchase_price', required=True, type=float)
parser.add_argument('-sgp','--suggested_retail_price', help='suggested_retail_price', required=True, type=float)
parser.add_argument('-o','--orders_quantity', help='orders_quantity', required=True, type=float)
parser.add_argument('-r','--revenue', help='revenue', required=True, type=float)
parser.add_argument('-m','--month', help='month', required=True, type=float)
parser.add_argument('-w','--weekday', help='weekday', required=True, type=float)
parser.add_argument('-y','--year', help='year', required=True, type=float)
parser.add_argument('-s','--sales_quantity', help='sales_quantity', required=True, type=float)

args = vars(parser.parse_args())

with open('random_forest.pkl', 'rb') as handle:
    clf = pickle.load(handle)

feats=list(args.values())

feats=np.reshape(feats,(1,-1))

print("Predicted sales quantity: ", clf.predict(feats)[0])