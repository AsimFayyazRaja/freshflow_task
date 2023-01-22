import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def label_encoding(df,cols_to_encode):
    le={}
    for col in cols_to_encode:
        le[col]=LabelEncoder()
        df[col]=le[col].fit_transform(df[col])
    return df


def clean_df(df):
    "cleans the df given"

    df['purchase_price']=pd.to_numeric(df['purchase_price'])
    df['suggested_retail_price']=pd.to_numeric(df['suggested_retail_price'])
    df['orders_quantity']=pd.to_numeric(df['orders_quantity'])
    df['sales_quantity']=pd.to_numeric(df['sales_quantity'])
    df['revenue']=pd.to_numeric(df['revenue'])
    df['date']=pd.to_datetime(df['day'])

    df['month']=df['date'].dt.month
    df['day']=df['date'].dt.day
    df['weekday']=df['date'].dt.day_name()
    df['year']=df['date'].dt.year
    df=df.sort_values(by='date')
    df.replace([np.inf, -np.inf], 0, inplace=True)
    drop_columns=['Unnamed: 0', 'item_number', 'date']
    df=df.drop(drop_columns,axis=1)
    df.fillna(0, inplace=True)
    return df


def make_feature_labels(df):
    # make features and labels
    features=[]
    labels=[]
    unique_items=df.item_name.unique() # unique items
    with tqdm(total=len(unique_items)) as pbar:
        for item in unique_items:
            df2=df.loc[df['item_name']==item]
            df2=df2.drop_duplicates()
            feature_flag=True
            for i,row in df2.iterrows():
                if feature_flag:
                    sublist=[row['day'],row['item_name'],row['purchase_price'],row['suggested_retail_price'],
                        row['orders_quantity'],row['revenue'],row['month'],row['weekday'],row['year'],row['sales_quantity']]
                    feature_flag=False
                    continue
                else:
                    features.append(sublist)
                    labels.append(row['sales_quantity'])
                    sublist=[]
                    feature_flag=True
            pbar.update(1)
    return np.array(features), np.array(labels)


def create_dataset(fname):
    "takes the filename and returns the features and labels"
    df=pd.read_csv(fname) # load csv to dataframe
    
    df=clean_df(df)
    cols_to_encode=['item_name', 'weekday'] # encode the non-numeric columns 
    df=label_encoding(df,cols_to_encode)

    feats,labs=make_feature_labels(df)
    return feats,labs
    
