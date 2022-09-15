import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.handler import train, test
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

args = parser.parse_args()


datas = pd.read_csv("predict_ape_clone.csv")
print(datas)

df2 = datas.set_axis(['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'RUB', 'CHF', 'CNY', 'INR'], axis=1, inplace=False)

print("DF2 İS :",df2)

for column in df2:
    print(df2[column])

df_mean = df2[['EUR', 'CAD', 'JPY', 'GBP', 'AUD', 'RUB', 'CHF', 'CNY', 'INR']].mean()*100
#print("df an is : ")
#print(df_mean)

df_mean.to_csv('file_name.csv')







#datas_array = datas.to_numpy()

#print(datas_array)

