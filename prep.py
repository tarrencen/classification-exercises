import pandas as pd
import numpy as np
import acquire as acq
from env import get_db_url
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

iris_df = acq.get_iris_data()

def prep_iris(iris_df):
    '''
    Takes in a DataFrame of the iris dataset as acquired and returns a cleaned DF
    Args: iris_df, pandas DataFrame with expected columns and feature names
    Return: cleaned iris_df, pandas DF with cleaning operations performed
    '''
    
    iris_df = acq.get_iris_data()
    iris_df = iris_df.drop_duplicates()
    cols_to_drop = ['species_id', 'measurement_id']
    iris_df = iris_df.drop(columns= (cols_to_drop))
    iris_df = iris_df.rename(columns= {'species_id.1': 'species_id', 'species_name': 'species'})
    dummy_df = pd.get_dummies(iris_df[['species']], dummy_na=False, drop_first=[True])
    iris_df = pd.concat([iris_df, dummy_df], axis=1)
    return iris_df

def prep_titanic(titanic_df):
    '''
    Takes in a pandas DataFrame of the Titanic dataset as acquired and returns a cleaned version of the DF
    Args: titanic_df, pandas DF with expected columns and feature names
    Return: titanic_df_clean, pandas DF with cleaning operations performed
    '''
    titanic_df = acq.get_titanic_data()
    titanic_df = titanic_df.drop_duplicates()
    titanic_df = titanic_df.drop(columns= ['passenger_id', 'age', 'embarked', 'class', 'deck'])
    titanic_df = titanic_df.fillna('Southampton')
    titanic_dummy_df = pd.get_dummies(titanic_df[['sex', 'embark_town']], dummy_na=False, drop_first= [True, True])
    titanic_df = pd.concat([titanic_df, titanic_dummy_df], axis=1)
    titanic_df_clean = titanic_df.drop(columns= ['sex', 'embark_town'])
    return titanic_df_clean

