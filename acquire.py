import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from env import get_db_url
from pydataset import data
import os

def show_codeup_dbs():
    url = get_db_url('employees')
    codeup_dbs = pd.read_sql('SHOW DATABASES', url)
    print('List of Codeup DBs:\n')
    return codeup_dbs

def get_titanic_data():
    filename= 'titanic.csv'

    if os.path.exists(filename):
        print('Reading from CSV file...')
        return pd.read_csv(filename)

    url = get_db_url('titanic_db')
    query = 'SELECT * FROM passengers'
    print('Getting a fresh copy from SQL db...')
    df = pd.read_sql(query, url)
    print('Saving to CSV...')
    df.to_csv(filename, index=False)
    return df

def get_iris_data():
    filename = 'iris.csv'

    if os.path.exists(filename):
        print('Reading from CSV file...')
        return pd.read_csv(filename)

    url = get_db_url('iris_db')
    query = '''
    SELECT * 
    FROM measurements m
    JOIN species s ON m.species_id = s.species_id
    '''
    print('Getting a fresh copy from SQL db...')
    df = pd.read_sql(query, url)
    print('Saving to CSV...')
    df.to_csv(filename, index=False)
    return df

def get_telco_data():
    filename = 'telco.csv'

    if os.path.exists(filename):
        print('Reading from CSV file...')
        return pd.read_csv(filename)

    url = get_db_url('telco_churn')
    # joins to get customers, contract, payment, and internet service
    query = '''
    SELECT *
    FROM customers c
    JOIN contract_types ct ON c.contract_type_id = ct.contract_type_id
    JOIN payment_types pt ON c.payment_type_id = pt.payment_type_id
    JOIN internet_service_types ist ON c.internet_service_type_id = ist.internet_service_type_id'''
    print('Getting a fresh copy from SQL db...')
    df = pd.read_sql(query, url)
    print('Saving to CSV...')
    df.to_csv(filename, index=False)
    return df

def get_titanic_data():
    filename = 'titanic.csv'

    if os.path.exists(filename):
            print('Reading from CSV file...')
            return pd.read_csv(filename)
        
    url = get_db_url('titanic_db')
    query = 'SELECT * FROM passengers'
    print('Getting a fresh copy from SQL db...')
    df = pd.read_sql(query, url)
    print('Saving to CSV...')
    df.to_csv(filename, index=False)
    return df