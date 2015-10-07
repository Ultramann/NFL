import psycopg2
import pandas as pd

class nfldb_conn(object):
    '''
    Class to easily query the nfl database for entire tables
    '''
    def __init__(self):
        self.conn = psycopg2.connect(dbname='nfldb')
    
    def close(self):
        self.conn.close()

    def get_table(self, table_name):
        '''
        Input:  Str with name of table from nfldb
        Output: Dataframe of entire table
        '''
        return pd.read_sql('SELECT * FROM {}'.format(table_name), self.conn)

def get(tables):
    '''
    Input:  List of table names to get from nfldb
    Output  List of Dataframes for corresponding table names
    '''
    nfldb = nfldb_conn()
    frames = [nfldb.get_table(table) for table in tables]
    nfldb.close()
    return frames
