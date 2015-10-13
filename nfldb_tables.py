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


class NFL_Frames(object):
    '''
    Class to hold the most frequently used nfldb tables (game, play_player, player) 
    as frames all bundled together
    '''
    def __init__(self, tables=['game', 'play_player', 'player']):
        self.get(tables) 
        self.make_fanduel_points()

    def get(self, tables):
        '''
        Input:  List of table names to get from nfldb

        Makes attributes corresponding to the table names
        '''
        nfldb = nfldb_conn()
        frames = [nfldb.get_table(table) for table in tables]
        self.game, self.play_player, self.player = frames
        nfldb.close()

    def make_fanduel_points(self):
        '''
        Calculates the fanduel point production based on https://www.fanduel.com/rules
        Currently only calculates points for offensive players
        '''
        fanduel_points = ( 0.1 * self.play_player.rushing_yds 
                         + 6 * self.play_player.rushing_tds 
                         + 0.04 * self.play_player.passing_yds 
                         + 4*self.play_player.passing_tds 
                         - 1*self.play_player.passing_int 
                         + 0.1*self.play_player.receiving_yds 
                         + 6*self.play_player.receiving_tds 
                         + 0.5*self.play_player.receiving_rec 
                         + 6*self.play_player.kickret_tds 
                         + 6*self.play_player.puntret_tds 
                         - 2*self.play_player.fumbles_lost 
                         + 6*self.play_player.fumbles_rec_tds 
                         + 2*self.play_player.receiving_twoptm 
                         + 3*self.play_player.kicking_fgm 
                         + 1*(self.play_player.kicking_fgm_yds>=40) 
                         + 1*(self.play_player.kicking_fgm_yds>=50) 
                         + 1*self.play_player.kicking_xpmade)

        self.play_player['fanduel_points'] = fanduel_points

