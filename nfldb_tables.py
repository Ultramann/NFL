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
        self._make_fanduel_points()

    def get(self, tables):
        '''
        Input:  List of table names to get from nfldb

        Makes attributes corresponding to the table names
        '''
        nfldb = nfldb_conn()
        frames = [nfldb.get_table(table) for table in tables]
        self.game, self.play_player, self.player = frames
        nfldb.close()

    def _make_fanduel_points(self):
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

    def get_year_week_frame(self, year, week, season_type='Regular'):
        '''
        Input:  Year as Int, Week as Int, NFL_Frame, Str
        Output: DataFrame of stats for a given week

        Returns DataFrame with all players aggregated stats from a specifc year and week
        '''
        # Make DataFrame games from the specified year, season type and week
        query_string = 'season_type == @season_type & season_year == @year & week == @week'
        week_df = self.game.query(query_string)

        # Get all of the play information for all of those games
        super_week_df = pd.DataFrame(week_df['gsis_id']).merge(self.play_player, 
                                                               how='left', 
                                                               on='gsis_id')
        agg_week_df = super_week_df.groupby(['gsis_id', 'player_id'], as_index=False).sum()

        # Add in the names of the players from the player frame
        player_columns = ['player_id', 'full_name', 'position', 'team']
        agg_week_names_df = agg_week_df.merge(self.player[player_columns], 
                                              how='left', 
                                              on='player_id')

        # Add in the home and away teams
        team_columns = ['gsis_id', 'home_team', 'away_team']
        agg_week_names_df = agg_week_names_df.merge(week_df[team_columns], how='left', on='gsis_id')

        # Compute opponent
        opp = lambda x: x.away_team if x.away_team != x.team else x.home_team
        agg_week_names_df['opponent'] = agg_week_names_df.apply(opp, axis=1)

        return agg_week_names_df.set_index('full_name')

    def get_position_year_week_frame(self, position, year, week, season_type='Regular'):
        '''
        Input:  Str, Year as Int, Week as Int, NFL_Frames, Str
        Output: DataFrame of stats for a players in given position during a given week
        '''
        # Dictionary of stats by type
        offense_types = {'pass': ['passing_att', 'passing_cmp', 'passing_cmp_air_yds', 
                                  'passing_incmp', 'passing_incmp_air_yds', 'passing_int', 
                                  'passing_yds', 'passing_tds'],
                         'rush': ['rushing_att', 'rushing_loss', 'rushing_loss_yds', 
                                  'rushing_yds', 'rushing_tds'],
                         'recp': ['receiving_rec', 'receiving_tar', 'receiving_yac_yds', 
                                  'receiving_yds', 'receiving_tds']}

        # Dictionary of stats by position
        position_dict = {'QB': offense_types['pass'] + offense_types['rush'],
                         'RB': offense_types['rush'] + offense_types['recp'],
                         'WR': offense_types['recp'],
                         'TE': offense_types['recp']}
                             
        df = self.get_year_week_frame(year, week, season_type)

        position_columns = ['position', 'team', 'opponent'] + \
                           position_dict[position] + \
                           ['fanduel_points']

        return df.query('position == @position')[position_columns]

