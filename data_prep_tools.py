from nfldb_tables import NFLFrames
import pandas as pd

positions_list = ['QB', 'TE', 'RB', 'K', 'WR']

def get_yr_until_wk(year, till_week, nfl_frames):
    '''
    Input: Integers denoting the year/season and what week to collect data to.
            Optionally NFLFrames instance (to save cost of building new one)
    Output:  DataFrame containing player_ids, opponent and fanduel points for season 
    up until week in question
    '''
    weeks = range(1, till_week+1)
    year_till_week = pd.concat([nfl_frames.get_year_week_frame(year, week) \
                         for week in weeks], axis=0)
    #dropping columns because many don't make sense at this level of aggregation
    cols_to_keep = ['player_id', 'fanduel_points', 'full_name', 'position', \
                    'team', 'opponent']
    return year_till_week.ix[:, cols_to_keep]

def get_preds_to_make(year, wk, nfl_frames):
    '''
    Gets DataFrame with list of players to predict on, their positions, opponents, etc.
    Doesn't include features to be used directly as predictors
    '''
    historical_data = get_yr_until_wk(year, wk, nfl_frames)
    opponents = nfl_frames.get_year_weeks_opponents(year, wk)
    player_data = historical_data.ix[:,['player_id', 'full_name', 'position', 'team']].drop_duplicates()
    player_with_opps = player_data.merge(opponents, how='inner')
    return player_with_opps

