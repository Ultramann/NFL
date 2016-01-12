import pandas as pd
from factorize import nmf_all_positions


def get_yr_until_wk(year, till_week, nfl_frames):
    '''
    Input:  Int, Int - week to collect data to, NFLFrames
    Output: DataFrame - containing player_ids, opponent and fanduel points for season 
            up until week in question
    '''
    weeks = range(1, till_week + 1)
    year_till_week = pd.concat([nfl_frames.get_year_week_frame(year, week)
                                for week in weeks], 
                               axis=0)
    return year_till_week


def get_preds_to_make(year, wk, nfl_frames):
    '''
    Gets DataFrame with list of players to predict on, their positions, opponents, etc.
    Doesn't include features to be used directly as predictors.
    '''
    historical_data = get_yr_until_wk(year, wk, nfl_frames)
    opponents = nfl_frames.get_year_weeks_opponents(year, wk)
    player_data_columns = ['player_id', 'full_name', 'position', 'team']
    player_data = historical_data[player_data_columns].drop_duplicates()
    player_with_opps = player_data.merge(opponents, how='inner')
    return player_with_opps


def iter_merge(df_gen, on):
    '''
    Input:  Generator - yielding DataFrames with common column names, List - column names
    Output: DataFrame - with columns from on, and all factorizations
    '''
    df = next(df_gen)
    for next_df in df_gen:
        df = df.merge(next_df, on=on)
    return df


def get_modeling_frame(nfl_frames, stats, year=None, week=None, season_type='Regular'):
    '''
    Input:  NFLFrames, Int, Int, Str
    Output: DataFrame 

    For each stat in stats factorize all positions for each week up till given week.
    For both week and year, if none is given, use most recent.
    Stack factorized stats for all players for each week.
    '''
    def gen_factorizations(stats, wk):
        '''
        Input:  List - stats to factorize, Int - week to factorize stats until
        
        Generates DataFrames with factorized stats from input list for each player that week,
        includes their position, team, the week number, their factorized skill, the skill of 
        the defense they played that week for that position, and the fantasy points they 
        actually produced that week.
        '''
        for_nmf_df = get_yr_until_wk(year, wk, nfl_frames) 
        for stat in stats:
            stat_cols = ['{}_factorized_{}'.format(ball_side, stat) for ball_side in ['off', 'def']]
            cols_to_keep = cols + stat_cols
            decomp_off_stat, decomp_def_stat = nmf_all_positions(stat, for_nmf_df)
            week_df = nfl_frames.get_year_week_frame(year, wk)
            stat_df = week_df.merge(decomp_off_stat, on='player_id')
            stat_opp_df = stat_df.merge(decomp_def_stat, how='left', on=['opponent', 'position'])
            yield stat_opp_df[cols_to_keep]
        
    year = year if year else nfl_frames.game.season_year.max()
    base_query = 'season_year == @year and season_type == @season_type and '
    week = week if week else nfl_frames.game.query(base_query + 'finished').week.max()
    cols = ['player_id', 'fanduel_points', 'position', 'team', 'week']
    
    week_decomposed_stats = pd.concat([iter_merge(gen_factorizations(stats, wk), on=cols) 
                                       for wk in range(1, week + 1)],
                                      axis=0)
    return week_decomposed_stats
