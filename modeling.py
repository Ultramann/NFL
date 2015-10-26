import nimfa
import pandas as pd
from nfldb_tables import NFLFrames
from scipy.sparse import csr_matrix as csrm

positions = ['QB', 'TE', 'RB', 'K', 'WR']

def get_yr_until_wk(year, till_week):
    '''
    Input: Integers denoting the year/season and what week to collect data to.
    Output:  DataFrame containing player_ids, opponent and fanduel points for season 
    up until week in question
    '''
    nfl_frames = NFLFrames()
    weeks = range(1, till_week+1)
    year_till_week = pd.concat([nfl_frames.get_year_week_frame(year, week) \
                         for week in weeks], axis=0)
    #dropping columns because many don't make sense at this level of aggregation
    cols_to_keep = ['player_id', 'fanduel_points', 'full_name', 'position', \
                    'team', 'opponent']
    return year_till_week.ix[:, cols_to_keep]

def nmf_all_positions(df_with_points):
    decomposition_results = [nmf_one_position(df_with_points, position) for position in positions]
    offense = pd.concat([results['offense'] for results in decomposition_results])
    defense = pd.concat([results['defense'] for results in decomposition_results])
    return offense, defense

def nmf_one_position(df_with_points, position):
    position_df = df_with_points.query('position == @position')
    sparse_position_matrix, player_ids, opponents = sparsify(position_df)
    offense_skill, defense_skill = decompose(sparse_position_matrix)
    offense_skill_df = pd.DataFrame({   'player_id': player_ids,
                                        'off_factorized_skill': offense_skill
                                    })
    defense_skill_df = pd.DataFrame({   'opponent': opponents,
                                        'position': position,
                                        'def_factorized_skill': defense_skill
                                    })
    return {'offense': offense_skill_df, 
            'defense': defense_skill_df}

def sparsify(input_df, clip_positive=True):
    '''
    Input:  DataFrame containing player_ids, opponent and fanduel points to be sparsified
            clip_positive specifies whether to clip all values at 0 (typically for nmf)
    Output: Sparse matrix (player count in position x opponents) of fanduel points scored,
            list of player_ids for the sparse matrix, list of opponents for the sparse matrix
    '''
    if clip_positive:
        input_df.fanduel_points = input_df.fanduel_points.clip(0)
    player_ids = input_df.player_id.unique()
    opponents = input_df.opponent.unique()

    # Make some categories to associate with the sparse matrix
    player_cats = input_df.player_id.astype('category', categories=player_ids).cat.codes
    opp_cats = input_df.opponent.astype('category', categories=opponents).cat.codes
    matrix_shape = input_df.player_id.nunique(), input_df.opponent.nunique()
    sparse_position_matrix = csrm((input_df.fanduel_points, (player_cats, opp_cats)), 
                                shape=matrix_shape)
    return sparse_position_matrix, player_ids, opponents

def decompose(sparse_mat_to_decompose):
    '''
    Input:  Sparse Matrix - fanduel points (rows-player_ids, columns-opponents)
    Output: Array with factorized skill value for each offensive player. Another Array
            with factorized skill for each team.  Orderings consistent with input matrix.
    '''

    nmf = nimfa.Snmf(sparse_mat_to_decompose, 
                     max_iter=10000, 
                     rank=1,
                     update='euclidean', 
                     objective='fro')
    position_nmf = nmf()
    offense, defense = position_nmf.basis(), position_nmf.coef()
    offense, defense = [my_mat.toarray().flatten() for my_mat in (offense, defense)]
    return offense, defense

def pred_from_factorized_skills(input_df):
    '''
    Input: DataFrame with off_factorized_skill and def_factorized_skill for each game we want
            predictions for.
    Output: Series containing predicted points for each game/row in the input data set 
    '''
    return input_df.off_factorized_skill * input_df.def_factorized_skill

def get_basic_player_data_for_predicting(year, wk):
    '''
    Gets DataFrame with list of players to predict on, their positions, opponents, etc.
    Not meant to have features to be used as predictors
    '''
    historical_data = get_yr_until_wk(year, wk)
    opponents = nfl_frames.get_year_weeks_opponents(year=year, week=wk)
    player_data = historical_data.ix[:,['player_id', 'full_name', 'position', 'team']].drop_duplicates()
    player_with_skill = player_data.merge(offense, how='inner')
    player_with_opps = player_with_skill.merge(opponents, how='left')

if __name__ == '__main__':
    nfl_frames = NFLFrames()
    historical_data = get_yr_until_wk(2015, 3)
    offense, defense = nmf_all_positions(historical_data)
