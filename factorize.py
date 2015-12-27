import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix as csrm
from positions import positions_list


def nmf_all_positions(stat, df_with_stats):
    '''
    Input:  Str - player statistic to be factorized 
            DataFrame - containing stat column, likely output from
                        data_prep_tools.get_yr_until_wk
    Output: DataFrames - factorized stat for offensive player / defense
    '''
    decomposition_results = [nmf_one_position(stat, df_with_stats, position)
                             for position in positions_list]
    offense = pd.concat([results['offense'] for results in decomposition_results])
    defense = pd.concat([results['defense'] for results in decomposition_results])
    return offense, defense


def nmf_one_position(stat, df_with_stats, position):
    '''
    Input:  DataFrame - containing fanduel_points column, likely output from
                        data_prep_tools.get_yr_until_wk
            Str - capitalized position abbreviation
    Output: Dict - (keys: offense/defense, values: nmf decomposed skill)
    '''
    position_df = df_with_stats[df_with_stats.position == position]
    sparse_position_matrix, player_ids, opponents = sparsify(stat, position_df)
    offense_skill, defense_skill = decompose(sparse_position_matrix)
    offense_skill_df = pd.DataFrame({'player_id': player_ids,
                                     'off_factorized_' + stat: offense_skill})
    defense_skill_df = pd.DataFrame({'opponent': opponents,
                                     'position': position,
                                     'def_factorized_' + stat: defense_skill})

    return {'offense': offense_skill_df, 'defense': defense_skill_df}


def sparsify(stat, input_df, clip_positive=True):
    '''
    Input:  DataFrame - with player_ids, opponent and statistics to be sparsified
            Bool - specifies whether to clip all values at 0 (typically for nmf)
    Output: Sparse matrix (player count in position x opponents) of fanduel points scored,
            list of player_ids for the sparse matrix, list of opponents for the sparse matrix
    '''
    stat_data = input_df[stat].clip(0) if clip_positive else input_df[stat]
    player_ids = input_df.player_id.unique()
    opponents = input_df.opponent.unique()

    # Make some categories to associate with the sparse matrix
    player_cats = input_df.player_id.astype('category', categories=player_ids).cat.codes
    opp_cats = input_df.opponent.astype('category', categories=opponents).cat.codes

    matrix_shape = input_df.player_id.nunique(), input_df.opponent.nunique()
    sparse_position_matrix = csrm((stat_data, (player_cats, opp_cats)), 
                                  shape=matrix_shape)

    return sparse_position_matrix, player_ids, opponents


def decompose(sparse_mat_to_decompose):
    '''
    Input:  Sparse Matrix - statistics (rows: player_ids, columns: opponents)
    Output: Array with factorized skill value for each offensive player. Another Array
            with factorized skill for each team.  Orderings consistent with input matrix.
    '''
    nmf = NMF(n_components=1, max_iter=10000)
    offense = nmf.fit_transform(sparse_mat_to_decompose).flatten()
    defense = nmf.components_.flatten()

    return offense, defense


def preds_from_factorized_skills(input_df, stat='fanduel_points'):
    '''
    Input:  DataFrame with off_factorized_stat and def_factorized_stat for 
            each game we want predictions for
    Output: DataFrame with predicted pointed points for each player in input_df
    '''
    out = pd.DataFrame({'player_id': input_df.player_id,
                        'name': input_df.full_name,
                        'position': input_df.position,
                        'pred': input_df['off_factorized_{}'.format(stat)] * \
                                input_df['def_factorized_{}'.format(stat)]})
    return out


def merge_factorizations_to_main_df(main_df, offense, defense):
    '''
    Input:  DataFrame, DataFrame, DataFrame
    Output: DataFrame

    Inner joins, starting with main_df, offense and defense skill
    '''
    main_with_o_skill = main_df.merge(offense, how='inner')
    out = main_with_o_skill.merge(defense, how='inner')
    return out
