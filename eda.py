import pandas as pd
import nfldb_tables
import matplotlib.pyplot as plt



def fandual_points_offense(in_df):
    '''
    Input:  DataFrame matching play_player format of nfldb
    Output: Series containing FanDual points for each row in input DataFrame

    based on https://www.fanduel.com/rules
    Currently only calculates points for offensive players
    '''
    fandual_points = \
                        0.1*in_df.rushing_yds + 6*in_df.rushing_tds + \
                        0.04*in_df.passing_yds + 4*in_df.passing_tds - 1*in_df.passing_int + \
                        0.1*in_df.receiving_yds + 6*in_df.receiving_tds + 0.5*in_df.receiving_rec + \
                        6*in_df.kickret_tds + 6*in_df.puntret_tds - 2*in_df.fumbles_lost + \
                        6*in_df.fumbles_rec_tds + 2*in_df.receiving_twoptm + \
                        3*in_df.kicking_fgm + 1*(in_df.kicking_fgm_yds>=40) + \
                        1*(in_df.kicking_fgm_yds>=50) + 1*in_df.kicking_xpmade

    return fandual_points

def graph_offence_stats_summary(player_game_points, verbose=False):
    '''
    Input:  DataFrame of player point per game
    Output: None

    Prints a description of fanduel point production by position for offensive players
    '''
    offensive_positions = ['QB', 'RB', 'WR', 'TE', 'K']
    offensive_query = 'position in {}'.format(offensive_positions)
    offensive_gb = player_game_points.query(offensive_query).groupby('position').offensive_points
    if verbose:
        print('Summary statistics of FanDual points by position')
        print(offensive_gb.describe())

    plt.figure(figsize=(15, 8))
    for i, pos in enumerate(offensive_gb, 1):
        plt.subplot(2, 3, i)
        plt.hist(pos[1].values)
        plt.title(pos[0])
    plt.show()


if __name__ == '__main__':
    games, pp, player = nfldb_tables.get(['game', 'play_player', 'player'])
    pp['offensive_points'] = fandual_points_offense(pp)
    game_points = pp.groupby(['player_id', 'gsis_id'], as_index=False).offensive_points.sum()
    player_game_points = game_points.merge(
                                           right=player['player_id', 'full_name', 'position'],
                                           how='left', 
                                           on='player_id',
                                          )
    graph_offence_stats_summary(player_game_points)
