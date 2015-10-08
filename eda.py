import pandas as pd
import nfldb_tables



def fandual_points_offense(in_df):
    """
    Input: DataFrame matching play_player format of nfldb
    Output: Series containing FanDual points for each row in input DataFrame

    based on https://www.fanduel.com/rules
    Currently only calculates points for offensive players
    """
    fandual_points = \
                        0.1*in_df.rushing_yds + 6*in_df.rushing_tds + \
                        0.04*in_df.passing_yds + 4*in_df.passing_tds - 1*in_df.passing_int + \
                        0.1*in_df.receiving_yds + 6*in_df.receiving_tds + 0.5*in_df.receiving_rec + \
                        6*in_df.kickret_tds + 6*in_df.puntret_tds - 2*in_df.fumbles_lost + \
                        6*in_df.fumbles_rec_tds + 2*in_df.receiving_twoptm + \
                        3*in_df.kicking_fgm + 1*(in_df.kicking_fgm_yds>=40) + \
                        1*(in_df.kicking_fgm_yds>=50) + 1*in_df.kicking_xpmade

    return fandual_points


if __name__ == '__main__':
    games, pp, player = nfldb_tables.get(['game', 'play_player', 'player'])
    pp['offensive_points'] = fandual_points_offense(pp)
    game_points = pp.groupby(['player_id', 'gsis_id']).offensive_points.sum()
    player_game_points = pd.DataFrame(game_points).reset_index().merge(	
    									right=player.ix[:,["player_id", "full_name", "position"]], 
    									how='left',
    									on="player_id", 
    									)
