import pandas as pd
import matplotlib.pyplot as plt
from nfldb_tables import NFLFrames


def check_player_table(player):
    '''
    Input:  Player DataFrame
    Output: None

    Prints whether or not all of the players without a known position are also of unknown status
    '''
    unknowns = player.query('position == "UNK"').status == 'Unknown'
    truth = all(unknowns.values)
    print 'All of the position UNKs have Unknown status: {}'.format(truth)


def make_player_game_points(nfl_frames, year=None):
    '''
    Input:  NFL_Frames, Int of year
    Output: DataFrame with aggregated fanduel points for players, 
            optionally, who played in a given year
    '''
    year_query = 'season_year == @year' if year else 'season_year > 0' 
    ppg = nfl_frames.play_player.merge(right=nfl_frames.game[['gsis_id', 'season_year']], 
                                       how='left', 
                                       on='gsis_id')
    ppgy = ppg.query(year_query)
    game_points = ppgy.groupby(['player_id', 'gsis_id'], as_index=False).fanduel_points.sum()
    player_columns = ['player_id', 'full_name', 'position']
    player_game_points = game_points.merge(right=nfl_frames.player[player_columns],
                                           how='left', 
                                           on='player_id')
                                        
    return player_game_points


def graph_offense_stats_summary(player_game_points, verbose=False, bins=50):
    '''
    Input:  DataFrame of player point per game
    Output: None

    Graphs fanduel point production by position for offensive players, optionally prints description 
    '''
    # Only want to grab the offensive point producing positions
    offensive_positions = ['QB', 'RB', 'WR', 'TE', 'K']
    offensive_query = 'position in {}'.format(offensive_positions)
    offensive_gb = player_game_points.query(offensive_query).groupby('position').fanduel_points

    if verbose:
        print('Summary statistics of FanDual points by position')
        print(offensive_gb.describe())

    # Cycle through the group by object and graph the data for each position
    plt.figure(figsize=(15, 8))
    for i, pos in enumerate(offensive_gb, 1):
        plt.subplot(2, 3, i)
        plt.hist(pos[1].values, bins)
        plt.title(pos[0])
    plt.show()

if __name__ == '__main__':
    nfl_frames = NFLFrames() 
    check_player_table(nfl_frames.player)
    player_game_points = make_player_game_points(nfl_frames)
    player_game_points_2014 = make_player_game_points(nfl_frames, year=2014)
    graph_offense_stats_summary(player_game_points)

