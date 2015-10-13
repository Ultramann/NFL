import pandas as pd
import matplotlib.pyplot as plt
from nfldb_tables import NFL_Frames


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


def get_year_week_frame(year, week, nfl_frames, season_type='Regular'):
    '''
    Input:  Year as Int, Week as Int, NFL_Frame, Str
    Output: DataFrame of stats for a given week

    Returns DataFrame with all players aggregated stats from a specifc year and week
    '''
    # Make DataFrame games from the specified year, season type and week
    query_string = 'season_type == @season_type & season_year == @year & week == @week'
    week_df = nfl_frames.game.query(query_string)

    # Get all of the play information for all of those games
    super_week_df = pd.DataFrame(week_df['gsis_id']).merge(nfl_frames.play_player, 
                                                           how='left', 
                                                           on='gsis_id')
    agg_week_df = super_week_df.groupby(['gsis_id', 'player_id'], as_index=False).sum()

    # Add in the names of the players from the player frame
    player_columns = ['player_id', 'full_name', 'position', 'team']
    agg_week_names_df = agg_week_df.merge(nfl_frames.player[player_columns], 
                                          how='left', 
                                          on='player_id')

    # Add in the home and away teams
    team_columns = ['gsis_id', 'home_team', 'away_team']
    agg_week_names_df = agg_week_names_df.merge(week_df[team_columns], how='left', on='gsis_id')

    # Compute opponent
    opp = lambda x: x.away_team if x.away_team != x.team else x.home_team
    agg_week_names_df['opponent'] = agg_week_names_df.apply(opp, axis=1)

    return agg_week_names_df.set_index('full_name')


def get_position_year_week_frame(position, year, week, nfl_frames, season_type='Regular'):
    '''
    Input:  Str, Year as Int, Week as Int, NFL_Frames, Str
    Output: DataFrame of stats for a players in given position during a given week
    '''
    # Dictionary of stats by type
    offense_types = {'pass': ['passing_att', 'passing_cmp', 'passing_cmp_air_yds', 'passing_incmp',
                              'passing_incmp_air_yds', 'passing_int', 'passing_yds', 'passing_tds'],
                     'rush': ['rushing_att', 'rushing_loss', 'rushing_loss_yds', 'rushing_yds',
                              'rushing_tds'],
                     'recp': ['receiving_rec', 'receiving_tar', 'receiving_yac_yds', 'receiving_yds',
                              'receiving_tds']}

    # Dictionary of stats by position
    position_dict = {'QB': offense_types['pass'] + offense_types['rush'],
                     'RB': offense_types['rush'] + offense_types['recp'],
                     'WR': offense_types['recp'],
                     'TE': offense_types['recp']}
                         
    df = get_year_week_frame(year, week, nfl_frames, season_type)

    position_columns = ['position', 'team', 'opponent'] + \
                       position_dict[position] + \
                       ['fanduel_points']

    return df.query('position == @position')[position_columns]


if __name__ == '__main__':
    nfl_frames = NFL_Frames() 
    check_player_table(nfl_frames.player)
    player_game_points = make_player_game_points(nfl_frames)
    player_game_points_2014 = make_player_game_points(nfl_frames, year=2014)
    graph_offense_stats_summary(player_game_points)

