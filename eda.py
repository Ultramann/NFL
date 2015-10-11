import pandas as pd
import nfldb_tables
import matplotlib.pyplot as plt


def check_player_table(player):
    '''
    Input:  Player DataFrame
    Output: None

    Prints whether or not all of the players without a known position are also of unknown status
    '''
    unknowns = player.query('position == "UNK"').status == 'Unknown'
    truth = all(unknowns.values)
    print 'All of the position UNKs have Unknown status: {}'.format(truth)


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


def make_player_game_points(g, pp_df, p_df, year=None):
    '''
    Input:  DataFrame of games, DataFame of play_player, DataFame of players, Int of year
    Output: DataFrame with aggregated fanduel points for players, 
            optionally, who played in a given year
    '''
    year_query = 'season_year == @year' if year else 'season_year > 0' 
    pp = pp_df.merge(right=g[['gsis_id', 'season_year']], how='left', on='gsis_id').query(year_query)
    game_points = pp.groupby(['player_id', 'gsis_id'], as_index=False).offensive_points.sum()
    player_game_points = game_points.merge(
                                           right=p_df[['player_id', 'full_name', 'position']],
                                           how='left', 
                                           on='player_id',
                                          )
    return player_game_points


def graph_offence_stats_summary(player_game_points, verbose=False, bins=50):
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
        plt.hist(pos[1].values, bins)
        plt.title(pos[0])
    plt.show()


def get_year_week_frame(year, week, game, play_player, player, season_type='Regular'):
    '''
    Input:  DataFrame of games, DataFrame for play_player, DataFrame of players, Int, Int, Str
    Output: DataFrame of stats for a given week

    Returns DataFrame with all players aggregated stats from a specifc year and week
    '''
    # Make DataFrame games from the specified year, season type and week
    week_df = game.query('season_type == @season_type & season_year == @year & week == @week')

    # Get all of the play information for all of those games
    super_week_df = pd.DataFrame(week_df['gsis_id']).merge(play_player, how='left', on='gsis_id')
    agg_week_df = super_week_df.groupby(['gsis_id', 'player_id'], as_index=False).sum()

    # Add in the names of the players from the player frame
    player_columns = ['player_id', 'full_name', 'position', 'team']
    agg_week_names_df = agg_week_df.merge(player[player_columns], how='left', on='player_id')

    # Add in the home and away teams
    team_columns = ['gsis_id', 'home_team', 'away_team']
    agg_week_names_df = agg_week_names_df.merge(week_df[team_columns], how='left', on='gsis_id')

    # Compute opponent
    opp = lambda x: x.away_team if x.away_team != x.team else x.home_team
    agg_week_names_df['opponent'] = agg_week_names_df.apply(opp, axis=1)

    return agg_week_names_df.set_index('full_name')


def get_position_year_week_frame(position, year, week, 
                                 game, play_player, player, season_type='Regular'):
    '''
    Input: Str, DataFrame of games, DataFrame for play_player, DataFrame of players, Int, Int, Str
    Output: DataFrame of stats for a players in given position during a given week
    '''
    # Dictionary of stats by type
    offence_types = {'pass': ['passing_att', 'passing_cmp', 'passing_cmp_air_yds', 'passing_incmp',
                              'passing_incmp_air_yds', 'passing_int', 'passing_yds', 'passing_tds'],
                     'rush': ['rushing_att', 'rushing_loss', 'rushing_loss_yds', 'rushing_yds',
                              'rushing_tds'],
                     'recp': ['receiving_rec', 'receiving_tar', 'receiving_yac_yds', 'receiving_yds',
                              'receiving_tds']}

    # Dictionary of stats by position
    position_dict = {'QB': offence_types['pass'] + offence_types['rush'],
                     'RB': offence_types['rush'] + offence_types['recp'],
                     'WR': offence_types['recp'],
                     'TE': offence_types['recp']}
                         
    df = get_year_week_frame(year, week, game, play_player, player, season_type)

    position_columns = ['team', 'position', 'opponent'] + \
                       position_dict[position] + \
                       ['offensive_points']

    return df.query('position == @position')[position_columns]


if __name__ == '__main__':
    game, pp, player = nfldb_tables.get(['game', 'play_player', 'player'])
    check_player_table(player)
    pp['offensive_points'] = fandual_points_offense(pp)
    player_game_points = make_player_game_points(game, pp, player)
    player_game_points_2014 = make_player_game_points(game, pp, player, year=2014)
    graph_offence_stats_summary(player_game_points)

