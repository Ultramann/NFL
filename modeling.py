import pandas as pd
from nfldb_tables import NFL_Frames

def get_position_year_tillweek_fanduel_points(nfl_frames, position, year, max_week):
    '''
    Input:  NFL_Frames, position as Str, Year as Int, Week as Int
    Output: DataFrame of players of given position, their player_id, and the corresponding fanduel
            points - opponent those points were scored against pairs
    '''
    def get_fanduel_opp(week):
        '''
        Helper function to grab the player_id, opponent, and fanduel points for a position in the
        given year until max_week
        '''
        columns = ['player_id', 'opponent', 'fanduel_points']
        return nfl_frames.get_position_year_week_frame(position, year, week)[columns]

    weeks = range(1, max_week + 1)
    position_opp_fdp_year = pd.concat([get_fanduel_opp(week) for week in weeks], axis=0)
    return position_opp_fdp_year

if __name__ == '__main__':
    nfl_frames = NFL_Frames() 
    wr_2015_5 = get_position_year_tillweek_fanduel_points(nfl_frames, 'WR', 2015, 5) 
    
