import nimfa
import pandas as pd
from nfldb_tables import NFL_Frames
from scipy.sparse import csr_matrix as csrm

def get_fanduel_points_year_till_week(nfl_frames, year, till_week):
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
        columns = ['player_id', 'position', 'opponent', 'fanduel_points']
        return nfl_frames.get_year_week_frame(year, week)[columns]

    weeks = range(1, till_week + 1)
    position_opp_fdp_year = pd.concat([get_fanduel_opp(week) for week in weeks], axis=0)
    return position_opp_fdp_year


def make_sparse_position_pivot_table(position, year_till_week_df):
    position_subset_df = year_till_week_df.query('position == @position')

    # Get unique player ids and opponent lists 
    player_ids = position_subset_df.player_id.unique().tolist()
    opponents = position_subset_df.opponent.unique().tolist()
    matrix_shape = position_subset_df.player_id.nunique(), position_subset_df.opponent.nunique()

    # Make some categories
    player_cats = position_subset_df.player_id.astype('category', categories=player_ids).cat.codes
    opp_cats = position_subset_df.opponent.astype('category', categories=opponents).cat.codes

    # Get rid of the negative point production for now
    fanduel_points = position_subset_df.fanduel_points.clip(lower=0).values

    sparse_position_matrix = csrm((fanduel_points, (player_cats, opp_cats)), shape=matrix_shape)

    return sparse_position_matrix, player_ids, opponents


def get_position_till_week(position, till_week, year_till_week_df):
    pass


def decompose_things(position_sparse, player_ids, opponents):
    '''
    Input:  Sparse Matrix of fanduel points (rows-player_ids, columns-opponents), 
            list of player_ids, list of opponents
    Output: 
    '''
    nmf = nimfa.Snmf(wr_2015_sparse, max_iter=100000, rank=1, update='euclidean', objective='fro')
    wr_2015_nmf = nmf()
    wr_skill = pd.DataFrame(wr_2015_nmf.basis().toarray(), index=player_ids, columns='skill')
    return wr_skill


if __name__ == '__main__':
    nfl_frames = NFL_Frames() 
    twenty_fifteen_5 = get_fanduel_points_year_till_week(nfl_frames, 2015, 5) 
    wr_2015_sparse, player_ids, opponents = make_sparse_position_pivot_table('WR', twenty_fifteen_5) 
    wr_2015_skill = nfl_frames.player.query('position == "WR"'). \
                    merge(right=decompose_things(wr_2015_sparse, player_ids, opponents), 
                          how='inner',
                          right_index=True,
                          left_on='player_id') 
