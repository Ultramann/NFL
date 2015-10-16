import nimfa
import pandas as pd
from nfldb_tables import NFL_Frames
from scipy.sparse import csr_matrix as csrm


class NMF_Skillz(object):
    def __init__(self, year, till_week):
        self.year = year
        self.till_week = till_week
        self.nfl_frames = NFL_Frames()
        self.make_year_till_week()

    def make_year_till_week(self):
        '''
        Output: DataFrame of players of given position, their player_id, and the corresponding 
                fanduel points and opponent those points were scored against
        '''
        def get_fanduel_opp(week):
            '''
            Helper function to grab the player_id, opponent, and fanduel points for a position 
            in the given year until max_week
            '''
            columns = ['player_id', 'position', 'opponent', 'fanduel_points']
            return self.nfl_frames.get_year_week_frame(self.year, week)[columns]

        weeks = range(1, self.till_week + 1)
        self.year_till_week = pd.concat([get_fanduel_opp(week) for week in weeks], axis=0)

    def get_position(self, position): 
        '''
        Input:  Str - position to get latent skills for
        Output: DataFrame - player names and latent skills for position
        '''
        position_sparse, player_ids, opponents = self.make_sparse_position_pivot_table(position)
        position_skill = self.decompose_position(position_sparse, player_ids, opponents)

        # Prepare to merge back with player names for human readability
        names_columns = ['player_id', 'full_name']
        position_names = self.nfl_frames.player.query('position == @position')[names_columns]

        skill_names =  position_names.merge(right=position_skill,
                                            how='inner',
                                            right_index=True,
                                            left_on='player_id') 
        return skill_names.sort(columns=['skill'], ascending=False)

    def make_sparse_position_pivot_table(self, position):
        '''
        Input:  Position for which to get the NMF decomposition latent features
        Output: Sparse matrix (player count in position x opponents) of fanduel points scored,
                list of player_ids for the sparse matrix, list of opponents for the sparse matrix
        '''
        position_df = self.year_till_week.query('position == @position')

        # Get unique player ids and opponent lists 
        player_ids = position_df.player_id.unique().tolist()
        opponents = position_df.opponent.unique().tolist()
        matrix_shape = position_df.player_id.nunique(), position_df.opponent.nunique()

        # Make some categories
        player_cats = position_df.player_id.astype('category', categories=player_ids).cat.codes
        opp_cats = position_df.opponent.astype('category', categories=opponents).cat.codes

        # Get rid of the negative point production for now
        fanduel_points = position_df.fanduel_points.clip(lower=0).values

        sparse_position_matrix = csrm((fanduel_points, (player_cats, opp_cats)), shape=matrix_shape)

        return sparse_position_matrix, player_ids, opponents

    def decompose_position(self, position_sparse, player_ids, opponents):
        '''
        Input:  Sparse Matrix - fanduel points (rows-player_ids, columns-opponents), 
                list of player_ids positionally associated with each row 
        Output: DataFrame - index of player_ids, skill from single latent feature NMF decomposition 
        '''
        nmf = nimfa.Snmf(position_sparse, 
                         max_iter=10000, 
                         rank=1, 
                         update='euclidean', 
                         objective='fro')
        position_nmf = nmf()
        position_skill = pd.DataFrame(position_nmf.basis().toarray(), 
                                      index=player_ids, columns=['skill'])
        return position_skill

if __name__ == '__main__':
    skillz = NMF_Skillz(2015, 5)
    wr_2015_5 = skillz.get_position('WR')
    te_2015_5 = skillz.get_position('TE')
