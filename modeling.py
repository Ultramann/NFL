import nimfa
import pandas as pd
from nfldb_tables import NFLFrames
from scipy.sparse import csr_matrix as csrm


class PositionNMF(object):
    '''
    Class to hold the model created with NMF for a position, along with all the associated meta-data
    '''
    def __init__(self, position_df, player_ids, opponents, model, week):
        self.df = position_df
        self.player_ids = player_ids
        self.opponents = opponents
        self.model = model
        self.week = week

    def get_position_skill(self):
        '''
        Output: DataFrame - index=player_ids, single column of data corresponding to the latent
                feature from the nmf model relating to player skill
        '''
        position_skill = pd.DataFrame(self.model.basis().toarray(), 
                                      index=self.player_ids, 
                                      columns=['skill'])
        return position_skill

    def get_defense_skill(self):
        '''
        Output: DataFrame - index=opponent (aka defense), single column of data corresponding 
                to the latent feature from the nmf model relating the defense's inability to 
                defend against the position
        '''
        defense_skill = pd.DataFrame(self.model.coef().toarray().T,
                                     index=self.opponents, 
                                     columns=['swiss']) 

        return defense_skill

    def view_sorted_position_skill(self):
        '''
        Output: DataFrame - only has names of players and 'skill' column from NMF to eyeball
                efficacy of model
        '''
        # Prepare to merge back with player names for human readability
        names_columns = ['player_id', 'full_name']
        position_names = self.df[names_columns]

        skill_names =  position_names.merge(right=self.get_position_skill(),
                                            how='inner',
                                            right_index=True,
                                            left_on='player_id') 

        skills = skill_names.groupby(names_columns).first()

        return skills.sort(columns=['skill'], ascending=False)

    def view_sorted_defense_skill(self):
        '''
        Output: DataFrame - only has defense name and 'swiss' columns, names for how like swiss
                cheese the defense is against the position
        '''
        return self.get_defense_skill().sort('swiss')

    def predict(self, opponents):
        '''
        Input:  DataFrame - Opponent pairs
        Output: DataFrame - predicted point production for the given opponents
        '''
        estimate_target = self.model.fitted().toarray()
        estimate_target_df = pd.DataFrame(estimate_target, 
                                          index=self.player_ids, 
                                          columns=self.opponents)
        player_opps = self.df[['player_id', 'team']].drop_duplicates()

        estimate_target_opp_df = player_opps.merge(right=estimate_target_df,
                                                   how='left',
                                                   left_on='player_id',
                                                   right_index=True)

        long_predictions_df = pd.melt(estimate_target_opp_df, 
                                      id_vars=['player_id', 'team'], 
                                      var_name='opponent', 
                                      value_name='prediction')

        week_predictions_df = long_predictions_df.merge(right=opponents,
                                                        how='inner',
                                                        on=['team', 'opponent'])
        return week_predictions_df

class PositionNMFFactory(object):
    '''
    Class for decomposing nfl data with non-negative factorization
    ''' 
    def __init__(self, year, till_week, nfl_frames=None):
        self.year = year
        self.till_week = till_week
        self.nfl_frames = NFLFrames() if nfl_frames is None else nfl_frames
        self.make_year_till_week()


    def make_year_till_week(self):
        '''
        Output: DataFrame - players of given position, their player_id, and the corresponding 
                fanduel points and opponent those points were scored against
        '''
        def get_year_week(week):
            '''
            Helper function to grab the player_id, opponent, and fanduel points for a position 
            in the given year until max_week
            '''
            return self.nfl_frames.get_year_week_frame(self.year, week)

        weeks = range(1, self.till_week + 1)
        self.year_till_week = pd.concat([get_year_week(week) for week in weeks], axis=0)

    def get_position_model(self, position): 
        '''
        Input:  Str - position for which to get the NMF decomposed latent features
        Output: DataFrame - player names and latent skills for position
        '''
        position_sparse, position_df, player_ids, opponents = self.sparsify_position_data(position)
        position_nmf_model = self.decompose_position(position_sparse, player_ids, opponents)

        position_model = PositionNMF(position_df, player_ids, opponents, 
                                     position_nmf_model, self.till_week)

        return position_model

    def sparsify_position_data(self, position):
        '''
        Input:  Str - position for which to get the NMF decomposed latent features
        Output: Sparse matrix (player count in position x opponents) of fanduel points scored,
                list of player_ids for the sparse matrix, list of opponents for the sparse matrix
        '''
        position_df = self.year_till_week.query('position == @position')

        # Get unique player ids and opponent lists 
        player_ids = position_df.player_id.unique().tolist()
        opponents = position_df.opponent.unique().tolist()
        matrix_shape = position_df.player_id.nunique(), position_df.opponent.nunique()

        # Make some categories to associate with the sparse matrix
        player_cats = position_df.player_id.astype('category', categories=player_ids).cat.codes
        opp_cats = position_df.opponent.astype('category', categories=opponents).cat.codes

        # Get rid of the negative point production for now
        fanduel_points = position_df.fanduel_points.clip(lower=0).values

        sparse_position_matrix = csrm((fanduel_points, (player_cats, opp_cats)), shape=matrix_shape)

        return sparse_position_matrix, position_df, player_ids, opponents

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

        return position_nmf


if __name__ == '__main__':
    skill_factory = PositionNMFFactory(2015, 5)
    wr_2015_5 = skill_factory.get_position_model('WR')
    te_2015_5 = skill_factory.get_position_model('TE')
