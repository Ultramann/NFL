import nfldb_tables
from data_prep_tools import get_yr_until_wk

my_frame = nfldb_tables.NFLFrames()
data = get_yr_until_wk(2015, 17, my_frame)
cols_to_keep = ['player_id', 'full_name', 'position', 'team', 'week',
                'fanduel_points', 'opponent', 'fumbles_lost', 'fumbles_tot',
                'passing_att', 'passing_cmp', 'passing_cmp_air_yds',
                'passing_incmp', 'passing_incmp_air_yds', 'passing_int', 'passing_sk',
                'passing_sk_yds', 'passing_tds', 'passing_yds', 'receiving_rec',
                'receiving_tar', 'receiving_tds', 'receiving_yac_yds', 'receiving_yds',
                'rushing_att', 'rushing_loss', 'rushing_loss_yds', 'rushing_tds',
                'rushing_yds', 'home_team', 'away_team']
data[cols_to_keep].to_csv('odsc_football_raw_data.csv', index=False)

cols_with_no_leakage = ['player_id', 'full_name', 'position', 'team', 'week',
                        'fanduel_points', 'opponent', 'home_team', 'away_team']
modeling_data = data.loc[:,cols_with_no_leakage]
data_by_player = data.groupby('player_id')
for col in ['fanduel_points', 'fumbles_lost', 'fumbles_tot',
            'passing_att', 'passing_cmp', 'passing_cmp_air_yds',
            'passing_incmp', 'passing_incmp_air_yds', 'passing_int', 'passing_sk',
            'passing_sk_yds', 'passing_tds', 'passing_yds', 'receiving_rec',
            'receiving_tar', 'receiving_tds', 'receiving_yac_yds', 'receiving_yds',
            'rushing_att', 'rushing_loss', 'rushing_loss_yds', 'rushing_tds',
            'rushing_yds']:
    modeling_data['prev_' + col] = data_by_player[col].shift(1)
    modeling_data['mean_' + col] = modeling_data.groupby('full_name')['prev_' + col].cumsum() / \
                                   modeling_data.groupby('full_name')['prev_' + col].cumcount()
modeling_data.to_csv('odsc_football_modeling_data.csv', index=False)
