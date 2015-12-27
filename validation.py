from nfldb_tables import NFLFrames
from sklearn.metrics import mean_squared_error
from factorize import nmf_all_positions, merge_factorizations_to_main_df, \
                      preds_from_factorized_skills
from data_prep_tools import get_yr_until_wk, get_preds_to_make


def year_week_rmse(week_actuals_df, preds, stat='fanduel_points'):
    '''
    Input:  DataFrame with actual fanduel_points, DataFrame with pred column.
            Both must have player_id
    Output: RMSE between predicted and actual values from year-week
    '''
    actuals_with_preds_df = week_actuals_df.merge(right=preds,
                                                  how='inner', 
                                                  on='player_id')
    rmse = mean_squared_error(actuals_with_preds_df[stat],
                              actuals_with_preds_df.pred) ** 0.5
    return rmse


def check_nmf_model(nfl_frames, year, week, position='All'):
    '''
    Input:  NFLFrames, Int, Int, Str
    Output: None

    Prints comparison of standard deviation in fanduel points versus RMSE in predictions.
    '''
    fp_stat = 'fanduel_points'
    historical_data = get_yr_until_wk(year, week, nfl_frames)
    offense, defense = nmf_all_positions(fp_stat, historical_data)
    all_week_actuals_df = nfl_frames.get_year_week_frame(year, week)
    preds_to_make = get_preds_to_make(year, week + 1, nfl_frames)
    for_preds_df = merge_factorizations_to_main_df(preds_to_make, offense, defense)
    if position is not 'All':
        week_actuals = all_week_actuals_df.query('position == @position')
        for_preds = for_preds_df.query('position == @position')
    else:
        week_actuals = all_week_actuals_df
        for_preds = for_preds_df
    preds = preds_from_factorized_skills(for_preds)
    my_rmse = year_week_rmse(week_actuals, preds)
    
    check_model_string = 'STD in fanduel points:\t{}\nRMSE in predictions:\t{}'
    print(check_model_string.format(week_actuals.fanduel_points.std(), my_rmse))
    

if __name__ == '__main__':
    nfl_frames = NFLFrames()
    year = 2015
    week = 5
    check_nmf_model(nfl_frames, year, week, 'WR')
