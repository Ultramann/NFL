from nfldb_tables import NFLFrames
from sklearn.metrics import mean_squared_error
from factorize import nmf_all_positions, merge_factorizations_to_main_df, pred_from_factorized_skills
from data_prep_tools import get_yr_until_wk, get_preds_to_make, positions_list

def year_week_rmse(week_actuals_df, preds):
    '''
    Input:  DataFrames with actual fanduel_points and DF with pred column.  
            Both must have player_id
    Output: RMSE between predicted and actual values from year-week
    '''
    
    actuals_with_preds_df = week_actuals_df.merge( right=preds,
                                        how='inner',
                                        on='player_id')
    rmse = mean_squared_error(  actuals_with_preds_df.fanduel_points,
                                actuals_with_preds_df.pred) ** .5
    return rmse

def check_nmf_model(year, week, position="All", nfl_frames=NFLFrames()):
    historical_data = get_yr_until_wk(year, week, nfl_frames)
    offense, defense = nmf_all_positions(historical_data)
    week_actuals_df = nfl_frames.get_year_week_frame(year, week)
    preds_to_make = get_preds_to_make(year, week+1, nfl_frames)
    for_pred_df = merge_factorizations_to_main_df(preds_to_make, offense, defense)
    if position is not "All":
        pos_week_actuals = week_actuals_df.query("position==@position")
        pos_for_pred = for_pred_df.query("position==@position")
    preds = pred_from_factorized_skills(pos_for_pred)
    my_rmse = year_week_rmse(pos_week_actuals, preds)
    
    check_model_string = 'STD in fanduel points:\t{}\nRMSE in predictions:\t{}'
    print check_model_string.format(pos_week_actuals.fanduel_points.std(), my_rmse)

if __name__ == '__main__':
    nfl_frames = NFLFrames()
    year = 2015
    week = 5
    check_nmf_model(year, week, "WR", nfl_frames)
