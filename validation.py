from nfldb_tables import NFLFrames
from modeling import PositionNMFFactory
from sklearn.metrics import mean_squared_error

def year_week_position_rmse(year, week, position, predictions, nfl_frames):
    '''
    Input:  Int - year, Int - week to validate, Str - position predictions are for,
            DataFrame - predictions against, NFLFrames
    Output: RMSE between the predicted values for year-week and actual values from year-week
    '''
    import pdb
    pdb.set_trace()
    week_position_df = nfl_frames.get_year_week_frame(year, week).query('position == @position')
    week_position_pred_df = week_position_df.merge(right=predictions,
                                                         how='inner',
                                                         on='player_id')
    rmse = mean_squared_error(  week_position_pred_df.fanduel_points,
                                week_position_pred_df.prediction) ** .5
    return rmse

def check_model(position, year, week, nfl_frames):
    skill_factory = PositionNMFFactory(year, week, nfl_frames)
    wr_2015_5 = skill_factory.get_position_model(position)
    week_6_opps = nfl_frames.get_year_weeks_opponents(year=year, week=week+1)
    week_6_predictions = wr_2015_5.predict(week_6_opps)
    wr_2015_6_rmse = year_week_position_rmse(year, week, position, week_6_predictions, nfl_frames)

    check_model_string = 'STD in fanduel points:\t{}\nRMSE in predictions:\t{}'
    print check_model_string.format(wr_2015_5.df.fanduel_points.std(), wr_2015_6_rmse)

if __name__ == '__main__':
    nfl_frames = NFLFrames()
    position = 'WR'
    year = 2015
    week = 5
    check_model(position, year, week, nfl_frames)
