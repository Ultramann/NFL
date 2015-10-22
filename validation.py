from nfldb_tables import NFLFrames
from modeling import PositionNMFFactory

def validate_year_week_position(year, week, predictions, nfl_frames):
    '''
    Input:  Int - year, Int - week to validate, Str - position predictions are for,
            DataFrame - predictions against, NFLFrames
    Output: RMSE between the predicted values for year-week and actual values from year-week
    '''
    week_position_df = nfl_frames.get_year_week_frame(year, week).query('position == @position')
    week_postion_predictions_df = week_position_df.merge(right=predictions,
                                                         how='left',
                                                         on='player_id')
    week_postion_predictions_df.eval('difference = fanduel_points - prediction')
    
    # Do some stuff to difference column to get RMSE
    return week_postion_predictions_df.rmse

if __name__ == '__main__':
    nfl_frames = NFLFrames()
    year = 2015
    week = 5
    skill_factory = PositionNMFFactory(year, week, nfl_frames)
    wr_2015_5 = skill_factory.get_position_model('WR')
    week_6_opps = nfl_frames.get_year_weeks_opponents(year=year, week=week+1)
    week_6_predictions = wr_2015_5.predict(week_6_opps)
    '''
    Next step:
    some_number = validate_year_week_position(year, week, 'WR', nfl_frames, week_6_predictions)
    '''
