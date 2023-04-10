# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter, informative, merge_informative_pair

# --------------------------------
# Add your lib to import here
from freqtrade.exchange import timeframe_to_minutes
import pyrenko
import pandas_ta as pta
from technical import qtpylib
import scipy.optimize as opt

logger = logging.getLogger(__name__)

def evaluate_renko(brick, history, column_name):
    renko_obj = pyrenko.renko()
    renko_obj.set_brick_size(brick_size = brick, auto = False)
    renko_obj.build_history(prices = history)
    return renko_obj.evaluate()[column_name]


class AdaptiveRenkoStrategy(IStrategy):
    
    # By: Mr Robot (@heymrrobot)
    # Renko Tower is a completely New Strategy to find strongly rising coins.

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '3m'
    timeframe_minutes = timeframe_to_minutes(timeframe)

    # Can this strategy go short?
    can_short: bool = True
    use_custom_stoploss = True
    startup_candle_count: int = 40

    # These values can be overridden in the config.ino
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True
    
    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '15m') for pair in pairs]
        return informative_pairs
        
 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        inf_tf = '15m'
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        # Get the 14 day ATR
        atr = pta.atr(informative['high'], informative['low'], informative['close'], length=14)
        atr = atr[np.isnan(atr) == False]  # drop any NaN values in the result
        informative['atr'] = atr
        # Shift date by 1 candle
        # This is necessary since the data is always the "open date"
        # and a 15m candle starting at 12:15 should not know the close of the 1h candle from 12:00 to 13:00
        # minutes = timeframe_to_minutes(inf_tf)
        # Only do this if the timeframes are different:
        # informative['date_merge'] = informative["date"] + pd.to_timedelta(minutes, 'm')
        # Use the helper function merge_informative_pair to safely merge the pair
        # Automatically renames the columns and merges a shorter timeframe dataframe and a longer timeframe informative pair
        # use ffill to have the 1d value available in every row throughout the day.
        # Without this, comparisons between columns of the original and the informative pair would only work once per day.
        # Full documentation of this method, see below
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)
        
        renko_obj_obs = pyrenko.renko()

        # When model is empty
        if len(renko_obj_obs.get_renko_prices()) == 0:
            renko_obj_obs = pyrenko.renko()

            # Get optimal brick size as maximum of score function by Brent's (or similar) method
            # First and Last ATR values are used as the boundaries
            opt_bs = opt.fminbound(lambda x: -evaluate_renko(brick = x, history = dataframe.close, column_name = 'score'), 
                                    np.min(dataframe['atr_15m']), np.max(dataframe['atr_15m']), disp=0)

            # Build the model
            print('REBUILDING RENKO >>>>>>>>>>>>>>  Pair name : ', metadata['pair'], 'Optimal brick size : ', opt_bs)
            last_brick_size = opt_bs            
            renko_obj_obs.set_brick_size(auto=False, brick_size=opt_bs)
            renko_obj_obs.build_history(prices = dataframe.close)
            
            # Store some information
            # evo_results = renko_obj_obs.evaluate()
            score_value = renko_obj_obs.evaluate()['score']
            last_brick_size = opt_bs    
            # dataframe['directions'] = renko_obj_obs.get_renko_directions()
            dataframe.loc[:, ['last_brick_size', 'score_value', 'rebuilding_status', 'brick_size', 'price', 'renko_price', 'num_created_bars']] = (
                last_brick_size,
                score_value,
                1,
                last_brick_size,
                dataframe.close.iloc[-1],
                renko_obj_obs.get_renko_prices()[-1],
                0
            )
            
        else:
            print('Entering else block...')
            last_price = dataframe(close,
                                   bar_count =1
                                    )
                                
            
            # Just for output and debug
            prev = renko_obj_obs.get_renko_prices()[-1]
            prev_dir = renko_obj_obs.get_renko_directions()[-1]
            num_created_bars = renko_obj_obs.do_next(last_price)
            if num_created_bars != 0:
                print('New Renko bars created')
                print('last price: ' + str(last_price))
                print('previous Renko price: ' + str(prev))
                print('current Renko price: ' + str(crenko_obj_obs.get_renko_prices()[-1]))
                print('direction: ' + str(prev_dir))
                print('brick size: ' + str(renko_obj_obs.brick_size))
            
            dataframe.loc[:, ['rebuilding_status', 'brick_size', 'price', 'renko_price', 'num_created_bars', 'last_brick_direction']] = (
                0,
                last_brick_size,
                last_price,
                renko_obj_obs.get_renko_prices()[-1],
                num_created_bars,
                prev_dir
            )
            # record(
            #     rebuilding_status = 0,
            #     brick_size = last_brick_size,
            #     price = last_price,
            #     renko_price = renko_obj_obs.get_renko_prices()[-1],
            #     num_created_bars = num_created_bars,
            #     last_brick_direction = prev_dir
            # )
        print(dataframe.columns)

        return dataframe
            
    
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['last_brick_direction'] == 1)
                &
                (dataframe['last_brick_direction'].shift(1) == -1)
            ),
            ['enter_long', 'enter_tag']] = (1, 'buy_direction')

        dataframe.loc[
            (
                (dataframe['last_brick_direction'] == -1)
                &
                (dataframe['last_brick_direction'].shift(1) == 1)
            ),
            ['enter_short', 'enter_tag']] = (1, 'short_direction')
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['last_brick_direction'] == -1)
                &
                (dataframe['last_brick_direction'].shift(1) == 1)
            ),
            ['exit_long', 'enter_tag']] = (1, 'direction_long_exit')

        dataframe.loc[
            (
                (dataframe['last_brick_direction'] == 1)
                &
                (dataframe['last_brick_direction'].shift(1) == -1)
            ),
            ['exit_short', 'enter_tag']] = (1, 'direction_short_exit')
        
        return dataframe
