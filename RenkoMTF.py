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

from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter, informative

# --------------------------------
# Add your lib to import here
from freqtrade.exchange import timeframe_to_minutes
import pyrenko
import pandas_ta as pta
from technical import qtpylib
import scipy.optimize as opt

logger = logging.getLogger(__name__)

""" Definitions
A- Evaluate Renko Chart
B- Build Renko Chart based on score function
C- Print pair Renko values and scores
D- Entries
    1- Check if direction transition/, mark as entry for buy/sell based on direction
    2- Check MA(7, hyperoptable), mark it's direction as rising/falling
    3- Maximum % stop allowed for entry (3.5, hyperoptable)
    4- Check Chandelier (21,3, hyperoptable) and compare with last brick if close higher than Chandelier mark as long vice versa
E- Stops
    1- Entry Stop based on ATR (5.5, x5, hyperoptable)
    2- Chandelier Exit kicks in when last brick close over X multiple (2, hyperoptable) 
 """

# Function for optimization
def evaluate_renko(brick, history, column_name):
    renko_obj = pyrenko.renko()
    renko_obj.set_brick_size(brick_size = brick, auto = False)
    renko_obj.build_history(prices = history)
    return renko_obj.evaluate()[column_name]


class RenkoMTF(IStrategy):
    
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
    #startup_candle_count: int = 240

    # These values can be overridden in the config.ino
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    # Run "populate_indicators()" only for new candle.
    # process_only_new_candles = True

