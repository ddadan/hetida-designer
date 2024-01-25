"""Documentation for component "Gap Detection Points"

# Gap Detection Points

## Description
Processes the given time series and returns the beginning and end timestamps of gaps larger than a
determined or given step size.

## Inputs
**timeseries** (Pandas Series):
    Expects datetime index.

**start_date_str** (String):
    Desired start date of the processing range. Expexcts iso format. If =None, there has to be an
    attribute `start_date` containing the start date as string.

**end_date_str** (String):
    Desired end date of the processing range. Expexcts iso format. If =None, `timeseries` has to
    have an attribute `start_date` containing the start date as string.

**history_end_date_str** (String):
    Expects a date between start_date and end_date in iso format.
    The desired end date for the training data used to determine the step unit.
    This is only relevant when auto_stepsize is True. If not specified, the entire
    `constricted_series` is used.

**step_size** (String):
    Must be not None, when auto_stepsize == False.
    The expected time step unit between consecutive timestamps in the time series.

**percentil** (Float):
    Expects value >= 0 & <= 100.
    The pecentile value to use for automatic determination of the expected gapsize between two
    consecutive data points.
    This is only relevant when auto_stepsize is True.

## Outputs
**gaps** (Pandas DataFrame) :
    A DataFrame containing the beginning and end timestamps of gaps larger than the determined or
    given step size. Index is of type DateTimeIndex.
    Columns are:
    - "time" (Timestamp): Index for the missing datapoint [NO COLUMN]
    - "value_to_left":
    - "value_to_right":
    - "mean_left_right":
    # start/end inclusive?
    # value_to_left & right Wert oder null; wenn beide ex.: Mittelwert aus L&R sonst null

## Raises

ValueError:
    - If start_date is greater than end_date.
    - If history_end_date is not between start_date and end_date.

## Details

The function follows these steps:
1. Validate input.
2. Constrict the timeseries to the given start_date and end_date.
3. If auto_stepsize is True, determine the step unit based on the timeseries data.
4. Ensure that start_date and end_date are present in the constricted timeseries.
5. Detect gaps in the timeseries and determine their boundaries.
"""

import contextlib
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# from pydantic import BaseModel, ValidationError, root_validator, validator  # noqa: ERA001

# from hetdesrun.runtime.exceptions import ComponentInputValidationException  # noqa: ERA001


def generate_gaps_length_0(
    gap_timestamps: pd.Series, add_gapsize_column: bool = True
) -> pd.DataFrame:
    length = len(gap_timestamps)
    pseudo_series_0 = pd.Series(
        [pd.Timedelta(seconds=0)] * length
    )  # TODO timedelta in gap_boundaries

    if add_gapsize_column:
        new_gaps = pd.DataFrame(
            {"start": gap_timestamps, "end": gap_timestamps, "gap": pseudo_series_0}
        )
    else:
        new_gaps = pd.DataFrame({"start": gap_timestamps, "end": gap_timestamps})

    return new_gaps


def generate_gaps_from_freq_and_offset(
    timeseries_data: pd.Series, start_timestamp: pd.Timestamp, count: int, freq_str: str
) -> pd.DataFrame:
    # under construction

    new_gaps = pd.DataFrame()

    date_rng = pd.date_range(
        start=start_timestamp, periods=count, freq=freq_str
    )  # was ist, wenn offset keine absolute timestamp, sondern zb "4 nach..."

    missing_timestamps = []

    for timestamp in date_rng:
        if timestamp not in timeseries_data.index:
            missing_timestamps.append(timestamp)

    missing_timestamp_series = pd.Series(missing_timestamps)

    # siehe gleitender Mittelwert
    return generate_gaps_length_0(missing_timestamp_series)


def freqstr2dateoffset(freqstr: str) -> pd.DateOffset:
    """Transform frequency string to Pandas DateOffset."""
    return pd.tseries.frequencies.to_offset(freqstr)


def freqstr2timedelta(freqstr: str) -> pd.Timedelta:
    """Transform frequency string to Pandas Timedelta."""
    try:
        return pd.to_timedelta(freqstr)
    except ValueError:
        return pd.to_timedelta(freqstr2dateoffset(freqstr))
