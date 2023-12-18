"""Documentation for component "Gap Detection"

# Gap Detection

Processes the given time series and returns the beginning and end timestamps of gaps larger than a
determined or given step size.

## Parameters

**timeseries** (Pandas Series):
    Expects datetime index.

**start_date_str** (String):
    Desired start date of the processing range. Expexcts iso format. If =None, there has to be an
    attribute `start_date` containing the start date as string.

**end_date_str** (String):
    Desired end date of the processing range. Expexcts iso format. If =None, `timeseries` has to
    have an attribute `start_date` containing the start date as string.

**auto_stepsize** (Bool):
    If True, the function will automatically determine the step unit based on the timeseries data.

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

**min_amount_datapoints** :
    Minimum amount of datapoints required for a feasible result.

## Returns

**gap_boundaries** (Pandas DataFrame) :
    A DataFrame containing the beginning and end timestamps of gaps larger than the determined or
    given step size.
    Columns are:
    - "start": Start index of the gap.
    - "end": End index of the gap.
    - ["gap": Size of the gap relative to the stepsize], optional if add_gapsize_column is True

## Raises

ValueError:
    - If start_date is greater than end_date.
    - If history_end_date is not between start_date and end_date.

## Notes

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


def constrict_series_to_dates(
    timeseries_data: pd.Series | pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.Series | pd.DataFrame:
    """
    Constricts a time series to only include data points between a specified start and end date.
    #INCLUSIVE

    Parameters:
    -----------
    timeseries_data :
        Expects datetime index.

    start_date :
        The start date to which the time series should be constricted.

    end_date :
        The end date to which the time series should be constricted.

    Returns:
    --------
    pandas.Series or pandas.DataFrame
        A constricted version of the input timeseries_data, containing only the data points
        that fall between the specified start and end dates.
    """

    ts = timeseries_data[
        (timeseries_data.index >= start_date) & (timeseries_data.index <= end_date)
    ]

    return ts


def determine_timestep_gapsize_percentile(
    timeseries_data: pd.Series | pd.DataFrame, percentil=95
) -> pd.Timedelta:  # determine_timestep_gapsize_percentile
    """
    Determines gap size (between consecutive timestamps) percentile in a time series.

    Parameters:
    -----------
    timeseries_data :
        Expects datetime index.

    percentil :
        The percentile value to use when determining the plausible gap. Expects int in range
        (0,100).

    Returns:
    --------
    percentile_gapsize :
        The gapsize (time step unit) of the {percentil}%-th percentile between consecutive
        timestamps.

    Notes:
    ------
    The function calculates the differences between consecutive timestamps
    to find the gaps. Then, it determines the plausible gap based on the specified
    percentile of these gaps.

    Example:
    --------
    >>> s = pd.Series([1,2,3], index=[pd.Timestamp('2022-01-01'),
                                      pd.Timestamp('2022-01-02'),
                                      pd.Timestamp('2022-01-04')])
    >>> determine_timestep_gapsize_percentile(s)
    Timedelta('1 days 22:48:00')
    """

    # value Error? Nein: nan

    gaps = timeseries_data.index.to_series().diff().dropna()

    percentile_gapsize = gaps.quantile(percentil / 100)  #  interpolation= "nearest"

    return percentile_gapsize


def check_add_boundary_dates(
    timeseries: pd.Series, start_date: datetime, end_date: datetime
) -> pd.Series:
    """
    Ensures that the given start_date and end_date are present in the timeseries index.
    If not, it adds them with a dummy value and then sorts the timeseries by index.

    Parameters:
    -----------
    timeseries :
        Expects datetime index.

    start_date :
        The desired start date to check and potentially add to the timeseries index.

    end_date :
        The desired end date to check and potentially add to the timeseries index.

    Returns:
    --------
    timeseries :
        A version of the input timeseries that includes the specified start_date and end_date
        in its index, each with a dummy value if they were not originally present.

    Notes:
    ------
    The function checks the presence of the start_date and end_date in the timeseries index.
    If either date is missing, it adds that date to the timeseries with a dummy value of math.pi.
    After any additions, the timeseries is sorted by its index.

    Example:
    --------
    >>> s = pd.Series([1,2,3], index=[pd.Timestamp('2022-01-01'),
                                      pd.Timestamp('2022-01-02'),
                                      pd.Timestamp('2022-01-04')])
    >>> start = pd.Timestamp('2021-12-31')
    >>> end = pd.Timestamp('2022-01-05')
    >>> check_add_boundary_dates(s, start, end)
    2021-12-31    3.141592653589793
    2022-01-01    1
    2022-01-02    2
    2022-01-04    3
    2022-01-05    3.141592653589793
    dtype: int64
    """

    if start_date not in timeseries.index:
        timeseries[start_date] = math.pi

    if end_date not in timeseries.index:
        timeseries[end_date] = math.pi

    timeseries = timeseries.sort_index()

    return timeseries


# doesn't handle missing data at beginning / end, fix: check_add_boundary_dates(...)
def check_length_timeseries(
    timeseries_data: pd.Series | pd.DataFrame, stepsize=timedelta(minutes=1)
) -> bool:
    """
    Checks if a time series is longer than or equal to one year in terms of duration and amount of
    data points.

    Parameters:
    -----------
    timeseries_data :
        Expects datetime index.

    stepsize :
        The expected time step unit between consecutive timestamps in the time series.
        This is used to determine the acceptable duration slightly less than one year.

    Returns:
    --------
    is longer :
        True if the timeseries_data has >= 365 data points and its duration is >=
        (365 days - stepsize). Otherwise, returns False.

    Example:
    --------
    >>> s = pd.Series([1,2,3], index=[pd.Timestamp('2020-01-01'),
                                      pd.Timestamp('2021-01-01'),
                                      pd.Timestamp('2022-01-01')])
    >>> check_length_timeseries(s)
    False
    """

    start_date = timeseries_data.index[0]
    end_date = timeseries_data.index[-1]

    length_timeseries = end_date - start_date

    is_longer = len(timeseries_data) >= 365

    if is_longer:
        is_longer = length_timeseries >= (pd.Timedelta(days=365) - stepsize)

    return is_longer


def check_amount_datapoints(
    timeseries_data: pd.Series | pd.DataFrame,
    min_amount: int = 365,
    dummy_value: float = math.pi,
) -> bool:
    """
    Returns {len(timeseries_data) >= min_amount}. # Checks if a time series has at least a
    specified amount of datapoints.

    Parameters:
    -----------
    timeseries_data :
        Expects datetime index and column "value".

    min_amount :
        Optional. Minimum desired amount of Data points.

    Returns:
    --------
    is_longer :
        True if the timeseries_data has >= 365 data points and its duration is >=
        (365 days - stepsize). Otherwise, returns False.

    Example:
    --------
    >>> s = pd.Series([1,2,3], index=[pd.Timestamp('2020-01-01'),
                                      pd.Timestamp('2021-01-01'),
                                      pd.Timestamp('2022-01-01')])
    >>> check_amount_datapoints(s)
    False
    """

    # timeseries_data = timeseries_data[timeseries_data["value"] != dummy_value], falsche Stelle für die Aufgabe

    return len(timeseries_data) >= min_amount


def determine_gap_length(
    timeseries: pd.Series, stepsize=timedelta(minutes=1)
) -> pd.DataFrame:
    """
    Determines the length of the gaps between consecutive timestamps in a time series
    normalized by a specified step unit.

    Parameters:
    -----------
    timeseries :
        Expects datetime index.

    stepsize :
        The time step unit used for normalization. Gaps between timestamps will
        be divided by this unit to determine the number of units the gap spans.

    Returns:
    --------
    result_df :
        A DataFrame with the original values of the timeseries and the normalized gaps.
        Columns are:
        - "value": original values from the timeseries.
        - "gap": normalized gaps, representing the number of timesteps between consecutive timestamps.

    Notes:
    ------
    The function calculates the differences between consecutive timestamps
    to find the gaps. Then, it normalizes these gaps by dividing them by the specified stepsize.
    This provides a measure of how many timesteps each gap spans.

    Example:
    --------
    >>> s = pd.Series([1,2,3], index=[pd.Timestamp('2022-01-01 00:00:00'),
                                      pd.Timestamp('2022-01-01 00:05:00'),
                                      pd.Timestamp('2022-01-01 00:20:00')])
    >>> determine_gap_length(s)
                  value  gap
    2022-01-01 00:00:00      1   NaN
    2022-01-01 00:05:00      2   5.0
    2022-01-01 00:20:00      3  15.0
    """

    gaps = timeseries.index.to_series().diff().values

    stepsize_seconds = stepsize.total_seconds()

    normalized_gaps = [
        pd.Timedelta(gap).total_seconds() / stepsize_seconds
        if pd.notnull(gap)
        else None
        for gap in gaps
    ]

    result_df = pd.DataFrame(
        {"value": timeseries.values, "gap": normalized_gaps}, index=timeseries.index
    )

    return result_df


def return_gap_boundary_timestamps(frame_with_gapsizes: pd.DataFrame) -> pd.DataFrame:
    """
    Detects the beginning and end timestamps of gaps larger than 1, gap size is read from column
    "gaps".

    Parameters:
    -----------
    frame_with_gapsizes :
        Expects column "gaps".

    Returns:
    --------
    result_df :
        A DataFrame with the beginning and end timestamps of gaps larger than the step_unit.
        Columns are:
        - "start": Start index of the gap.
        - "end": End index of the gap.

    Example:
    --------
    >>> df = pd.DataFrame(f{"value": [1, 2, 4],"gap": [None, 1.0, 3.0]},
                index=[ pd.Timestamp('2022-01-01 00:00:00'),
                        pd.Timestamp('2022-01-01 00:01:00'),
                        pd.Timestamp('2022-01-01 00:04:00')])
    >>> return_gap_boundary_timestamps(df)
                     start                 end
    2022-01-01 00:01:00 2022-01-01 00:01:00 2022-01-01 00:04:00
    """

    # Identify rows where gap is greater than 1
    large_gap_indices = frame_with_gapsizes[frame_with_gapsizes["gap"] > 1].index

    # Extract the start and end timestamps of the gaps
    gap_starts = [
        frame_with_gapsizes.index[idx - 1]
        for idx, _ in enumerate(frame_with_gapsizes.index)
        if _ in large_gap_indices
    ]

    # Create a DataFrame to store the results
    result_df = pd.DataFrame({"start": gap_starts, "end": large_gap_indices})

    return result_df


def add_gapsize_column_to_frame(
    frame_with_gap_boundaries: pd.DataFrame, frame_with_gapsizes: pd.DataFrame
) -> pd.DataFrame:
    """
    Given a DataFrame with a DateTimeIndex and a DataFrame with timestamps
    in the "end" column, add a new column "gaps" to the timestamps DataFrame
    with the corresponding gap values.

    Parameters:
    -----------
    frame_with_gapsizes :
        Expects column "gaps" and DateTimeIndex.

    frame_with_gap_boundaries :
        Expects timestamps in column "end".

    Returns:
    --------
    frame_with_gap_boundaries :
        Copy of frame_with_gap_boundaries with a new column "gaps".
    """
    frame_with_gap_boundaries["gapsize"] = frame_with_gap_boundaries["end"].apply(
        lambda timestamp: frame_with_gapsizes.at[timestamp, "gap"]
        if timestamp in frame_with_gapsizes.index
        else None
    )

    return frame_with_gap_boundaries


def generate_gaps_length_0(
    gap_timestamps: pd.Series, add_gapsize_column: bool = True
) -> pd.DataFrame:
    """
    [...]

    Parameters:
    -----------
    gap_timestamps :
        Expects datetime values.

    add_gapsize_column :
        Determines whether a column "gap" with the gapsizes is added.

    Returns:
    --------
    new_gaps :
        A DataFrame with the beginning and end timestamps of gaps [...]
    """

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


def generate_gap_intervals(
    timeseries_data: pd.Series, gap_timestamps: pd.Series, add_gapsize_column: bool
) -> pd.DataFrame:
    """
    [...]

    Parameters:
    -----------
    timeseries_data :
        The timeseries for which the gaps are generated. Expects DateTimeIndex.

    gap_timestamps :
        Expects datetime values.

    add_gapsize_column :
        Determines whether a column "gap" with the gapsizes is added.

    Returns:
    --------
    new_gaps :
        A DataFrame with the beginning and end timestamps of gaps [...]
    """

    start_stamps = []
    end_stamps = []
    gap_sizes = []

    # TODO sortieren, falls erlaubt?

    for timestamp in gap_timestamps:
        # found = False
        prev_index = None
        next_index = None

        # TODO geht das auch eleganter?
        with contextlib.suppress(IndexError):
            prev_index = timeseries_data[timeseries_data.index < timestamp].index[-1]
        with contextlib.suppress(IndexError):
            next_index = timeseries_data[timeseries_data.index > timestamp].index[0]

        # if timestamp in timeseries_data.index:
        #     found = True
        #     timeseries_data = timeseries_data[timeseries_data.index != timestamp]

        # TODO was ist mit None?

        start_stamps.append(prev_index)
        end_stamps.append(next_index)
        gap_sizes.append(next_index - prev_index)

    if add_gapsize_column:
        new_gaps = pd.DataFrame(
            {"start": start_stamps, "end": end_stamps, "gap": gap_sizes}
        )
    else:
        new_gaps = pd.DataFrame({"start": start_stamps, "end": end_stamps})

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


def main(
    timeseries: pd.Series,
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    auto_stepsize: bool = True,
    history_end_date_str: str | None = None,
    step_size_str: str | None = None,
    percentil: int = 95,
    min_amount_datapoints: int = 21,
    add_gapsize_column=True,
) -> dict[str, pd.DataFrame]:
    """Example:
    --------
    >>> ts = pd.Series(
            range(367),
            index=pd.date_range(start="2022-01-01T00:00:00Z", periods=367, freq="D"))
    >>> s = s[s.index != pd.to_datetime('2022-01-03 00:00:00', utc = True)]
    >>> start = pd.Timestamp('2022-01-01 00:00:00')
    >>> end = pd.Timestamp('2023-01-02 00:00:00')
    >>> main(ts, start, end)
                        start                 end
    0	2022-01-02 00:00:00+00:00	2022-01-04 00:00:00+00:00"""

    timeseries = timeseries.dropna()

    if timeseries.empty:
        raise ValueError(
            "Input timeseries must be not empty for gap detection."
        )  # grammatik, Zweck/Handlungsanweisung hinzufügen: check

    if start_date_str is None:
        try:
            start_date_str = timeseries.attrs["start_date"]  # wirklich ein String?
        except KeyError as e:
            raise KeyError(
                "The gap detection requires a start date. The parameter can be passed via start_date_str or as attribute 'start_date' of the timeseries."
            ) from e

    try:
        start_date = datetime.fromisoformat(start_date_str).replace(
            tzinfo=timezone.utc
        )  # TODO utc erzwingen: check
    except ValueError:  # TODO Errortyp überprüfen: check
        raise ValueError(
            "The date in start_date_str has to be formatted in iso format to allow conversion to datetime type for gap detection."
        )  # TODO fehlermeldung: check
    except TypeError:
        raise TypeError(
            f"The start_date_str, possibly obtained from timeseries.attrs['start_date'], has to be of type str while it is of type {type(start_date_str)}."
        )

    if end_date_str is None:
        try:
            end_date_str = timeseries.attrs["end_date"]
        except KeyError:
            raise KeyError(  # Hier auch KeyError?
                "The gap detection requires an end date. The parameter can be passed via end_date_str or as attribute 'end_date' of the timeseries."
            )

    try:
        end_date = datetime.fromisoformat(end_date_str).replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(
            "The date in end_date_str has to be formatted in iso format to allow conversion to datetime type for gap detection."
        )
    except TypeError:
        raise TypeError(
            f"The end_date_str, possibly obtained from timeseries.attrs['end_date'], has to be of type str while it is of type {type(end_date_str)}."
        )

    if start_date > end_date:
        raise ValueError(
            f"The value start_date has to be lower than end_date, while it is {start_date} > {end_date}."
        )
    if (auto_stepsize is False) and (step_size_str is None):
        raise ValueError(
            f"A step_size is required for gap detection, if it is not automatically determined."
        )
    if history_end_date_str is not None:
        try:
            history_end_date = datetime.fromisoformat(history_end_date_str).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            raise ValueError(
                "The date in history_end_date_str has to be formatted in iso format to allow conversion to datetime type for gap detection."
            )

        if start_date > history_end_date:
            raise ValueError(
                f"The value history_end_date has to be inbetween start_date and end_date, while it is {history_end_date} < {start_date}."
            )
        elif end_date < history_end_date:
            raise ValueError(
                f"The value history_end_date has to be inbetween start_date and end_date, while it is {history_end_date} > {end_date}."
            )
    else:
        history_end_date = None

    if (
        check_amount_datapoints(timeseries, min_amount_datapoints) == False
    ):  # null werte ignorieren: werden schon vorher entfernt.
        raise ValueError(
            f"Timeseries must contain at least {min_amount_datapoints} datapoints"
        )

    constricted_series = constrict_series_to_dates(timeseries, start_date, end_date)
    if auto_stepsize:
        if history_end_date != None:
            training_series = constrict_series_to_dates(
                timeseries, start_date, history_end_date
            )
        else:
            training_series = constricted_series
        step_size = determine_timestep_gapsize_percentile(training_series, percentil)
    else:
        step_size = freqstr2timedelta(step_size_str)
    series_with_bounds = check_add_boundary_dates(
        constricted_series, start_date, end_date
    )

    df_with_gaps = determine_gap_length(series_with_bounds, step_size)
    gap_boundaries = return_gap_boundary_timestamps(df_with_gaps)

    if add_gapsize_column:
        gap_boundaries = add_gapsize_column_to_frame(gap_boundaries, df_with_gaps)

    return {"gap_boundaries": gap_boundaries}
