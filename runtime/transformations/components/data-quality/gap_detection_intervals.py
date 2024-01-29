"""Documentation for component "Gap Detection Intervals"

# Gap Detection Intervals

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

**min_amount_datapoints** (Integer):
    Minimum amount of datapoints required for a feasible result.

## Outputs
**gap_boundaries** (Pandas DataFrame) :
    A DataFrame containing the beginning and end timestamps of gaps larger than the determined or
    given step size.
    Columns are:
    - "start" (Timestamp): Start index of the gap.
    - "end"(Timestamp): End index of the gap.
    - ["gap": Size of the gap relative to the stepsize], optional if add_gapsize_column is True
    - "start_inclusive" (Boolean):
    - "end_inclusive" (Boolean):
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
from pydantic import BaseModel, ValidationError, root_validator, validator

from hetdesrun.runtime.exceptions import ComponentInputValidationException


class GapDetectionParameters(BaseModel):
    timeseries: pd.Series
    start_date: str  # pydantic kann auch direkt datetime, muss aber getestet werden
    end_date: str
    auto_stepsize: bool = True
    history_end_date: str = None
    step_size_str: str = None
    percentil: float
    min_amount_datapoints: int
    add_gapsize_column: bool = True

    @validator("min_amount_datapoints")
    def verify__min_amount_datapoints(cls, min_amount) -> int:
        if min_amount < 0:
            raise ComponentInputValidationException(
                "The minimum amount of datapoints has to be a non-negative integer.",
                error_code=422,
                invalid_component_inputs=["min_amount_datapoints"],
            )
        return min_amount

    @validator("timeseries")
    def verify_amount_datapoints_in_series(cls, series, values: dict) -> pd.Series:
        min_amount = values["min_amount_datapoints"]
        if len(series) < min_amount:
            raise ValueError(
                f"Timeseries must contain at least {min_amount} datapoints.",
                error_code=422,
                invalid_component_inputs=["timeseries"],
            )

    @validator(
        "start_date", "end_date"
    )  # TODO was ist mit dem Fall Attribut der Zeitreihe?
    def verify_date_strings(cls, date) -> datetime:
        date = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        return date

    @validator("end_date")
    def verify_dates(cls, end_date, values: dict):
        start_date = values["start_date"]
        if start_date > end_date:
            raise ComponentInputValidationException(
                "The value start_date must not be later than the end_date, while it is "
                f"{start_date} > {end_date}.",
                error_code=422,
                invalid_component_inputs=["end_date_str", "start_date_str"],
            )
        return values

    @validator("history_end_date")
    def verify_history_end_date(cls, history_end_date, values: dict) -> datetime | None:
        start_date = values["start_date"]
        end_date = values["end_date"]
        if history_end_date is not None:
            try:
                history_end_date = datetime.fromisoformat(history_end_date).replace(
                    tzinfo=timezone.utc
                )
            except ValueError as err:
                raise ComponentInputValidationException(
                    "The date in history_end_date has to be formatted in iso format to allow "
                    "conversion to datetime type for gap detection.",
                    error_code=422,
                    invalid_component_inputs=["history_end_date_str"],
                ) from err

            if start_date > history_end_date:
                raise ComponentInputValidationException(
                    "The value history_end_date has to be inbetween start_date and end_date, while "
                    f"it is {history_end_date} < {start_date}.",
                    error_code=422,
                    invalid_component_inputs=["history_end_date_str"],
                )
            if end_date < history_end_date:
                raise ComponentInputValidationException(
                    "The value history_end_date has to be inbetween start_date and end_date, while "
                    f"it is {history_end_date} > {end_date}.",
                    error_code=422,
                    invalid_component_inputs=["history_end_date_str"],
                )
        else:
            history_end_date = None
        return history_end_date

    @validator("step_size_str")  # TODO auf freq string überprüfen
    def verify_step_size(cls, step_size, values: dict) -> str:
        auto_stepsize = values["auto_stepsize"]
        if (auto_stepsize is False) and (step_size is None):
            raise ComponentInputValidationException(
                "A step_size is required for gap detection, if it is not automatically determined.",
                error_code=422,
                invalid_component_inputs=["step_size_str"],
            )
        return step_size

    @validator("percentil")
    def verify_percentile(cls, percentil) -> int:
        if (percentil < 0) or (percentil > 100):
            raise ComponentInputValidationException(
                "The percentil value has to be a non-negative integer less or equal to 100.",
                error_code=422,
                invalid_component_inputs=["percentil"],
            )
        return percentil


def constrict_series_to_dates(
    timeseries_data: pd.Series | pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.Series | pd.DataFrame:
    ts = timeseries_data[
        (timeseries_data.index >= start_date) & (timeseries_data.index <= end_date)
    ]

    return ts


def determine_timestep_gapsize_percentile(
    timeseries_data: pd.Series | pd.DataFrame, percentil=95
) -> pd.Timedelta:
    gaps = timeseries_data.index.to_series().diff().dropna()

    percentile_gapsize = gaps.quantile(
        percentil / 100, interpolation="nearest"
    )  #  TODO nachfragen wegen Interpol

    return percentile_gapsize


def check_add_boundary_dates(
    timeseries: pd.Series, start_date: datetime, end_date: datetime
) -> pd.Series:
    if start_date not in timeseries.index:
        timeseries[start_date] = math.pi

    if end_date not in timeseries.index:
        timeseries[end_date] = math.pi

    timeseries = timeseries.sort_index()

    return timeseries


# TODO als fehlerhaft markierte Werte aussortieren (nicht hier)


def determine_gap_length(
    timeseries: pd.Series, stepsize=timedelta(minutes=1)
) -> pd.DataFrame:
    gaps = timeseries.index.to_series().diff().to_numpy()

    stepsize_seconds = stepsize.total_seconds()

    normalized_gaps = [
        pd.Timedelta(gap).total_seconds() / stepsize_seconds if pd.notnna(gap) else None
        for gap in gaps
    ]

    result_df = pd.DataFrame(
        {"value": timeseries.to_numpy(), "gap": normalized_gaps}, index=timeseries.index
    )

    return result_df


def return_gap_boundary_timestamps(frame_with_gapsizes: pd.DataFrame) -> pd.DataFrame:
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
    frame_with_gap_boundaries["gapsize"] = frame_with_gap_boundaries["end"].apply(
        lambda timestamp: frame_with_gapsizes.loc[
            timestamp, "gap"
        ]  # TODO vielleicht numpy für bessere Laufzeit?
        if timestamp in frame_with_gapsizes.index
        else None
    )

    return frame_with_gap_boundaries


def freqstr2dateoffset(freqstr: str) -> pd.DateOffset:
    """Transform frequency string to Pandas DateOffset."""
    return pd.tseries.frequencies.to_offset(freqstr)


def freqstr2timedelta(freqstr: str) -> pd.Timedelta:
    """Transform frequency string to Pandas Timedelta."""
    try:
        return pd.to_timedelta(freqstr)
    except ValueError:
        return pd.to_timedelta(freqstr2dateoffset(freqstr))


# ***** DO NOT EDIT LINES BELOW *****
# These lines may be overwritten if component details or inputs/outputs change.
COMPONENT_INFO = {
    "inputs": {
        "timeseries": {"data_type": "SERIES"},
        "start_date_str": {"data_type": "STRING", "default_value": None},
        "end_date_str": {"data_type": "STRING", "default_value": None},
        "auto_stepsize": {"data_type": "BOOLEAN", "default_value": True},
        "history_end_date_str": {"data_type": "STRING", "default_value": None},
        "step_size_str": {"data_type": "STRING", "default_value": None},
        "percentil": {"data_type": "INT", "default_value": 95},
        "min_amount_datapoints": {"data_type": "INT", "default_value": 21},
        "add_gapsize_column": {"data_type": "BOOLEAN", "default_value": True},
    },
    "outputs": {
        "gap_boundaries": {"data_type": "DATAFRAME"},
    },
    "name": "Gap Detection Intervals",
    "category": "Data Quality",
    "description": "Determine gaps in the input series and return gap intervals",
    "version_tag": "0.1.1",
    "id": "9caff8af-3dcb-4b23-aa23-86dfa7e406c8",
    "revision_group_id": "4ae5d3c6-9c3e-4ea6-a602-e927b470ba3c",
    "state": "RELEASED",
    "released_timestamp": "2024-01-10T09:26:09.131242+00:00",
}


def main(
    *,
    timeseries,
    start_date_str=None,
    end_date_str=None,
    auto_stepsize=True,
    history_end_date_str=None,
    step_size_str=None,
    percentil=95,
    min_amount_datapoints=21,
    add_gapsize_column=True,
):
    # entrypoint function for this component
    # ***** DO NOT EDIT LINES ABOVE *****
    # write your function code here.

    timeseries = timeseries.dropna()

    input_params = GapDetectionParameters(
        timeseries=timeseries,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        auto_stepsize=auto_stepsize,
        history_end_date_str=history_end_date_str,
        step_size=step_size_str,
        percentil=percentil,
        min_amount_datapoints=min_amount_datapoints,
        add_gapsize_column=add_gapsize_column,
    )

    constricted_series = constrict_series_to_dates(
        timeseries, input_params.start_date, input_params.end_date
    )  # TODO was ist, wenn dadurch Prüfung auf mindestanzahl umgangen wird?
    if auto_stepsize:
        if input_params.history_end_date is not None:
            training_series = constrict_series_to_dates(
                timeseries, input_params.start_date, input_params.history_end_date
            )
        else:
            training_series = constricted_series
        step_size = determine_timestep_gapsize_percentile(training_series, percentil)
    else:
        step_size = freqstr2timedelta(step_size_str)
    series_with_bounds = check_add_boundary_dates(
        constricted_series, input_params.start_date, input_params.end_date
    )

    df_with_gaps = determine_gap_length(series_with_bounds, step_size)
    gap_boundaries = return_gap_boundary_timestamps(df_with_gaps)

    if add_gapsize_column:
        gap_boundaries = add_gapsize_column_to_frame(gap_boundaries, df_with_gaps)

    return {"gap_boundaries": gap_boundaries}


TEST_WIRING_FROM_PY_FILE_IMPORT = {
    "input_wirings": [
        {
            "workflow_input_name": "timeseries",
            "adapter_id": "direct_provisioning",
            "filters": {
                "value": (
                    "{\n"
                    '    "2020-01-01T01:15:00.000Z": 10.0,\n'
                    '    "2020-01-01T01:16:00.000Z": 10.0,\n'
                    '    "2020-01-01T01:17:00.000Z": 10.0,\n'
                    '    "2020-01-01T01:18:00.000Z": 10.0,\n'
                    '    "2020-01-01T01:19:00.000Z": 10.0,\n'
                    '    "2020-01-01T01:20:00.000Z": 10.0,\n'
                    '    "2020-01-01T01:21:00.000Z": 10.0,\n'
                    '    "2020-01-02T16:20:00.000Z": 20.0,\n'
                    '    "2020-01-02T16:21:00.000Z": 20.0,\n'
                    '    "2020-01-02T16:22:00.000Z": 20.0,\n'
                    '    "2020-01-02T16:23:00.000Z": 20.0,\n'
                    '    "2020-01-02T16:24:00.000Z": 20.0,\n'
                    '    "2020-01-02T16:25:00.000Z": 20.0,\n'
                    '    "2020-01-02T16:26:00.000Z": 20.0,\n'
                    '    "2020-01-03T08:20:00.000Z": 30.0,\n'
                    '    "2020-01-03T08:21:04.000Z": 30.0,\n'
                    '    "2020-01-03T08:22:00.000Z": 30.0,\n'
                    '    "2020-01-03T08:23:04.000Z": 30.0,\n'
                    '    "2020-01-03T08:24:00.000Z": 30.0,\n'
                    '    "2020-01-03T08:25:04.000Z": 30.0,\n'
                    '    "2020-01-03T08:26:06.000Z": 30.0\n'
                    "}"
                )
            },
        },
        {
            "workflow_input_name": "start_date_str",
            "adapter_id": "direct_provisioning",
            "filters": {"value": "2020-01-01T01:15:27.000Z"},
        },
        {
            "workflow_input_name": "end_date_str",
            "adapter_id": "direct_provisioning",
            "filters": {"value": "2020-01-03T08:26:06.000Z"},
        },
        {
            "workflow_input_name": "auto_stepsize",
            "adapter_id": "direct_provisioning",
            "filters": {"value": "True"},
        },
        {
            "workflow_input_name": "history_end_date_str",
            "adapter_id": "direct_provisioning",
            "filters": {"value": "2020-01-01T01:21:00.000Z"},
        },
        {
            "workflow_input_name": "step_size_str",
            "adapter_id": "direct_provisioning",
            "filters": {"value": "60s"},
        },
        {
            "workflow_input_name": "percentil",
            "adapter_id": "direct_provisioning",
            "filters": {"value": "95"},
        },
        {
            "workflow_input_name": "min_amount_datapoints",
            "adapter_id": "direct_provisioning",
            "filters": {"value": "21"},
        },
        {
            "workflow_input_name": "add_gapsize_column",
            "adapter_id": "direct_provisioning",
            "filters": {"value": "True"},
        },
    ],
}
