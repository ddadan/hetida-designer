import contextlib
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, validator

from hetdesrun.runtime.exceptions import ComponentInputValidationException


class GapDetectionParameters(BaseModel):
    start_date: str  # pydantic kann auch direkt datetime, muss aber getestet werden
    end_date: str
    auto_stepsize: bool = True
    history_end_date: str = None
    step_size_str: str
    percentil: float
    min_amount_datapoints: int
    add_gapsize_column: bool = True

    @validator("start_date", "end_date")
    def verify_date_strings(cls, date) -> datetime:
        date = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        return date

    @validator("history_end_date")
    def verify_history_end_date(cls, history_end_date, values: dict) -> datetime | None:
        start_date = values["start_date"]
        end_date = values["end_date"]
        if history_end_date is not None:
            try:
                history_end_date = datetime.fromisoformat(history_end_date).replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                raise ValueError(
                    "The date in history_end_date has to be formatted in iso format to allow conversion to datetime type for gap detection."
                )

            if start_date > history_end_date:
                raise ValueError(  # TODO vielleicht ComponenInputValidationError?
                    f"The value history_end_date has to be inbetween start_date and end_date, while it is {history_end_date} < {start_date}."
                )
            elif end_date < history_end_date:
                raise ValueError(
                    f"The value history_end_date has to be inbetween start_date and end_date, while it is {history_end_date} > {end_date}."
                )
        else:
            history_end_date = None
        return history_end_date

    @validator("step_size_str")
    def verify_step_size(cls, step_size) -> str:
        return step_size

    @validator("percentil")
    def verify_percentile(cls, percentil) -> int:
        if (percentil < 0) or (percentil > 100):
            raise ValueError(
                "The percentil value has to be a positive value less or equal to 100."
            )
        if percentil >= 1:
            percentil = percentil / 100
        return percentil

    @root_validator(
        skip_on_failure=True
    )  # nur validator fürs end date + einen für step_size
    def verify_dates(cls, values: dict):
        start_date = values["start_date"]
        end_date = values["end_date"]
        history_end_date = values["history_end_date"]
        auto_stepsize = values["auto_stepsize"]
        step_size_str = values["step_size_str"]
        if start_date > end_date:
            raise ValueError(
                f"The value start_date has to be lower than end_date, while it is {start_date} > {end_date}."
            )
        if (auto_stepsize is False) and (step_size_str is None):
            raise ValueError(
                f"A step_size is required for gap detection, if it is not automatically determined."
            )

        return values

    @root_validator(skip_on_failure=True)
    def verify(cls, values: dict):
        start_date = values["start_date"]
        end_date = values["end_date"]
        history_end_date = values["history_end_date"]
        auto_stepsize = values["auto_stepsize"]
        step_size_str = values["step_size_str"]

        return values
