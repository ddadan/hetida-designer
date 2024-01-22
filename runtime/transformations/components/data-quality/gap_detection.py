import contextlib
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, validator

from hetdesrun.runtime.exceptions import ComponentInputValidationException
