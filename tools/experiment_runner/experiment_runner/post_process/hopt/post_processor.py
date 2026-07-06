from itertools import islice
import json
import logging
from pathlib import Path
import re
from typing import Any, Optional, Sequence

import pandas as pd

from experiment_runner.post_process.post_processor import PostProcessor
import tqdm

_logger = logging.getLogger(__name__)


class HOptPostProcessor(PostProcessor):
    @staticmethod
    def supports(path_experiment: Path) -> bool:
        return "_hopt_" in path_experiment.name