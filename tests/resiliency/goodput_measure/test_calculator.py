"""Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Unit Tests for Goodput Calculator

This script contains unit tests for the Goodput Calculator, focusing on the
preprocess_events static method.
"""

import datetime
import json
import os
import sys
from typing import Dict, List
from dateutil import parser as date_parser
import pytest

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resiliency.goodput_measure.constant import (
    USER_SCHEDULED,
    USER_TERMINATED,
    JOB_STARTED,
    JOB_TERMINATED,
    CHECKPOINT_LOADED,
    CHECKPOINT_SAVED,
)

from resiliency.goodput_measure.calculator import GoodputCalculator


class TestGoodputCalculator:
  """Test cases for the GoodputCalculator class."""

  @pytest.fixture(autouse=True)
  def setup(self):
    """Set up test fixtures."""
    # Sample job name for testing
    self.job_name = "test-job"

    # Create a base timestamp for generating event sequences
    self.base_time = datetime.datetime(2025, 4, 1, 12, 0, 0)

  def create_timestamp(self, minutes_offset: int) -> str:
    """Helper to create timestamps with an offset from base time."""
    time = self.base_time + datetime.timedelta(minutes=minutes_offset)
    return time.isoformat()

  def generate_test_events(self) -> List[Dict]:
    """Generate a standard set of test events for a multi-node job."""
    return [
        {
            "node_rank": 0,
            "job_name": self.job_name,
            "event_type": JOB_STARTED,
            "timestamp": self.create_timestamp(1),
        },
        {
            "node_rank": 1,
            "job_name": self.job_name,
            "event_type": JOB_STARTED,
            "timestamp": self.create_timestamp(2),
        },
        {
            "node_rank": 2,
            "job_name": self.job_name,
            "event_type": JOB_STARTED,
            "timestamp": self.create_timestamp(3),
        },
        {
            "job_name": self.job_name,
            "event_type": CHECKPOINT_LOADED,
            "timestamp": self.create_timestamp(5),
            "step": 100,
        },
        {
            "job_name": self.job_name,
            "event_type": CHECKPOINT_SAVED,
            "timestamp": self.create_timestamp(10),
            "step": 200,
        },
        {
            "job_name": self.job_name,
            "event_type": JOB_TERMINATED,
            "timestamp": self.create_timestamp(13),
            "node_rank": 0,
        },
        {
            "job_name": self.job_name,
            "event_type": JOB_TERMINATED,
            "timestamp": self.create_timestamp(14),
            "node_rank": 1,
        },
        {
            "job_name": self.job_name,
            "event_type": JOB_TERMINATED,
            "timestamp": self.create_timestamp(15),
            "node_rank": 2,
        },
    ]

  def test_preprocess_events_basic(self):
    """Test basic preprocessing with complete events."""
    # Create a complete set of events
    events = [{
        "job_name": self.job_name,
        "event_type": USER_SCHEDULED,
        "timestamp": self.create_timestamp(0),
    }]

    # Add standard events
    events.extend(self.generate_test_events())

    # Add user terminated event
    events.append({
        "job_name": self.job_name,
        "event_type": USER_TERMINATED,
        "timestamp": self.create_timestamp(12),
    })

    # Preprocess events
    processed = GoodputCalculator.preprocess_events(events, self.job_name)

    # Verify processing
    expected = [
        (USER_SCHEDULED, self.create_timestamp(0)),
        (JOB_STARTED, self.create_timestamp(3)),
        (CHECKPOINT_LOADED, self.create_timestamp(5)),
        (CHECKPOINT_SAVED, self.create_timestamp(10)),
        (USER_TERMINATED, self.create_timestamp(12)),
        (JOB_TERMINATED, self.create_timestamp(15)),
    ]
    assert len(processed) == len(expected)

    for i, event in enumerate(expected):
      expected_event_type, expected_ts = event
      assert processed[i]["event_type"] == expected_event_type
      assert processed[i]["timestamp"] == expected_ts

  def test_preprocess_events_without_user_scheduled(self):
    """Test basic preprocessing with complete events."""

    # Add standard events
    events = self.generate_test_events()

    # Preprocess events
    processed = GoodputCalculator.preprocess_events(events, self.job_name)

    # Verify processing
    expected = [
        (USER_SCHEDULED, self.create_timestamp(1)),
        (JOB_STARTED, self.create_timestamp(3)),
        (CHECKPOINT_LOADED, self.create_timestamp(5)),
        (CHECKPOINT_SAVED, self.create_timestamp(10)),
        (JOB_TERMINATED, self.create_timestamp(15)),
    ]
    assert len(processed) == len(expected)

    for i, event in enumerate(expected):
      expected_event_type, expected_ts = event
      assert processed[i]["event_type"] == expected_event_type
      assert processed[i]["timestamp"] == expected_ts

  def test_calculate_goodput_with_reference_time(self):
    """Test goodput calculation with a reference step time."""
    # Create a complete set of events
    events = [{
        "job_name": self.job_name,
        "event_type": USER_SCHEDULED,
        "timestamp": self.create_timestamp(0),
    }]

    # Add standard events
    events.extend(self.generate_test_events())

    # Add user terminated event
    events.append({
        "job_name": self.job_name,
        "event_type": USER_TERMINATED,
        "timestamp": self.create_timestamp(12),
    })

    # Preprocess events
    processed = GoodputCalculator.preprocess_events(events, self.job_name)

    # Create a calculator instance
    calculator = GoodputCalculator(job_name=self.job_name)

    # Calculate goodput with reference step time of 1 second per step
    reference_step_time = 8.0  # 8 second per step
    metrics = calculator.calculate_goodput(events, reference_step_time)

    # Verify step-based metrics
    assert metrics.get("min_loaded_step") == 100
    assert metrics.get("max_saved_step") == 200
    assert metrics.get("step_diff") == 100

    # Total time should be 15 minutes = 900 seconds
    assert metrics.get("total_runtime_seconds") == 900

    # Effective computation time should be 100 steps * 8 second = 2000 seconds
    assert metrics.get("effective_computation_time") == 800

    # Goodput = (800 / 900) * 100 = 88.89%
    assert abs(metrics.get("goodput_percentage") - (800 / 900) * 100) < 0.001

  def test_calculate_goodput_with_local_logfile(self):
    job_name = "test-040104-goodput-n112-70b-a3m"
    calculator = GoodputCalculator(
        job_name=job_name, local_log_path="event_log.log"
    )
    events = calculator.load_events()
    preprocessed_events = calculator.preprocess_events(events, job_name)
    reference_step_time = 8
    metrics = calculator.calculate_goodput(
        preprocessed_events, reference_step_time
    )
    expected_metrics = {
        "total_events": 193,
        "job_started_count": 48,
        "checkpoints_loaded": 44,
        "checkpoints_saved": 96,
        "total_runtime_seconds": 35364.285491,
        "useful_runtime_seconds": 0,
        "effective_computation_time": 15360.0,
        "goodput_percentage": 43.43365004195837,
        "job_intervals": [],
        "checkpoint_intervals": [],
        "step_diff": 1920.0,
        "min_loaded_step": 40.0,
        "max_saved_step": 1960.0,
    }
    assert metrics == expected_metrics


if __name__ == "__main__":
  pytest.main([__file__])
