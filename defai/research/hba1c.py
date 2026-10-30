"""Batched independent client learning; importing does not change GPU/thread limits."""
from defai.research.resources import load_api

_impl = load_api("hba1c")
initial_parameters = _impl.initial_parameters
local_fit = _impl.local_fit
Learner = _impl.Learner
evaluate = _impl.evaluate
run_training = _impl.run_training
_cohort = load_api("cohort")
World = _cohort.World
calibrate = _cohort.calibrate
make_world = _cohort.make_world
training_schedule = _cohort.training_schedule
