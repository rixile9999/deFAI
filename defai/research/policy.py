"""Calendar-aligned additive policy statistics, with exploratory inference."""
from defai.research.resources import load_api

_impl = load_api("policy")
AnalysisConfig = _impl.AnalysisConfig
analyze = _impl.analyze
daily_counts = _impl.daily_counts
district_vector = _impl.district_vector
province_summary = _impl.province_summary
read_district_file = _impl.read_district_file
district_identity = _impl.district_identity
district_treatment = _impl.district_treatment
