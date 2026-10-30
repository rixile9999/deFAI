"""Federated Burr XII estimation; intervals must be strictly positive days."""
from defai.research.resources import load_api

_impl = load_api("generation")
DistrictNode = _impl.DistrictNode
ProvinceNode = _impl.ProvinceNode
build_hierarchy = _impl.build_hierarchy
burr_ll_grad = _impl.burr_ll_grad
federated_fit = _impl.federated_fit
pooled_reference = _impl.pooled_reference
load_district_pairs = _impl.load_district_pairs
