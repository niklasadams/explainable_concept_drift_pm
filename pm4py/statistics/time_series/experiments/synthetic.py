import sys
sys.path.append("../../../../")
from datetime import datetime
import pm4py.statistics.time_series as time_series

print("Hello")
logs = time_series.construct.subdivide_log(
    '/data/Synthetic_Insurance_Claim_large.xes',
    datetime(1970,1,15), datetime(1970,10,27), 1)

pen_primary = 3
pen_secondary = 1.5

primary_names, primary_features = time_series.construct.apply_feature_extraction(
    logs, ["direct_follows_relations"])
reduced_primary = time_series.construct.pca_reduction(primary_features,3,
                                                         normalize = True, 
                                                         normalize_function="max")
cp_1 = time_series.change_points.rpt_pelt(reduced_primary, pen = pen_primary)
print(cp_1)

secondary_names, secondary_features = time_series.construct.feature_extraction.apply_feature_extraction(logs,
                                                                                  ["all_numeric_data"])
reduced_secondary = time_series.construct.pca_reduction(secondary_features,
                                                           3, 
                                                           normalize = True,
                                                           normalize_function="max")
cp_2 = time_series.change_points.rpt_pelt(reduced_secondary, pen = pen_secondary)
print(cp_2)

res = time_series.cause_effect.granger_causality(primary_features, secondary_features,
                                    secondary_names, cp_1, cp_2,
                                    p_value_start = 0.0000000000001)

time_series.cause_effect.draw_ca(res, primary_features, primary_names, secondary_features, secondary_names)