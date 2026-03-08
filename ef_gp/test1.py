import inspect
import ef_features_hetero_conformal as m
print("FUNC FILE:", inspect.getsourcefile(m.compute_conformal_q_gp_scaled))
src = inspect.getsource(m.compute_conformal_q_gp_scaled)
print("HAS test_cov ?", "test_cov" in src)
print("HAS _collect_split('test') ?", "_collect_split(\"test\")" in src)

