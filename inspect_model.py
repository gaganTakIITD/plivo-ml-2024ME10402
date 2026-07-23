"""Inspect shipped model.joblib hyperparameters."""
import os
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
obj = joblib.load(os.path.join(ROOT, "model.joblib"))
print("keys:", list(obj.keys()) if isinstance(obj, dict) else type(obj))
if isinstance(obj, dict):
    for k, v in obj.items():
        if k == "model":
            print("model type:", type(v))
            print("model:", v)
            if hasattr(v, "named_steps"):
                for n, s in v.named_steps.items():
                    print(f"  step {n}: {s}")
                    if hasattr(s, "get_params"):
                        print("   params:", {kk: vv for kk, vv in s.get_params().items()
                                             if kk in ("n_estimators", "max_depth",
                                                       "min_samples_leaf", "class_weight",
                                                       "max_features", "random_state",
                                                       "bootstrap", "criterion")})
            elif hasattr(v, "get_params"):
                p = v.get_params()
                keep = ("n_estimators", "max_depth", "min_samples_leaf", "class_weight",
                        "max_features", "random_state", "bootstrap", "criterion")
                print("params:", {kk: p[kk] for kk in keep if kk in p})
        else:
            print(f"{k}: {v}")
