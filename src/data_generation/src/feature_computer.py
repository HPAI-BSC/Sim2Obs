from typing import Dict


class FeatureComputer:
    # Registry of available feature computations
    _registry: Dict[str, callable] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a feature computation method."""

        def decorator(func):
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def compute(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise NotImplementedError(f"Feature '{name}' is not implemented in FeatureComputer.")
        return cls._registry[name](**kwargs)


# --- Auto-discover and register all features at import time ---
import importlib
import pkgutil
import os

_features_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features_dataset')
print(f"Loading features from: {_features_dir}")  # remove after confirming

if os.path.exists(_features_dir):
    for _, module_name, _ in pkgutil.iter_modules([_features_dir]):
        importlib.import_module(f'src.data_generation.src.features_dataset.{module_name}')
else:
    print(f"WARNING: features directory not found at {_features_dir}")