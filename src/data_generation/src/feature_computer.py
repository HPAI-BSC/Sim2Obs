from typing import Dict


class FeatureComputer:
    """
    Registry and dispatcher for named feature-computation functions.

    Feature functions register themselves at import time via the
    `@FeatureComputer.register(name)` decorator (see
    ``src/data_generation/src/features_dataset/*.py``). ``catalog.fields.*``
    entries that declare a `needs` mapping reference these names, and
    `compute()` dispatches to the matching registered function. The
    features_dataset package is auto-imported below so every module in it
    registers itself as soon as `FeatureComputer` is imported, without
    callers needing to import each feature module individually.
    """
    # Registry of available feature computations
    _registry: Dict[str, callable] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a feature computation method."""

        def decorator(func):
            """Register `func` under `name` in the FeatureComputer registry and return it unchanged."""
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def compute(cls, name: str, **kwargs):
        """
        Dispatch to the feature-computation function registered under `name`.

        Parameters:
        -----------
        name : str
            Name the feature function was registered under via
            `@FeatureComputer.register(name)`.
        \*\*kwargs
            Forwarded as-is to the registered function.

        Returns:
        --------
        Any
            Whatever the registered function returns.

        Raises:
        -------
        NotImplementedError
            If no function is registered under `name`.
        """
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