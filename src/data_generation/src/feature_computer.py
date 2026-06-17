from typing import Dict

import numpy as np


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
    def compute(cls, name: str, *args, **kwargs):
        if name not in cls._registry:
            raise NotImplementedError(f"Feature '{name}' is not implemented in FeatureComputer.")
        return cls._registry[name](*args, **kwargs)

    @staticmethod
    def _subhalo_shmr(**kwargs) -> float:
        """Compute stellar half mass radius for a single subhalo."""
        particles = kwargs.get('particles', None)

        if particles is None:
            raise ValueError('Particles cannot be None.')

        pos = particles.get('POS', None)
        mass = particles.get('MASS', None)

        if pos is None or mass is None or len(mass) == 0:
            return np.nan

        pos = np.asarray(pos)
        mass = np.asarray(mass)

        center = np.average(pos, weights=mass, axis=0)
        r = np.linalg.norm(pos - center, axis=1)
        sort_idx = np.argsort(r)
        cumulative_mass = np.cumsum(mass[sort_idx])
        half_mass = cumulative_mass[-1] / 2.0
        idx = np.clip(np.searchsorted(cumulative_mass, half_mass), 0, len(r) - 1)

        return r[sort_idx[idx]]


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