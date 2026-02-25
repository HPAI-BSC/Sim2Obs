import importlib
import pkgutil
import os

# Auto-import all modules in the features/ folder
# This triggers the @FeatureComputer.register decorators
package_dir = os.path.dirname(__file__)

for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    importlib.import_module(f'src.data_generation.src.features_dataset.{module_name}')