from importlib.metadata import PackageNotFoundError, version

from .mta import MTA, MTAConfig

try:
    __version__ = version("mta")
except PackageNotFoundError:
    # Package is not installed (e.g. running from source tree)
    __version__ = "0.0.0"

__all__ = ["MTA", "MTAConfig", "__version__"]
