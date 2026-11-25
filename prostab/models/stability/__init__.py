"""Stability models, with optional training base classes."""
def __getattr__(name):
    if name == "RegressionModel":
        from .regression import RegressionModel
        return RegressionModel
    raise AttributeError(name)
