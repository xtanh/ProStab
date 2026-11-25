"""ProStab: protein stability prediction for single substitutions."""
__version__ = "0.1.0"


def __getattr__(name):
    if name == "Predictor":
        from .inference import Predictor
        return Predictor
    raise AttributeError(name)
