import numpy as np


class BaseController:

    def __init__(self):
        pass

    def initialize_resources(self, env) -> None:
        raise NotImplementedError("initialize_resources method must be implemented by subclasses.")

    def compute(self, env, target: np.ndarray) -> None:
        raise NotImplementedError("compute method must be implemented by subclasses.")
