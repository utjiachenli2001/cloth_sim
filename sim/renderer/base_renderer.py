class BaseRenderer:
    def __init__(self):
        pass

    def initialize_resources(self, *args, **kwargs):
        raise NotImplementedError("initialize_resources method must be implemented by subclasses.")

    def render(self, *args, **kwargs):
        raise NotImplementedError("Render method must be implemented by subclasses.")
