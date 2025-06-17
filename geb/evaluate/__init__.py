from .hydrology import Hydrology


class Evaluate(Hydrology):
    def __init__(self, model):
        self.model = model

    def run(self):
        self.evaluate_discharge_grid()

    @property
    def output_folder_evaluate(self):
        folder = self.model.output_folder / "evaluate"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
