from src.models.ner_model import NERModel

def test_model_loads():
    model = NERModel()
    assert model is not None
