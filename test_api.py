import pytest
from app import predict

# Test unitaire pour la fonction predict
def test_predict():
    test_text = "Here is an example text mentioning Python and Linux"
    result = predict(test_text)
    assert isinstance(result, str)
    assert "python" in result
    assert "linux" in result
    assert len(result) > 0

# Test d'intégration pour la fonction predict
def test_integration_predict():
    test_text = "Here is an example text mentioning Python and Linux"
    result = predict(test_text)
    assert isinstance(result, str)
    assert "python" in result
    assert "linux" in result
    assert len(result) > 0