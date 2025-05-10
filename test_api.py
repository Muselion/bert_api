import pytest
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from app import predict  # nom du fichier sans l'extension .py

# Charger le modèle et le tokenizer dans un fixture pytest
@pytest.fixture(scope="module")
def setup_model():
    model_name = "Muselion/Test_BERT"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, bert_model

# Test de la fonction predict
def test_predict(setup_model):
    tokenizer, bert_model = setup_model
    
    # Exemple de texte à tester
    test_text = "Here is an example text mentioning Python and Linux"
    
    # Appel de la fonction
    result = predict(test_text)
    
    # Assertions (par exemple vérifier que les classes prédites contiennent 'python' et 'linux')
    assert "python" in result, f"Expected 'python' to be in {result}"
    assert "linux" in result, f"Expected 'linux' to be in {result}"

    # Vérifier que la sortie est bien une chaîne de caractères
    assert isinstance(result, str), f"Expected output to be of type str, but got {type(result)}"