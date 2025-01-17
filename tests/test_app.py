# tests/test_app.py
from src.main import lin_reg  # Exemple simple, testez une fonction ou comportement


def test_dummy():
    assert True
    
    
def test_linear_regression_coefs():
    # Vérifier que le modèle linéaire a été entraîné et possède des coefficients
    assert hasattr(lin_reg, 'coef_')


