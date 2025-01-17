# tests/test_app.py
from src.main import lin_reg  


def test_dummy():
    assert True
    
    
def test_linear_regression_coefs():
    # Vérifier que le modèle linéaire a été entraîné et possède des coefficients
    assert hasattr(lin_reg, 'coef_')


