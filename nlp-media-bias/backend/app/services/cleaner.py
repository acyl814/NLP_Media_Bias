import re

def clean_text(text: str) -> str:
    """
    Nettoie le texte brut d'un article de presse.
    """

    # Supprimer copyright AP / Reuters / etc.
    text = re.sub(r'Copyright.*?Reserved\.', ' ', text, flags=re.IGNORECASE)

    # Supprimer crédits photo (AP Photo/...)
    text = re.sub(r'\(AP Photo.*?\)', ' ', text)
    text = re.sub(r'\(Reuters.*?\)', ' ', text)

    # Supprimer dates et légendes photo isolées
    text = re.sub(r'\b(AP Photo|Reuters|Getty Images).*?\)', ' ', text)

    # Corriger mots collés (ex: warfatigue -> war fatigue)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Supprimer caractères non utiles
    text = re.sub(r'[^a-zA-Z0-9\s\.,;:\'\"!?()-]', ' ', text)

    # Normaliser espaces
    text = re.sub(r'\s+', ' ', text)

    # Minuscule + trim
    text = text.lower().strip()

    return text
