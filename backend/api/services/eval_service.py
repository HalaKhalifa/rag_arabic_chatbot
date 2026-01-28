from ragchat.evaluation.evaluation import bleu, f1
from ragchat.data.utils import normalize_arabic_text

def evaluate_prediction(expected: str, predicted: str):
    """
    Compare generated prediction with expected answer using BLEU and F1.
    """
    expected = normalize_arabic_text(expected)
    predicted = normalize_arabic_text(predicted)

    return {
        "bleu": bleu(predicted, expected),
        "f1": f1(predicted, expected)
    }
