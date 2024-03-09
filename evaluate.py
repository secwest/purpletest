from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bleurt import score as bleurt_score
from rouge_score import rouge_scorer

def evaluate_exact_match(predicted, actual):
    return {"exact_match": int(predicted == actual)}

def evaluate_list_match(predicted, actual_list):
    return {"list_match": int(predicted in actual_list)}

def evaluate_f1(predicted, actual, average='macro'):
    # Note: Ensure `predicted` and `actual` are properly prepared
    return {"f1_score": f1_score([actual], [predicted], average=average)}

def evaluate_semantic_similarity(predicted, actual):
    predicted_embedding = model.encode([predicted])[0]
    actual_embedding = model.encode([actual])[0]
    similarity = cosine_similarity([predicted_embedding], [actual_embedding])[0][0]
    return {"semantic_similarity": similarity}

def evaluate_bleu(reference, candidate):
    reference = [reference.split()]  # BLEU expects a list of reference translations
    candidate = candidate.split()
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
    return {"bleu_score": bleu_score}

def evaluate_bleurt(reference, candidate):
    checkpoint = "path/to/bleurt-checkpoint-XXXX"  # Specify your BLEURT checkpoint path
    scorer = bleurt_score.BleurtScorer(checkpoint)
    bleurt_score = scorer.score(references=[reference], candidates=[candidate])[0]
    return {"bleurt_score": bleurt_score}

def evaluate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)
    return rouge_scores

evaluators = {
    'exact_match': evaluate_exact_match,
    'list_match': evaluate_list_match,
    'f1_score': evaluate_f1,
    'semantic_similarity': evaluate_semantic_similarity,
    'bleu': evaluate_bleu,
    'rouge': evaluate_rouge,
    'bleurt': evaluate_bleurt,
}

def evaluate_response(predicted, actual, task_type, **kwargs):
    if task_type in evaluators:
        return evaluators[task_type](predicted, actual, **kwargs)
    else:
        raise ValueError(f"No evaluator available for task type '{task_type}'")

