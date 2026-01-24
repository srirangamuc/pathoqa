import torch
import evaluate
import string
import re
from sentence_transformers import SentenceTransformer

class PathVQA_Evaluator:
    def __init__(self, device):
        print("Loading Metrics (SBERT, BLEU, ROUGE)")
        self.device = device
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")

    def normalize_text(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_metrics(self, predictions, ground_truths):
        clean_preds = [self.normalize_text(p) for p in predictions]
        clean_truths = [self.normalize_text(t) for t in ground_truths]

        binary_indices = [i for i, t in enumerate(clean_truths) if t in ['yes', 'no']]
        open_indices = [i for i, t in enumerate(clean_truths) if t not in ['yes', 'no']]
        
        binary_acc = 0.0
        if binary_indices:
             matches = sum([1 for i in binary_indices if clean_preds[i] == clean_truths[i]])
             binary_acc = matches / len(binary_indices)

        open_sbert = 0.0
        open_exact = 0.0
        open_bleu = 0.0
        open_rouge = 0.0
        
        if open_indices:
            op_p = [clean_preds[i] for i in open_indices]
            op_t = [clean_truths[i] for i in open_indices]
            
            # A. Exact Match (Open)
            # Count how many match exactly after normalization
            open_exact = sum([1 for p, t in zip(op_p, op_t) if p == t]) / len(op_p)

            # B. SBERT (Optimized O(N))
            # We use torch.nn.functional.cosine_similarity for paired score
            p_emb = self.sbert.encode(op_p, convert_to_tensor=True, show_progress_bar=False)
            t_emb = self.sbert.encode(op_t, convert_to_tensor=True, show_progress_bar=False)
            # Correct usage: Paired Cosine Similarity
            scores = torch.nn.functional.cosine_similarity(p_emb, t_emb)
            open_sbert = torch.mean(scores).item()

            # C. BLEU & ROUGE (Using Evaluator)
            # BLEU expects references as list of lists
            # We use max_order=1 (BLEU-1) because VQA answers are often very short (1-3 words)
            # Default BLEU-4 returns 0 for such short sequences.
            bleu_score = self.bleu_metric.compute(predictions=op_p, references=[[t] for t in op_t], max_order=1)
            open_bleu = bleu_score['bleu']
            
            rouge_score = self.rouge_metric.compute(predictions=op_p, references=op_t)
            open_rouge = rouge_score['rougeL']

        return {
            "overall_score": (binary_acc + open_sbert + open_exact) / 3,
            "binary_accuracy": binary_acc,
            "binary_count": len(binary_indices),
            "open_sbert": open_sbert,
            "open_exact_match": open_exact,
            "open_bleu": open_bleu,
            "open_rouge_l": open_rouge,
            "open_count": len(open_indices)
        }
