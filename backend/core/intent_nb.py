"""
Intent NB - מסווג כוונות נלמד מאפס (Naive Bayes על char n-grams)
ללא תלויות חיצונות - pure Python.

משלים את מסווג כללי-האצבע: כשהכללים לא בטוחים, המודל הנלמד מצביע.
יתרון: עמיד לשגיאות כתיב, סלנג חדש, וניסוחים שלא הוגדרו מראש.
"""
import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional

N_MIN, N_MAX = 3, 4  # char n-gram sizes (3+ = ראיות חזקות, פחות רעש מ-bigrams גנריים)
MIN_EVIDENCE = 3     # מינימום n-grams תומכים במנצח - בלי זה נמנעים מניחוש


def _ngrams(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    padded = f"^{text}$"
    grams = []
    for n in range(N_MIN, N_MAX + 1):
        for i in range(len(padded) - n + 1):
            grams.append(padded[i:i + n])
    return grams


class IntentNaiveBayes:
    """Naive Bayes גזמני ( multinomial ) על char n-grams עם החלקת Laplace"""

    def __init__(self):
        self.class_counts: Counter = Counter()          # מסמכים לכל intent
        self.feature_counts: Dict[str, Counter] = {}    # intent -> ngram -> count
        self.class_total_features: Counter = Counter()
        self.vocab: set = set()
        self.trained = False

    def fit(self, examples: Dict[str, List[str]]):
        """examples: intent -> [phrase, phrase, ...]"""
        self.__init__()
        for intent, phrases in examples.items():
            fc = self.feature_counts.setdefault(intent, Counter())
            for phrase in phrases:
                grams = _ngrams(phrase)
                if not grams:
                    continue
                self.class_counts[intent] += 1
                fc.update(grams)
                self.class_total_features[intent] += len(grams)
                self.vocab.update(grams)
        self.trained = sum(self.class_counts.values()) > 0

    def predict(self, text: str) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
        """מחזיר (intent, ביטחון 0-1, top3).

        מנו credit רק ל-n-grams שקיימים ב-vocab של האימון (evidence בלבד) -
        בלי זה המודל היה overconfident על טקסטים שהוא מעולם לא ראה.
        """
        if not self.trained:
            return None, 0.0, []
        all_grams = _ngrams(text)
        grams = [g for g in all_grams if g in self.vocab]
        if len(grams) < 2:  # מעט מדי ראיות - נמנעים מניחוש
            return None, 0.0, []

        total_docs = sum(self.class_counts.values())
        V = max(len(self.vocab), 1)
        log_scores: Dict[str, float] = {}
        for intent in self.class_counts:
            # prior
            logp = math.log(self.class_counts[intent] / total_docs)
            denom = self.class_total_features[intent] + V  # Laplace
            fc = self.feature_counts[intent]
            for g in grams:
                logp += math.log((fc.get(g, 0) + 1) / denom)
            log_scores[intent] = logp

        # softmax לביטחון יציב נומרית
        mx = max(log_scores.values())
        exps = {i: math.exp(s - mx) for i, s in log_scores.items()}
        Z = sum(exps.values())
        probs = sorted(((i, e / Z) for i, e in exps.items()), key=lambda x: x[1], reverse=True)
        best_intent, best_prob = probs[0]

        # דרישת ראיות: לפחות MIN_EVIDENCE n-grams שקיימים בפיצ'רים של המנצח
        winner_fc = self.feature_counts.get(best_intent, Counter())
        evidence = sum(1 for g in grams if winner_fc.get(g, 0) > 0)
        if evidence < MIN_EVIDENCE:
            return None, 0.0, probs[:3]

        return best_intent, best_prob, probs[:3]
