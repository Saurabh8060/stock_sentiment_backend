import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_MODEL_ID_FALLBACK = os.getenv("MODEL_ID_FALLBACK", "ProsusAI/finbert")
LEGACY_MODEL_ID_FALLBACK = os.getenv("MODEL_ID_FALLBACK_SECONDARY", "srbh08/stock-sentiment-model")
DEFAULT_LABEL_ORDER = tuple(
    part.strip().lower()
    for part in os.getenv("LABEL_ORDER", "negative,neutral,positive").split(",")
    if part.strip()
)


def _canonical_label(raw: str) -> str:
    normalized = raw.strip().lower()
    alias_map = {
        "negative": "negative",
        "bearish": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "bullish": "positive",
    }
    return alias_map.get(normalized, normalized)


def _has_model_weights(model_path: str) -> bool:
    weight_files = (
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "flax_model.msgpack",
    )
    return any(os.path.exists(os.path.join(model_path, filename)) for filename in weight_files)


def _model_candidates() -> list[str]:
    candidates: list[str] = []
    explicit_model_id = os.getenv("MODEL_ID", "").strip()
    if explicit_model_id:
        candidates.append(explicit_model_id)
    if os.path.exists(MODEL_DIR) and _has_model_weights(MODEL_DIR):
        candidates.append(MODEL_DIR)
    candidates.append(DEFAULT_MODEL_ID_FALLBACK)
    candidates.append(LEGACY_MODEL_ID_FALLBACK)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped

class SentimentPredictor:
    def __init__(self):
        self.model_source = None
        self.tokenizer = None
        self.model = None
        load_errors: list[str] = []
        for model_source in _model_candidates():
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_source)
                model = AutoModelForSequenceClassification.from_pretrained(model_source)
                model.eval()
                self.model_source = model_source
                self.tokenizer = tokenizer
                self.model = model
                break
            except Exception as exc:  # pragma: no cover - exercised in runtime environments
                load_errors.append(f"{model_source}: {exc}")
        if self.tokenizer is None or self.model is None:
            joined_errors = "; ".join(load_errors) or "No model candidates were configured."
            raise RuntimeError(f"Unable to load sentiment model. Tried: {joined_errors}")

        self.labels = self._resolve_labels()
        self.bullish_patterns = [
            re.compile(r"\b(increased|raised|boosted|expanded)\s+(its\s+)?(position|stake|holding|holdings)\b", re.IGNORECASE),
            re.compile(r"\b(bought|acquired|added to|initiated)\s+(a\s+)?(new\s+)?(position|stake|holding|holdings)\b", re.IGNORECASE),
            re.compile(r"\b(beat|beats|beating)\s+(earnings|estimates|expectations)\b", re.IGNORECASE),
            re.compile(r"\b(raised|raises)\s+(guidance|outlook|forecast)\b", re.IGNORECASE),
            re.compile(r"\b(golden\s+cross|bullish\s+trendline|bullish\s+breakout|bullish\s+signal)\b", re.IGNORECASE),
            re.compile(r"\b(cross(?:ed|es|ing)?\s+(above|over)|break(?:s|ing)?\s+above|broke\s+above|move(?:d|s|ing)?\s+above)\b.*\b(moving average|trendline|resistance)\b", re.IGNORECASE),
            re.compile(r"\b(overtook|reclaimed)\s+(the\s+)?\d{1,3}(?:-day)?\s+moving average\b", re.IGNORECASE),
        ]
        self.bearish_patterns = [
            re.compile(r"\b(reduced|reduces|lowered|cut|cuts|trimmed)\s+(its\s+)?(position|stake|holding|holdings)\b", re.IGNORECASE),
            re.compile(r"\b(sold|sells|selling|offloaded|disposed of)\s+(its\s+)?(position|stake|shares|holding|holdings)\b", re.IGNORECASE),
            re.compile(r"\b(miss|misses|missed)\s+(earnings|estimates|expectations)\b", re.IGNORECASE),
            re.compile(r"\b(cut|cuts|lowered|lowers)\s+(guidance|outlook|forecast)\b", re.IGNORECASE),
            re.compile(r"\b(death\s+cross|bearish\s+trendline|bearish\s+breakdown|bearish\s+signal)\b", re.IGNORECASE),
            re.compile(r"\b(cross(?:ed|es|ing)?\s+below|break(?:s|ing)?\s+below|broke\s+below|move(?:d|s|ing)?\s+below|fell\s+below)\b.*\b(moving average|trendline|support)\b", re.IGNORECASE),
            re.compile(r"\b(lost|dropped\s+below)\s+(the\s+)?\d{1,3}(?:-day)?\s+moving average\b", re.IGNORECASE),
        ]

    def _resolve_labels(self) -> list[str]:
        num_labels = int(getattr(self.model.config, "num_labels", len(DEFAULT_LABEL_ORDER)))
        labels: list[str] = []
        id2label = getattr(self.model.config, "id2label", {}) or {}

        for idx in range(num_labels):
            candidate = id2label.get(idx, id2label.get(str(idx), ""))
            if candidate:
                candidate_str = str(candidate).lower()
                if not candidate_str.startswith("label_"):
                    labels.append(_canonical_label(candidate_str))
                    continue
            if idx < len(DEFAULT_LABEL_ORDER):
                labels.append(_canonical_label(DEFAULT_LABEL_ORDER[idx]))
            else:
                labels.append(f"label_{idx}")
        return labels

    def _rule_based_sentiment(self, text: str) -> tuple[str | None, str | None]:
        normalized = " ".join(text.split())
        for pattern in self.bearish_patterns:
            if pattern.search(normalized):
                return "negative", "finance_phrase_bearish"
        for pattern in self.bullish_patterns:
            if pattern.search(normalized):
                return "positive", "finance_phrase_bullish"
        return None, None

    def predict(self, text: str):
        if not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        conf, idx = torch.max(probs, dim=1)
        model_sentiment = self.labels[idx.item()]
        model_confidence = conf.item()
        override_sentiment, override_reason = self._rule_based_sentiment(text)
        final_sentiment = override_sentiment or model_sentiment
        final_confidence = model_confidence

        result = {
            "sentiment": final_sentiment,
            "confidence": round(final_confidence, 4)
        }

        if override_sentiment and override_sentiment != model_sentiment:
            result["model_sentiment"] = model_sentiment
            result["model_confidence"] = round(model_confidence, 4)
            result["override_reason"] = override_reason

        return result
