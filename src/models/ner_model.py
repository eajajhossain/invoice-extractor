import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from src.data.preprocessor import preprocess_text
from src.models.model_architecture import build_invoice_ner_model
import src.utils.pickle_compat as pickle_compat


# ------------------------------------------------------------------
# Pickle compatibility for legacy training artifacts
# ------------------------------------------------------------------
sys.modules["scripts.test_inference"] = pickle_compat


class NERModel:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)

        # ----------------------------
        # Load config
        # ----------------------------
        with open(self.model_dir / "config.json", "r") as f:
            self.config = json.load(f)

        self.max_len: int = self.config["max_seq_length"]

        # ----------------------------
        # Load vocabulary
        # ----------------------------
        with open(self.model_dir / "vocabulary.pkl", "rb") as f:
            self.vocab = pickle.load(f)

        # ----------------------------
        # Load label encoder
        # ----------------------------
        with open(self.model_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        # ----------------------------
        # Build model architecture
        # ----------------------------
        self.model = build_invoice_ner_model(
            vocab_size=self.config["vocab_size"],
            num_labels=self.config["num_labels"],
            max_seq_length=self.max_len,
            embedding_dim=self.config["embedding_dim"],
            lstm_units=self.config["lstm_units"],
        )

        # ----------------------------
        # Load trained weights (best-effort)
        # ----------------------------
        self.model.load_weights(
            self.model_dir / "invoice_ner_model.h5",
            by_name=True,
            skip_mismatch=True,
        )

    # --------------------------------------------------------------
    # Public inference
    # --------------------------------------------------------------
    def predict(self, text: str):
        tokens = preprocess_text(text)
        encoded = self._encode(tokens)

        predictions = self.model.predict(encoded, verbose=0)[0]
        return self._decode(tokens, predictions)

    # --------------------------------------------------------------
    # Encode tokens to model input
    # --------------------------------------------------------------
    def _encode(self, tokens):
        token_to_id = self._get_token_to_id()

        unk_id = token_to_id.get("<UNK>", 1)
        sequence = [token_to_id.get(token, unk_id) for token in tokens]
        sequence = sequence[: self.max_len]

        padded = sequence + [0] * (self.max_len - len(sequence))
        return np.array([padded], dtype=np.int32)

    # --------------------------------------------------------------
    # Decode predictions to entities
    # --------------------------------------------------------------
    def _decode(self, tokens, preds):
        label_ids = preds.argmax(axis=-1)
        entities = []

        for token, label_id in zip(tokens, label_ids):
            label = self.label_encoder.inverse_transform([label_id])[0]
            if label != "O":
                entities.append(
                    {
                        "token": token,
                        "label": label,
                    }
                )

        return entities

    # --------------------------------------------------------------
    # Resolve vocabulary structure safely
    # --------------------------------------------------------------
    def _get_token_to_id(self) -> Dict[str, int]:
        """
        Supports multiple vocabulary formats:
        - dict
        - object with word2idx
        - object with stoi
        """
        if isinstance(self.vocab, dict):
            return self.vocab

        if hasattr(self.vocab, "word2idx"):
            return self.vocab.word2idx

        if hasattr(self.vocab, "stoi"):
            return self.vocab.stoi

        raise TypeError(
            "Unsupported vocabulary type. "
            "Expected dict or object with word2idx / stoi."
        )
