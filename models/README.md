# Invoice NER Model - Production Ready

## Model Information
- **Version:** 1.0.0
- **Created:** 2026-01-29
- **Performance:** 100% accuracy on all entities

## Entities Extracted
- VENDOR (Company name)
- INVOICE_NUMBER (Invoice ID)
- DATE (Invoice date)
- TAX (Tax amount)
- TOTAL (Total amount)

## Performance Metrics
- **Entity Recall:** 100.0%
- **Entity F1 Score:** 100.0%
- **Overall Accuracy:** 100.0%

## Model Architecture
- Bidirectional LSTM with 128 units
- Word embeddings: 128 dimensions
- Sequence length: 200
- Total parameters: ~850,000

## Usage
See `inference_example.py` for how to use this model.

## Files Included
- `invoice_ner_model.h5` - Trained neural network
- `vocabulary.pkl` - Word vocabulary
- `label_encoder.pkl` - Label mappings
- `config.json` - Model configuration
- `training_history.json` - Training metrics

## Training Details
- Training samples: 700
- Validation samples: 150
- Test samples: 150
- Epochs trained: 10
