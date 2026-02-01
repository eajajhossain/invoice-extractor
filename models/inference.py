"""
Invoice Entity Extraction - Inference Script
Usage: python inference.py <invoice_text>
"""

import pickle
import numpy as np
from tensorflow import keras

# Load model and artifacts
model = keras.models.load_model('invoice_ner_model.h5')

with open('vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def extract_entities(text, max_length=200):
    """Extract entities from invoice text"""
    # Preprocess
    text = text.lower().strip()
    tokens = text.split()
    
    # Encode
    token_ids = [vocab.word2idx.get(token, 1) for token in tokens]
    
    # Pad
    if len(token_ids) < max_length:
        token_ids += [0] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]
    
    # Predict
    X = np.array([token_ids])
    predictions = model.predict(X, verbose=0)
    pred_labels = np.argmax(predictions[0], axis=-1)[:len(tokens)]
    
    # Extract entities
    entities = {}
    current_entity = None
    current_tokens = []
    
    for token, label_idx in zip(tokens, pred_labels):
        label = label_encoder.idx2label[label_idx]
        
        if label.startswith('B-'):
            if current_entity:
                entity_type = current_entity.split('-')[1]
                entities[entity_type] = ' '.join(current_tokens)
            current_entity = label
            current_tokens = [token]
        elif label.startswith('I-') and current_entity:
            current_tokens.append(token)
        else:
            if current_entity:
                entity_type = current_entity.split('-')[1]
                entities[entity_type] = ' '.join(current_tokens)
            current_entity = None
            current_tokens = []
    
    if current_entity:
        entity_type = current_entity.split('-')[1]
        entities[entity_type] = ' '.join(current_tokens)
    
    return entities

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py '<invoice_text>'")
        sys.exit(1)
    
    text = sys.argv[1]
    entities = extract_entities(text)
    
    print("\nExtracted Entities:")
    for entity_type, value in entities.items():
        print(f"  {entity_type}: {value}")
