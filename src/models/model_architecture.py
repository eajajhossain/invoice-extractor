from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed


def build_invoice_ner_model(
    vocab_size: int,
    num_labels: int,
    max_seq_length: int,
    embedding_dim: int,
    lstm_units: int,
):
    inputs = Input(
        shape=(max_seq_length,),
        dtype="int32",
        name="input_ids"
    )

    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding"
    )(inputs)

    x = LSTM(
        lstm_units,
        return_sequences=True,
        name="lstm"
    )(x)

    outputs = TimeDistributed(
        Dense(num_labels, activation="softmax"),
        name="classifier"
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
