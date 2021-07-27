import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_decoder_hidden_state(dec_hidden_state_):
    plt.figure(figsize=(2.5, 4.5))
    sns.heatmap(np.transpose(np.matrix(dec_hidden_state_)), annot=True, cmap=sns.light_palette("purple", as_cmap=True),
                linewidths=1)
    plt.show()


def visualize_encoder_hidden_state(annotation_):
    plt.figure(figsize=(2.5, 4.5))
    sns.heatmap(np.transpose(np.matrix(annotation_)), annot=True, cmap=sns.light_palette("orange", as_cmap=True),
                linewidths=1)
    plt.show()


def visualize_encoder_hidden_states(annotations_):
    """each column is an annotation"""
    sns.heatmap(annotations_, annot=True, cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)
    plt.show()


def single_dot_attention_score(dec_hidden_state_, enc_hidden_state_):
    return np.dot(dec_hidden_state_, enc_hidden_state_)


def dot_attention_score(dec_hidden_state_, annotations_):
    return np.matmul(np.transpose(dec_hidden_state_), annotations_)


def apply_attention_scores(attention_weights_, annotations_):
    # multiply each annotation by its score to proceed closer to the attention context vector
    return attention_weights_ * annotations_


def calculate_attention_vector(applied_attention_):
    return np.sum(applied_attention_, axis=1)


def softmax(x):
    x = np.array(x, dtype=np.float128)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    dec_hidden_state = [5, 1, 20]
    # visualize_decoder_hidden_state(dec_hidden_state)

    annotation = [3, 12, 45]  # encoder hidden state
    # visualize_encoder_hidden_state(annotation)

    print(single_dot_attention_score(dec_hidden_state, annotation))

    annotations = np.transpose([[3, 12, 45], [59, 2, 5], [1, 43, 5], [4, 3, 45.3]])
    # visualize_encoder_hidden_states(annotations)

    attention_weights_raw = dot_attention_score(dec_hidden_state, annotations)
    print(attention_weights_raw)

    attention_weights = softmax(attention_weights_raw)
    print(attention_weights)

    applied_attention = apply_attention_scores(attention_weights, annotations)
    print(applied_attention)
    # visualize_encoder_hidden_states(applied_attention)

    attention_vector = calculate_attention_vector(applied_attention)
    print(attention_vector)
    visualize_decoder_hidden_state(attention_vector)
    """
    concatenate it with the hidden state and pass it through a hidden layer
    to produce the the result of this decoding time step
    """
