import pandas as pd
import os
import numpy as np
import ast
import re
def load_embeddings():

    """Argument: loads embeddings dataset
    Returns: a pandas dataframe"""

    file_path = "/home/corina/Documents/Math_Machine_Learning/midtermExam/"
    dataset =  "wv.csv"
    df = pd.read_csv(os.path.join(file_path, dataset))

    # convert dataframe to dictionary
    embeddings_dict = df.set_index('word')['vectors'].to_dict()

    def string_to_list(s):
        s = s.strip('[]')  # Remove the starting and ending square brackets
        return [float(item) for item in s.split()]

    # Convert string representations to actual lists
    for key, value in embeddings_dict.items():
        embeddings_dict[key] = string_to_list(value)

    return embeddings_dict

def create_embeddings_matrix(embeddings_index, word_index):
    """
    Creates the embeddings matrix.
    """
    EMBEDDING_DIM = 64
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM)) # 64 is the dimensionality of the embedding
    for word, index in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

def embed_sentence(sentence, embedding_matrix):
    """
    Given a tokenized sentence (list of integers), return the embedded version.
    """
    EMBEDDING_DIM = embedding_matrix.shape[1]
    embedded_sentence = np.zeros((len(sentence), EMBEDDING_DIM))
    for i, word_id in enumerate(sentence):
        # If the word_id is a padding token (assuming 0), skip embedding
        if word_id == 0:
            continue
        # If word_id exists in the embedding matrix, embed it
        if word_id < embedding_matrix.shape[0]:
            embedded_sentence[i] = embedding_matrix[word_id]
    return embedded_sentence


# To get the embedded version of all input sentences:
def embed_all_sentences(input_sentences, embedding_matrix, max_length_sentence):
    """
    Embed all input sentences.
    """
    num_sentences = len(input_sentences)
    EMBEDDING_DIM = embedding_matrix.shape[1]
    embedded_data = np.zeros((num_sentences, max_length_sentence, EMBEDDING_DIM))

    for i, sentence in enumerate(input_sentences):
        embedded_sentence = embed_sentence(sentence, embedding_matrix)
        embedded_data[i, :len(embedded_sentence)] = embedded_sentence

    return embedded_data



if __name__ == "__main__":

    data = load_embeddings()
    print(data)

