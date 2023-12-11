import pandas as pd
def tokenize_sentences(df_column):

    # Flatten the list of tokens and identify unique words
    all_tokens = [word for sublist in df_column for word in sublist]
    unique_tokens = set(all_tokens)

    # Create a dictionary that maps each word to a unique integer
    word_to_int = {word: idx for idx, word in enumerate(unique_tokens)}

    # Convert each sentence (list of tokens) to a list of integers
    tokenized_sentences = df_column.apply(lambda x: [word_to_int[word] for word in x])

    return tokenized_sentences, word_to_int

if __name__ == "__main__":

    from load_datasets import load_datasets

    train_df = load_datasets("train_data.csv")

    train_df['tokenized'], word_index = tokenize_sentences(train_df['tokens'])
    print(train_df)

