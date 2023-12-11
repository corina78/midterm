def padding(sequence, dataset, padding_value=0):

    # Calculate the length of each tokenized sentence
    dataset['sentence_length'] = dataset['tokens'].apply(len)
    # Find the maximum sentence length
    max_length = dataset['sentence_length'].max()

    return sequence + [padding_value] * (max_length - len(sequence))

