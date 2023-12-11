# preprocess steps for transformer:
from LoadData import load_datasets
from Tokenize import tokenize_sentences
from CategCls import POSTagMapper
from Padding import padding
from Masking import create_padding_mask
from PosEndode import positional_encoding
import numpy as np
from scipy.io import savemat

# 1. Load the data
train_df = load_datasets("train_data.csv")
train_df_dict = train_df.to_dict("list")
#savemat("test_data.mat", test_df_dict)

# 2. Tokenize the sentences
train_df['tokenized'], word_index = tokenize_sentences(train_df['tokens'])
print("number of words in the training set", len(word_index))

# 3. Categorize the POS tags
# Create an instance of the POSTagMapper class
pos_tag_mapper = POSTagMapper()
# Use the add_categorized_pos_column method directly without apply
updated_train_df = pos_tag_mapper.add_categorized_pos_column(train_df, column_name="pos_tags")

# 4. Pad the sequences
train_df['padded_tokenized'] = train_df['tokenized'].apply(padding, args=(train_df, 0))
max_sentence_length = train_df['tokenized'].apply(len).max()
print("The maximum sentence length is:", max_sentence_length)

# 5. Pad the classes
train_df['padded_classes'] = train_df['categorized_pos'].apply(padding, args=(train_df, 0))

# 6. Create the mask for the classes and the tokens
train_df['mask_classes'] = train_df['padded_classes'].apply(create_padding_mask)
train_df['mask_tokens'] = train_df['padded_tokenized'].apply(create_padding_mask)

# 7. Create the positional encoding
pos_encoding = positional_encoding(max_sentence_length, d=64)

# 8. Load the embedding matrix4
from load_and_create_embeddings import load_embeddings
embeddings_index = load_embeddings()

# 9. Create the embeddings matrix:
from load_and_create_embeddings import create_embeddings_matrix
embedding_matrix = create_embeddings_matrix(embeddings_index, word_index)
print(embedding_matrix.shape)

# 10. Embed the sentences

from load_and_create_embeddings import embed_all_sentences

embedded_data = embed_all_sentences(train_df["padded_tokenized"], embedding_matrix, max_length_sentence= 113)
print(embedded_data.shape)
print("reached the end of the preprocessing step")

## Add the positional encoding to the embedded_data

final_embedding = embedded_data * 1.0 + pos_encoding[:,:,:] * 1.0
savemat("final_embeddings_test.mat", final_embedding)
print(final_embedding.shape)




