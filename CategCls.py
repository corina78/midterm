class POSTagMapper:
    def __init__(self):
        self.index_to_pos = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10,
                             'CD': 11, 'DT': 12,
                             'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21,
                             'NNP': 22, 'NNPS': 23,
                             'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31,
                             'RBS': 32, 'RP': 33,
                             'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41,
                             'VBZ': 42, 'WDT': 43,
                             'WP': 44, 'WP$': 45, 'WRB': 46}

        # Invert this dictionary to look up POS tags based on their indices
        self.pos_to_index = {v: k for k, v in self.index_to_pos.items()}

    def indices_to_classes(self, indices):
        for index in indices:
            print("debugging")
            print(type(index), index)
        pos_tags = [self.pos_to_index[index] for index in indices]  # Convert the indices to POS tags
        classes = []

        for tag in pos_tags:
            if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'NN|SYM', 'PRP', 'PRP$']:
                classes.append('Noun')
            elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                classes.append('Verb')
            elif tag in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
                classes.append('Adjective/Adverb')
            else:
                classes.append('Others')

        return classes

    def add_categorized_pos_column(self, df, column_name="pos_tags"):
        df['categorized_pos'] = df[column_name].apply(self.indices_to_classes)
        return df

if __name__ == "__main__":

    from load_datasets import load_datasets

    train_df= load_datasets("train_data.csv")

    categorizer = POSTagMapper()
    updated_df = categorizer.add_categorized_pos_column(train_df)
