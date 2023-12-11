def create_padding_mask(padded_classes_list):
    return [1 if tag != 0 else 0 for tag in padded_classes_list]

