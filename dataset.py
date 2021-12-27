import torch
import config

columns = ['subject', 'predicate', 'object', 'label']
special_chars = [',' , '.', '+', '-', '[', ']', '/', '\\']

def unify_text(text):
    # remove special chars at beginning
    ntext = text.strip()
    while ntext[0] in special_chars:
        ntext = ntext[1:].strip()

    # remove special chats at ending
    while ntext[-1] in special_chars:
        ntext = ntext[:-1].strip()

    ntext = ' '.join(ntext.split())
    return ntext


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()

    return tokens_a, tokens_b, tokens_c


def convert_to_features(dataframe, label_list):
    """Loads a data file into a list of `InputBatch`s."""
    tokenizer = config.TOKENIZER
    max_seq_length = config.MAX_LEN

    dataframe.columns = columns
    label_map = {label : i for i, label in enumerate(label_list)}

    ids_set = []
    token_type_ids_set = []
    mask_set = []
    label_set = []


    for line in dataframe.iloc:
        subject = unify_text(line[columns[0]])
        predicate = unify_text(line[columns[1]])
        object = unify_text(line[columns[2]])
        label = unify_text(line[columns[3]])

        tokens_s = tokenizer.tokenize(subject)
        tokens_p = tokenizer.tokenize(predicate)
        tokens_o = tokenizer.tokenize(object)

        # reduce the characters to match the maxlen
        tokens_s, tokens_p, tokens_o = _truncate_seq_triple(tokens_s, tokens_p, tokens_o, max_seq_length - 4)

        tokens = ["[CLS]"] + tokens_s + ["[SEP]"] + tokens_p + ["[SEP]"] + tokens_o + ["[SEP]"]
        token_type_ids = [0] * (len(tokens_s) + 2)
        token_type_ids += [0] * (len(tokens_p) + 1)
        token_type_ids += [0] * (len(tokens_o) + 1)        

        ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        mask = [1] * len(ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(ids))
        ids += padding
        mask += padding
        token_type_ids += padding

        # check in len of input is correct. if not, alert it
        assert len(ids) == max_seq_length
        assert len(mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        label_id = label_map[label]

        ids_set.append(ids)
        token_type_ids_set.append(token_type_ids)
        mask_set.append(mask)
        label_set.append(label_id)

    return {
        "ids": torch.tensor(ids_set, dtype=torch.long),
        "mask": torch.tensor(mask_set, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids_set, dtype=torch.long),
        "label": torch.tensor(label_set, dtype=torch.long),
    }

    