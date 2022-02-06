import time
import datetime
import torch
from torch.utils.data import TensorDataset, random_split
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def good_update_interval(total_iters, num_desired_updates):
    '''
    This function will try to pick an intelligent progress update interval
    based on the magnitude of the total iterations.

    Parameters:
      `total_iters` - The number of iterations in the for-loop.
      `num_desired_updates` - How many times we want to see an update over the
                              course of the for-loop.
    '''
    # Divide the total iterations by the desired number of updates. Most likely
    # this will be some ugly number.
    exact_interval = total_iters / num_desired_updates

    # The `round` function has the ability to round down a number to, e.g., the
    # nearest thousandth: round(exact_interval, -3)
    #
    # To determine the magnitude to round to, find the magnitude of the total,
    # and then go one magnitude below that.

    # Get the order of magnitude of the total.
    order_of_mag = len(str(total_iters)) - 1

    # Our update interval should be rounded to an order of magnitude smaller.
    round_mag = order_of_mag - 1

    # Round down and cast to an int.
    update_interval = int(round(exact_interval, -round_mag))

    # Don't allow the interval to be zero!
    if update_interval == 0:
        update_interval = 1

    return update_interval

def convert_dataset(args, dataset, tokenizer):
    labels_tr = []
    input_ids_tr = []
    attn_masks_tr = []
    max_len = 128
    for ex in dataset:
        # Retrieve the premise and hypothesis strings.
        premise = ex['premise'][args.language_code].numpy().decode("utf-8")
        hypothesis = ex['hypothesis']['translation'][args.language_index].numpy().decode("utf-8")

        # Convert sentence pairs to input IDs, with attention masks.
        encoded_dict = tokenizer.encode_plus(premise, hypothesis,
                                                  max_length=max_len,
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  return_tensors='pt')

        # Add this example to our lists.
        input_ids_tr.append(encoded_dict['input_ids'])
        attn_masks_tr.append(encoded_dict['attention_mask'])
        labels_tr.append(ex['label'].numpy())

    # Convert each Python list of Tensors into a 2D Tensor matrix.
    input_ids_tr = torch.cat(input_ids_tr, dim=0)
    attn_masks_tr = torch.cat(attn_masks_tr, dim=0)

    # Cast the labels list to a Tensor.
    labels_tr = torch.tensor(labels_tr)
    print('   DONE. {:,} examples.'.format(len(labels_tr)))
    dataset = TensorDataset(input_ids_tr, attn_masks_tr, labels_tr)
    return dataset

def convert_super_dataset(args, dataset, tokenizer):
    labels_tr = []
    input_ids_tr = []
    attn_masks_tr = []
    max_len = 128
    if(args.models == 'mroberta_base' or args.models == 'mroberta_cnn'):
        unk = "<unk>"
    else:
        unk = "[UNK]"
    unk_text = tokenizer.tokenize(unk)
    unk_id = tokenizer.convert_tokens_to_ids(unk_text)[0]
    for ex in dataset:
        # Retrieve the premise and hypothesis strings.
        premise = ex['premise'][args.language_code].numpy().decode("utf-8")
        hypothesis = ex['hypothesis']['translation'][args.language_index].numpy().decode("utf-8")

        # Convert sentence pairs to input IDs, with attention masks.
        encoded_dict = tokenizer.encode_plus(premise, hypothesis,
                                                  max_length=max_len,
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  return_tensors='pt')

        # Add this example to our lists.
        super_ids = encoded_dict['input_ids'].numpy()
        if unk_id not in super_ids:
            continue
        input_ids_tr.append(encoded_dict['input_ids'])
        attn_masks_tr.append(encoded_dict['attention_mask'])
        labels_tr.append(ex['label'].numpy())

    # Convert each Python list of Tensors into a 2D Tensor matrix.
    input_ids_tr = torch.cat(input_ids_tr, dim=0)
    attn_masks_tr = torch.cat(attn_masks_tr, dim=0)

    # Cast the labels list to a Tensor.
    labels_tr = torch.tensor(labels_tr)
    print('   DONE. {:,} examples.'.format(len(labels_tr)))
    dataset = TensorDataset(input_ids_tr, attn_masks_tr, labels_tr)
    return dataset