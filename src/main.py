import tensorflow as tf
import tensorflow_datasets as tfds
import textwrap
import os
import  random
import pandas as pd
import csv
import torch
import argparse
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from utils import *
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from models import SeqClassification, mroberta_cnn, mbert_base, mbert_cnn, cnn_base
from create_img import create_samples
# Modify XLMR
import cv2
import jieba
from transformers import BertTokenizer, BertModel

#args
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=2, help="gpu")
parser.add_argument("--language_code", type=str, default='zh', help="language code string")
parser.add_argument("--language_index", type=int, default=14, help="language index")
parser.add_argument("--epochs", type=int, default=30, help="learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
parser.add_argument("--eps", type=float, default=1e-8, help="adam_epsilon")
parser.add_argument("--seed", type=int, default=0, help="adam_epsilon")
parser.add_argument("--num_labels", type=int, default=3, help="num labels")
parser.add_argument("--cnn", type=bool, default=False, help="add cnn or not")
parser.add_argument("--models", type=str, default='mroberta_base', help="type of PLMs")
parser.add_argument("--alpha", type=int, default=4, help="type of lr of CNN part")
parser.add_argument("--mode", type=str, default='word', help="type of spilit")
args = parser.parse_args()

#set seed
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

# set device
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu)
    device = args.gpu
#device = torch.device('cpu')

# set language
language_code = args.language_code
language_index = args.language_index

#set Model
if args.models == 'mroberta_base':
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base" )
    plm_model = SeqClassification(args, 768, 3).to(device)
    args.cnn = False
if args.models == 'cnn':
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    plm_model = cnn_base(args, 768, 3).to(device)
    args.cnn = True
if args.models == 'mroberta_cnn':
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    plm_model = mroberta_cnn(args, 768, 3).to(device)
    args.cnn = True
if args.models == 'mbert_base':
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    plm_model = mbert_base(args, 768, 3).to(device)
    args.cnn = False
if args.models == 'mbert_cnn':
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    plm_model = mbert_cnn(args, 768, 3).to(device)
    args.cnn = True
#dataset
xnli_train_dataset = tfds.load(name='xnli', split="test")
xnli_test_dataset = tfds.load(name='xnli', split="validation")
train_dataset = convert_dataset(args, xnli_train_dataset, tokenizer)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
test_dataset = convert_dataset(args, xnli_test_dataset, tokenizer)
super_test_dataset = convert_super_dataset(args, xnli_test_dataset, tokenizer)
print("test size")
print(len(test_dataset))
print("super test size")
print(len(super_test_dataset))


#dataloader
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = args.batch_size # Trains with this batch size.
        )
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = args.batch_size # Evaluate with this batch size.
        )

test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = args.batch_size # Evaluate with this batch size.
        )

super_test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(super_test_dataset), # Pull out batches sequentially.
            batch_size = args.batch_size # Evaluate with this batch size.
        )

# optimizer
if args.cnn:
    optimizer = AdamW(plm_model.get_parameter(args.lr),
                      lr = args.lr, # args.learning_rate
                      eps = args.eps # args.adam_epsilon  - default is 1e-8.
                    )
if not args.cnn:
    optimizer = AdamW(plm_model.parameters(),
                      lr=args.lr,  # args.learning_rate
                      eps=args.eps  # args.adam_epsilon  - default is 1e-8.
                      )

if args.mode == 'character':
    maxlen = 400
else:
    maxlen = 400

# Number of training epochs.
epochs = args.epochs

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# training
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    plm_model.train()

    # Pick an interval on which to print progress updates.
    update_interval = good_update_interval(
        total_iters=len(train_dataloader),
        num_desired_updates=10
    )

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update.
        if (step % update_interval) == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        plm_model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This call returns the loss (because we provided labels) and the
        # "logits"--the model outputs prior to activation.

        if args.cnn:
            #convert ids to text
            input_text = tokenizer.batch_decode(b_input_ids, skip_special_tokens = True)
            #去掉s 去掉 pad
            if args.mode == 'character':
                input_images, seq_len = create_samples(input_text, is_word=True, maxlen=maxlen)
            else:
                input_images, seq_len = create_samples(input_text, is_word = False, maxlen=maxlen)
            input_images = np.array(input_images)
            input_images = np.reshape(input_images, (-1, 36, 36, 3))
            input_images = input_images.transpose(0, 3, 1, 2)
            input_images = torch.from_numpy(input_images).to(device)
            input_images = input_images.float()
            outputs = plm_model(input_images,
                                b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels,
                                seq_len=seq_len,
                                maxlen=maxlen)




        if not args.cnn:
            outputs = plm_model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
        loss, logits = outputs
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(plm_model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    plm_model.eval()

    # Tracking variables
    total_eval_loss = 0

    predictions, true_labels = [], []

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            if args.cnn:
                # convert ids to text
                input_text = tokenizer.batch_decode(b_input_ids, skip_special_tokens=True)
                # 去掉s 去掉 pad
                if args.mode == 'character':
                    input_images, seq_len = create_samples(input_text, is_word=True, maxlen=maxlen)
                else:
                    input_images, seq_len = create_samples(input_text, is_word=False, maxlen=maxlen)
                input_images = np.array(input_images)
                input_images = np.reshape(input_images, (-1, 36, 36, 3))
                input_images = input_images.transpose(0, 3, 1, 2)
                input_images = torch.from_numpy(input_images).to(device)
                input_images = input_images.float()
                outputs = plm_model(input_images,
                                    b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    seq_len=seq_len,
                                    maxlen = maxlen)

            if not args.cnn:
                outputs = plm_model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            loss, logits = outputs

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Calculate the accuracy for this batch of test sentences.

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # Measure validation accuracy...

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # For each sample, pick the label (0, 1, or 2) with the highest score.
    predicted_labels = np.argmax(flat_predictions, axis=1).flatten()

    # Calculate the validation accuracy.
    val_accuracy = (predicted_labels == flat_true_labels).mean()

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    path_file_name = '/home/zijian/Probing_Font4Commonsense/src/' + args.models + '_epoch_result.txt'
    if not os.path.exists(path_file_name):
        fileObject = open('/home/zijian/Probing_Font4Commonsense/src/' + args.models + args.language_code + '_epoch_result.txt', 'a+', encoding='utf-8')
        fileObject.write(
            "alpha" + str(args.alpha) + "lr " + str(args.lr) + ' val ' +args.language_code + ' ' + args.models + ' ' + str(epoch_i + 1) + ' loss ' + str(avg_val_loss) + ' ' + str(args.mode) + ' ' + str(val_accuracy))
        fileObject.write('\n')
        fileObject.close()
    else:
        fileObject = open('/home/zijian/Probing_Font4Commonsense/src/' + args.models + args.language_code + '_epoch_result.txt', 'a+', encoding='utf-8')
        fileObject.write(
            "alpha" + str(args.alpha) + "lr " + str(args.lr) + ' val ' + args.language_code + ' ' + args.models + ' ' + str(epoch_i + 1) + ' loss ' + str(avg_val_loss) + ' ' + str(args.mode) + ' ' + str(val_accuracy))
        fileObject.write('\n')
        fileObject.close()

    # Evaluate data for one epoch
    print("Running Test for one epoch...")
    t0 = time.time()
    plm_model.eval()
    # Tracking variables
    total_test_loss = 0
    predictions, true_labels = [], []
    for batch in test_dataloader:
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            if args.cnn:
                # convert ids to text
                input_text = tokenizer.batch_decode(b_input_ids, skip_special_tokens=True)
                # 去掉s 去掉 pad
                if args.mode == 'character':
                    input_images, seq_len = create_samples(input_text, is_word=True, maxlen=maxlen)
                else:
                    input_images, seq_len = create_samples(input_text, is_word=False, maxlen=maxlen)
                input_images = np.array(input_images)
                input_images = np.reshape(input_images, (-1, 36, 36, 3))
                input_images = input_images.transpose(0, 3, 1, 2)
                input_images = torch.from_numpy(input_images).to(device)
                input_images = input_images.float()
                outputs = plm_model(input_images,
                                    b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    seq_len=seq_len,
                                    maxlen=maxlen)

            if not args.cnn:
                outputs = plm_model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            loss, logits = outputs

        # Accumulate the validation loss.
        total_test_loss += loss.item()

        # Calculate the accuracy for this batch of test sentences.

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # Measure validation accuracy...

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # For each sample, pick the label (0, 1, or 2) with the highest score.
    predicted_labels = np.argmax(flat_predictions, axis=1).flatten()

    # Calculate the validation accuracy.
    val_accuracy = (predicted_labels == flat_true_labels).mean()

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(test_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Test Loss: {0:.2f}".format(avg_val_loss))
    print("  Test took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_test_loss,
            'Valid. Accur.': val_accuracy,
            'Training Time': training_time,
            'Test Time': validation_time
        }
    )
    path_file_name = '/home/zijian/Probing_Font4Commonsense/src/' + args.models + '_epoch_result.txt'
    if not os.path.exists(path_file_name):
        fileObject = open('/home/zijian/Probing_Font4Commonsense/src/' + args.models + args.language_code + '_epoch_result.txt', 'a+', encoding='utf-8')
        fileObject.write(
            "alpha" + str(args.alpha) + "lr " + str(args.lr) + ' test ' + args.language_code + ' ' + args.models + ' ' + str(epoch_i + 1) + ' loss ' + str(avg_test_loss) + ' ' + str(args.mode) + ' ' + str(val_accuracy))
        fileObject.write('\n')
        fileObject.close()
    else:
        fileObject = open('/home/zijian/Probing_Font4Commonsense/src/' + args.models + args.language_code + '_epoch_result.txt', 'a+', encoding='utf-8')
        fileObject.write(
            "alpha" + str(args.alpha) + "lr " + str(args.lr) + 'test ' + args.language_code + ' ' + args.models + ' ' + str(epoch_i + 1) + ' loss ' + str(avg_test_loss) + ' ' + str(args.mode) + ' ' + str(val_accuracy))
        fileObject.write('\n')
        fileObject.close()
    # Super Test Evaluate data for one epoch
    print("Running Super Test for one epoch...")
    t0 = time.time()
    plm_model.eval()
    # Tracking variables
    total_test_loss = 0
    predictions, true_labels = [], []
    for batch in super_test_dataloader:
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            if args.cnn:
                # convert ids to text
                input_text = tokenizer.batch_decode(b_input_ids, skip_special_tokens=True)
                # 去掉s 去掉 pad
                if args.mode == 'character':
                    input_images, seq_len = create_samples(input_text, is_word=True, maxlen=maxlen)
                else:
                    input_images, seq_len = create_samples(input_text, is_word=False, maxlen=maxlen)
                input_images = np.array(input_images)
                input_images = np.reshape(input_images, (-1, 36, 36, 3))
                input_images = input_images.transpose(0, 3, 1, 2)
                input_images = torch.from_numpy(input_images).to(device)
                input_images = input_images.float()
                outputs = plm_model(input_images,
                                    b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    seq_len=seq_len,
                                    maxlen=maxlen)

            if not args.cnn:
                outputs = plm_model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            loss, logits = outputs

        # Accumulate the validation loss.
        total_test_loss += loss.item()

        # Calculate the accuracy for this batch of test sentences.

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # Measure validation accuracy...

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # For each sample, pick the label (0, 1, or 2) with the highest score.
    predicted_labels = np.argmax(flat_predictions, axis=1).flatten()

    # Calculate the validation accuracy.
    val_accuracy = (predicted_labels == flat_true_labels).mean()

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(test_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Test Loss: {0:.2f}".format(avg_val_loss))
    print("  Test took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_test_loss,
            'Valid. Accur.': val_accuracy,
            'Training Time': training_time,
            'Test Time': validation_time
        }
    )
    path_file_name = '/home/zijian/Probing_Font4Commonsense/src/' + args.models + '_epoch_result.txt'
    if not os.path.exists(path_file_name):
        fileObject = open('/home/zijian/Probing_Font4Commonsense/src/' + args.models + args.language_code + '_epoch_result.txt', 'a+', encoding='utf-8')
        fileObject.write(
            "alpha" + str(args.alpha) + "lr " + str(args.lr) + ' supertest ' + args.language_code + ' ' + args.models + ' ' + str(epoch_i + 1) + ' loss ' + str(avg_test_loss) + ' ' + str(args.mode) + ' ' + str(val_accuracy))
        fileObject.write('\n')
        fileObject.close()
    else:
        fileObject = open('/home/zijian/Probing_Font4Commonsense/src/' + args.models + args.language_code + '_epoch_result.txt', 'a+', encoding='utf-8')
        fileObject.write(
            "alpha" + str(args.alpha) + "lr " + str(args.lr) + 'supertest ' + args.language_code + ' ' + args.models + ' ' + str(epoch_i + 1) + ' loss ' + str(avg_test_loss) + ' ' + str(args.mode) + ' ' + str(val_accuracy))
        fileObject.write('\n')
        fileObject.close()

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

#
# print("")
# print("Running Test...")
#
# t0 = time.time()
#
# # Put the model in evaluation mode--the dropout layers behave differently
# # during evaluation.
# plm_model.eval()
#
# # Tracking variables
# total_eval_loss = 0
#
# predictions, true_labels = [], []
#
# # Evaluate data for one epoch
# for batch in test_dataloader:
#     # Unpack this training batch from our dataloader.
#     #
#     # As we unpack the batch, we'll also copy each tensor to the GPU using
#     # the `to` method.
#     #
#     # `batch` contains three pytorch tensors:
#     #   [0]: input ids
#     #   [1]: attention masks
#     #   [2]: labels
#     b_input_ids = batch[0].to(device)
#     b_input_mask = batch[1].to(device)
#     b_labels = batch[2].to(device)
#
#     # Tell pytorch not to bother with constructing the compute graph during
#     # the forward pass, since this is only needed for backprop (training).
#     with torch.no_grad():
#         # Forward pass, calculate logit predictions.
#         # token_type_ids is the same as the "segment ids", which
#         # differentiates sentence 1 and 2 in 2-sentence tasks.
#         # The documentation for this `model` function is here:
#         # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
#         # Get the "logits" output by the model. The "logits" are the output
#         # values prior to applying an activation function like the softmax.
#         if args.cnn:
#             # convert ids to text
#             input_text = tokenizer.batch_decode(b_input_ids, skip_special_tokens=True)
#             # 去掉s 去掉 pad
#             if args.mode == 'character':
#                 input_images, seq_len = create_samples(input_text, is_word=True, maxlen=maxlen)
#             else:
#                 input_images, seq_len = create_samples(input_text, is_word=False, maxlen=maxlen)
#             input_images = np.array(input_images)
#             input_images = np.reshape(input_images, (-1, 36, 36, 3))
#             input_images = input_images.transpose(0, 3, 1, 2)
#             input_images = torch.from_numpy(input_images).to(device)
#             input_images = input_images.float()
#             outputs = plm_model(input_images,
#                                 b_input_ids,
#                                 token_type_ids=None,
#                                 attention_mask=b_input_mask,
#                                 labels=b_labels,
#                                 seq_len=seq_len,
#                                 maxlen=maxlen)
#
#         if not args.cnn:
#             outputs = plm_model(b_input_ids,
#                                 token_type_ids=None,
#                                 attention_mask=b_input_mask,
#                                 labels=b_labels)
#         loss, logits = outputs
#
#     # Accumulate the validation loss.
#     total_eval_loss += loss.item()
#
#     # Calculate the accuracy for this batch of test sentences.
#
#     # Move logits and labels to CPU
#     logits = logits.detach().cpu().numpy()
#     label_ids = b_labels.to('cpu').numpy()
#
#     # Store predictions and true labels
#     predictions.append(logits)
#     true_labels.append(label_ids)
#
# # Measure validation accuracy...
#
# # Combine the results across all batches.
# flat_predictions = np.concatenate(predictions, axis=0)
# flat_true_labels = np.concatenate(true_labels, axis=0)
#
# # For each sample, pick the label (0, 1, or 2) with the highest score.
# predicted_labels = np.argmax(flat_predictions, axis=1).flatten()
#
# # Calculate the validation accuracy.
# val_accuracy = (predicted_labels == flat_true_labels).mean()
#
# # Report the final accuracy for this validation run.
# print("  Accuracy: {0:.2f}".format(val_accuracy))
#
# # Calculate the average loss over all of the batches.
# avg_val_loss = total_eval_loss / len(validation_dataloader)
#
# # Measure how long the validation run took.
# validation_time = format_time(time.time() - t0)
#
# print("  Test Loss: {0:.2f}".format(avg_val_loss))
# print("  Validation took: {:}".format(validation_time))
#
#
# print("")
# print("Testing complete!")
#
# # print("Total testing took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
#
# path_file_name = '/home/zijian/Probing_Font4Commonsense/src/result.txt'
# if not os.path.exists(path_file_name):
#     fileObject = open('/home/zijian/Probing_Font4Commonsense/src/result.txt', 'w', encoding='utf-8')
#     fileObject.write(args.language_code + ' ' + args.models + ' ' + str(args.epochs) + ' ' + str(args.mode) +  ' ' + str(val_accuracy))
#     fileObject.write('\n')
#     fileObject.close()
# else:
#     fileObject = open('/home/zijian/Probing_Font4Commonsense/src/result.txt', 'a+', encoding='utf-8')
#     fileObject.write(args.language_code + ' ' + args.models + ' ' + str(args.epochs) + ' ' + str(val_accuracy))
#     fileObject.write('\n')
#     fileObject.close()