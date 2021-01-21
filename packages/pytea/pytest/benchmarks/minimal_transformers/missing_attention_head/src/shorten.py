import torch

import transformers

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np

import argparse


parser = argparse.ArgumentParser(description="BERT with Naver Movie Ratings")
parser.add_argument(
    "--device", default="auto", help="device type that will be used while training"
)
parser.add_argument("--max_len", default=128, help="max length for BERT sequence input")
parser.add_argument("--batch_size", default=16, help="batch size for training BERT")
parser.add_argument("--lr", default=2e-5, help="learning rate for AdamW")
parser.add_argument("--epochs", default=1, help="training epochs")

args = parser.parse_args()


### Device settings
if args.device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("No GPU available, using the CPU instead.")
else:
    device = args.device


### Data fetching
print("Fetching the training data")
'''
train = pd.read_csv("nsmc/ratings_train.txt", sep="\t")

sentences = train["document"]
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]


### Input labels and tokenizers
labels = train["label"].values

print("Load tokenizer")
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased", do_lower_case=False
)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# make all input sequence lengths to be args.max_len
# maybe there should be a builtin function that supports this feature..
for i in range(len(input_ids)):
    input_ids[i] = input_ids[i][: args.max_len]
    if len(input_ids[i]) < args.max_len:
        input_ids[i] += [0] * (args.max_len - len(input_ids[i]))

input_ids = np.array(input_ids, dtype=np.long)  ## (data_size, args.max_len)
'''

# Without tokenizer preprocessing..
input_ids = torch.randint(low=0, high=10000, size=(150, 128)).numpy()
labels = torch.randint(low=0, high=2, size=(150,)).numpy()

### Attention masks
'''
attention_masks = []

for seq in input_ids:
    seq_mask = (seq > 0).astype(np.float)
    attention_masks.append(seq_mask)
'''
attention_masks = torch.rand(150, 128).numpy()


### Dataset split into training & validation
print("Preparing datasets")

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids, labels, random_state=2018, test_size=0.1
)

train_masks, validation_masks, _, _ = train_test_split(
    attention_masks, input_ids, random_state=2018, test_size=0.1
)

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)

validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(
    train_data, batch_size=args.batch_size
)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_dataloader = DataLoader(
    validation_data, batch_size=args.batch_size
)


config = transformers.BertConfig(
    #vocab_size=len(tokenizer),
    vocab_size=10000,
    hidden_size=768,
    num_hidden_layers=6,
    max_position_embeddings=128,
    type_vocab_size=1,
)
model = transformers.BertForSequenceClassification(config)
if device == "cuda":
    model.cuda()

# print(model)


### Training settings
optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
total_steps = len(train_dataloader) * args.epochs
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


### Training
model.zero_grad()

for epoch in range(1, args.epochs + 1):
    print("")
    print("======== Epoch {:} / {:} ========".format(epoch, args.epochs))
    print("Training...")

    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        if step % 100 == 0 and not step == 0:
            print("  Batch {:>5,}  of  {:>5,}.".format(step, len(train_dataloader)))

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()

        optimizer.step()
        scheduler.step()
        model.zero_grad()

    avg_train_loss = total_loss / len(train_dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("")
    print("Running Validation...")

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        logits_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        eval_accuracy += np.mean(logits_flat == labels_flat)
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

print("")
print("Training complete!")
