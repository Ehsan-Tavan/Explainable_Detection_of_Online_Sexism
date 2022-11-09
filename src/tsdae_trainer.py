"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.
TSDAE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.
Usage:
python train_tsdae_from_file.py path/to/sentences.txt
"""
import logging
from sentence_transformers import SentenceTransformer, LoggingHandler, models, datasets, losses
from torch.utils.data import DataLoader
from datetime import datetime
import sys
from data_loader import read_csv
from cleantext.sklearn import CleanTransformer

cleaner = CleanTransformer(no_punct=True, lower=True, no_line_breaks=True,
                           no_urls=True, no_emails=True, no_digits=True, no_numbers=True,
                           no_phone_numbers=True, no_currency_symbols=True)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Train Parameters
model_name = '/home/LanguageModels/bert_large_uncased'
batch_size = 32

# Save path to store our model
output_name = ''
if len(sys.argv) >= 3:
    output_name = "-" + sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")

model_output_path = '../assets/saved_models/tsdae/train_tsdae{}-{}'.format(output_name,
                                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

################# Read the train corpus  #################
train_sentences = []
reddit_data = read_csv("../data/Raw/starting_ki/reddit_1M_unlabelled.csv")
gab_data = read_csv("../data/Raw/starting_ki/gab_1M_unlabelled.csv")
for sentence in list(reddit_data["text"]):
    train_sentences.append(sentence)
for sentence in gab_data.text:
    train_sentences.append(sentence)

train_sentences = cleaner.transform(train_sentences)
################# Intialize an SBERT model #################

word_embedding_model = models.Transformer("bert-large-uncased")
# Apply **cls** pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

################# Train and evaluate the model (it needs about 1 hour for one epoch of AskUbuntu) #################
# We wrap our training sentences in the DenoisingAutoEncoderDataset to add deletion noise on the fly
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name,
                                             tie_encoder_decoder=True)

logging.info("Start training")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True,
    checkpoint_path=model_output_path,
    checkpoint_save_steps=10000,
    use_amp=False  # Set to True, if your GPU supports FP16 cores
)
