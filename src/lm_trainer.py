# -*- coding: utf-8 -*-

"""
    Explainable Detection of Online Sexism Project:
        src:
            lm_trainer.py
"""
# ============================ Third Party libs ============================
import os
import numpy as np
import logging
import torch
import transformers
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.utils import class_weight

# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_csv, write_json
from dataset import DataModule
from utils import make_labels
from models.lm_classifier import Classifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    print(ARGS.lm_model_path)
    TOKENIZER = transformers.BertTokenizer.from_pretrained(ARGS.lm_model_path)
    LOGGER = CSVLogger(ARGS.saved_model_dir, name=ARGS.model_name)

    # Load data
    logging.info("Loading train data ...")
    TRAIN_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.train_data_file),
                          columns=ARGS.data_headers,
                          names=ARGS.customized_headers)
    TRAIN_DATA, DEV_DATA = train_test_split(TRAIN_DATA, test_size=ARGS.dev_size)
    logging.info("Loading test data ...")
    TEST_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.test_data_file),
                         columns=ARGS.test_data_headers,
                         names=ARGS.test_customized_headers)

    logging.debug(TRAIN_DATA.head(), TEST_DATA.head())
    logging.debug("length of Train data is: %s" % len(TRAIN_DATA))
    logging.debug("length of Dev data is: %s" % len(DEV_DATA))
    logging.debug("length of Test data is: %s" % len(TEST_DATA))

    LABEL_ENCODER = make_labels(list(TRAIN_DATA["label_sexist"]))
    TRAIN_LABELS = LABEL_ENCODER.transform(list(TRAIN_DATA["label_sexist"]))
    DEV_LABELS = LABEL_ENCODER.transform(list(DEV_DATA["label_sexist"]))

    logging.debug("Maximum length is: %s", ARGS.max_len)
    # Calculate class_weights
    class_weights = class_weight.compute_class_weight(
        "balanced",
        classes=np.unique(TRAIN_LABELS),
        y=np.array(TRAIN_LABELS))

    logging.debug("class_weights is: %s", class_weights)

    DATA_MODULE = DataModule(train_data=list(TRAIN_DATA["text"]), train_labels=TRAIN_LABELS,
                             val_data=list(DEV_DATA["text"]), val_labels=DEV_LABELS, config=ARGS,
                             tokenizer=TOKENIZER)

    DATA_MODULE.setup()

    # Instantiate the Model Trainer
    CHECKPOINT_CALLBACK = ModelCheckpoint(monitor="val_loss",
                                          filename="QTag-{epoch:02d}-{val_loss:.2f}",
                                          save_top_k=1,  # ARGS.save_top_k,
                                          mode="min")
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=10)

    # Instantiate the Model Trainer
    TRAINER = pl.Trainer(max_epochs=ARGS.n_epochs, gpus=[0],
                         callbacks=[CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER)
    N_CLASSES = len(np.unique(TRAIN_LABELS))

    # Train the Classifier Model
    MODEL = Classifier(lm_model_path=ARGS.lm_model_path, num_classes=N_CLASSES,
                       class_weights=torch.Tensor(class_weights), lr=ARGS.lr)

    TRAINER.fit(MODEL, DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)

    # save best model path
    write_json(path=os.path.join(ARGS.assets_dir, ARGS.model_name,
                                 "b_model_path.json"),
               data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
