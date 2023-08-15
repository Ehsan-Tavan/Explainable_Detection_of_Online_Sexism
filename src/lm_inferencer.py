# -*- coding: utf-8 -*-

"""
    Explainable Detection of Online Sexism Project:
        src:
            lm_inferencer.py
"""
# ============================ Third Party libs ============================
import os
import logging
import pandas
import torch
import numpy
import transformers
from torch.utils.data import DataLoader
# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_pickle, read_csv
from models.lm_classifier import Classifier
from dataset import InferenceDataset

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TOKENIZER = transformers.AutoTokenizer.from_pretrained(ARGS.lm_model_path)

    LABEL_ENCODER = read_pickle(
        path=os.path.join(ARGS.saved_model_dir, ARGS.model_name, "label_encoder.pkl"))
    logging.info("Loading test data ...")
    TEST_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.test_data_file),
                         columns=ARGS.test_data_headers,
                         names=ARGS.test_customized_headers)

    MODEL_PATH = "/home/ehsan.tavan/repetitive-news/assets/saved_models/assets/saved_models/Bert/" \
                 "version_2/checkpoints/QTag-epoch=01-val_loss=0.41.ckpt"
    MODEL = Classifier.load_from_checkpoint(MODEL_PATH)

    MODEL.model.save_pretrained("./xlm")
    MODEL.eval()#.to("cuda:0")

    DATASET = InferenceDataset(texts=list(TEST_DATA.text),
                               tokenizer=TOKENIZER,
                               max_len=ARGS.max_len)
    DATALOADER = DataLoader(DATASET, batch_size=64,
                            shuffle=False, num_workers=4)

    PREDICTED_LABELS = []
    for i_batch, sample_batched in enumerate(DATALOADER):
        with torch.no_grad():
            sample_batched["inputs_ids"] = sample_batched["inputs_ids"]#.to("cuda:0")
            sample_batched["attention_mask"] = sample_batched["attention_mask"]#.to("cuda:0")
            OUTPUTS = MODEL(sample_batched)
            OUTPUTS = numpy.argmax(OUTPUTS.cpu().detach().numpy(), axis=1)
            PREDICTED_LABELS.extend(OUTPUTS)

    PREDICTED_LABELS = list(LABEL_ENCODER.inverse_transform(PREDICTED_LABELS))

    RESULTS = pandas.DataFrame(
        {"rewire_id": list(TEST_DATA["rewire_id"]), "label_pred": PREDICTED_LABELS})

    RESULTS.to_csv("result.csv", index=False, encoding="utf-8")
