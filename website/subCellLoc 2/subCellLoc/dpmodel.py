import os
import re

import torch
from djangoProject_myh.settings import STATICFILES_DIRS
from transformers import T5EncoderModel, T5Tokenizer
from subCellLoc.ModelClassify import ModelClassify


def getTrueGenes(path):
    result_list = []

    with open(path, 'r') as file:
        for line in file:
            stripped_line = line.strip()  # 去除每行的换行符和空格
            result_list.append(stripped_line)

    return result_list


device = torch.device("mps")
model_path = os.path.join(STATICFILES_DIRS[0], "T5Model")


def get_T5_model():
    model = T5EncoderModel.from_pretrained(model_path)
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

    return model, tokenizer


def getFeatureT5(seq):
    sequence_examples = [seq]
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    model, tokenizer = get_T5_model()
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    emb_0 = embedding_repr.last_hidden_state[0, :len(seq)]
    emb_0_per_protein = emb_0.mean(dim=0)
    return emb_0_per_protein

def getClassifyModel(seq_tensor):
    device = torch.device("mps")
    model = ModelClassify()
    model = torch.load("/Users/kongge/code/python/djangoProject_myh/static/ClassifyModel/netModel.pt")
    model.to(device)
    model.eval()
    seq_tensor = seq_tensor.view(1, 1024)
    result = model(seq_tensor)
    result = result.to('cpu')
    threshold = 0.5
    labels_cov = torch.where(result > threshold, torch.tensor(1), torch.tensor(0))
    label_list = labels_cov.numpy().tolist()
    label_list = [item for sublist in label_list for item in sublist]
    en_label = ['Envelope', 'Thylakoid lumen', 'Plastoglobule', 'Stroma', 'Thylakoid membrane']
    label_en_list = []
    for item in range(len(label_list)):
        if label_list[item] == 1:
            label_en_list.append(en_label[item])
    return label_en_list


if __name__ == '__main__':
    result_tensor = getFeatureT5("MASISSFGCFPQSTALAGTSSTTRCRTTVAARLADQSDDFAPLRSSGGNCGCVNNSGEFDRRKLLVSSVGLLIGALSYDSKDGDFASASQFADMPALKGKDYGKTKMKYPDYTETQSGLQYKDLRVGTGPIAKKGDKVVVDWDGYTIGYYGRIFEARNKTKGGSFEGDDKEFFKFTLGSNEVIPAFEEAVSGMALGGIRRIIVPPELGYPDNDYNKSGPRPMTFSGQRALDFVLRNQGLIDKTLLFDVELLKIVPN")
    label_en_list = getClassifyModel(result_tensor)
    print(label_en_list)