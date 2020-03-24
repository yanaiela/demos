from typing import List

import pandas as pd
import streamlit as st
import torch
from transformers import BertTokenizer, BertForMaskedLM, \
    RobertaTokenizer, RobertaForMaskedLM

MODELS = {
    'bert_base_uncased': (BertForMaskedLM, BertTokenizer, 'bert-base-uncased'),
    'bert_base_cased': (BertForMaskedLM, BertTokenizer, 'bert-base-cased'),
    'bert_large_uncased': (BertForMaskedLM, BertTokenizer, 'bert-large-uncased'),
    'bert_large_cased': (BertForMaskedLM, BertTokenizer, 'bert-large-cased'),
    'roberta_base': (RobertaForMaskedLM, RobertaTokenizer, 'roberta-base'),
    'roberta_large': (RobertaForMaskedLM, RobertaTokenizer, 'roberta-large'),

    'mbert_base-uncased': (BertForMaskedLM, BertTokenizer, 'bert-base-multilingual-uncased')
}


MASK = '[MASK]'

def preds_clean(predictions):
    new_preds = []
    for x in predictions:
        if x.startswith("Ġ"):
            x = x[1:]
        new_preds.append(x)
    return new_preds


def get_predictions(sentence, lm_tokenizer, lm_model, k=10) -> List:
    # pre, post = sentence.split('[MASK]')
    masked_preds = []
    vals = sentence.split('[MASK]')
    mask_token = lm_tokenizer.mask_token
    for i in range(len(vals) - 1):
        pre = f' {mask_token} '.join(vals[: i + 1]).strip()
        post = f' {mask_token} '.join(vals[i + 1:]).strip()
        print(pre, post)
        target = [lm_tokenizer.mask_token]
        tokens = [lm_tokenizer.cls_token] + lm_tokenizer.tokenize(pre)
        target_idx = len(tokens)
        tokens += target + lm_tokenizer.tokenize(post) + [lm_tokenizer.sep_token]
        print(tokens)
        print(target_idx)
        input_ids = lm_tokenizer.convert_tokens_to_ids(tokens)
        tens = torch.LongTensor(input_ids).unsqueeze(0)
        res = lm_model(tens)[0][0, target_idx]
        res = torch.nn.functional.softmax(res, -1)
        probs, best_k = torch.topk(res, k)
        best_k = [int(x) for x in best_k]
        best_k = lm_tokenizer.convert_ids_to_tokens(best_k)
        best_k = preds_clean(best_k)
        masked_preds.append(best_k)
    return masked_preds


@st.cache(allow_output_mutation=True)
def get_bert_models(model_name):
    model_cls, model_tokenizer, model_proper_name = MODELS[model_name]
    tokenizer = model_tokenizer.from_pretrained(model_proper_name)
    lm_model = model_cls.from_pretrained(model_proper_name)
    return tokenizer, lm_model


st.title("MLM Demo")

lm_models = list(MODELS.keys())

st.sidebar.title("Pre trained LMs")
used_models = []
for model in lm_models:
    if st.sidebar.checkbox(model):
        used_models.append(model)

models, tokenizers = [], []
for m in used_models:
    m_tokenizer, m_model = get_bert_models(m)
    models.append(m_model)
    tokenizers.append(m_tokenizer)

text = st.text_input("Input Sentence ('[MASK]' for the masking token)")
k = st.number_input("top_k", min_value=1, max_value=100, value=10, step=1)

st.subheader('LM predictions')

if MASK not in text:
    st.text(f'the "{MASK}" must appear in the text')
else:
    if text != '':
        progress_bar = st.progress(0)

        num_mask = text.count(MASK)
        model_predictions_per_mask = []
        for i in range(num_mask):
            model_predictions_per_mask.append([])
        # model_predictions
        for ind, (model, tokenizer) in enumerate(zip(models, tokenizers)):
            preds = get_predictions(text, tokenizer, model, k=k)
            for mask_ind, pred in enumerate(preds):
                model_predictions_per_mask[mask_ind].append(pred)
            # model_predictions.append(preds)
            progress_bar.progress(int((float(ind + 1) / len(models)) * 100))
        progress_bar.progress(100)
        progress_bar.empty()
        print(model_predictions_per_mask)

        for ind, model_predictions in enumerate(model_predictions_per_mask):
            st.text(f'{ind + 1}# mask')
            dict_data = {}
            for model, answers in zip(used_models, model_predictions):
                dict_data[model] = answers
            df = pd.DataFrame(dict_data)

            st.dataframe(df, width=1000, height=1000)
