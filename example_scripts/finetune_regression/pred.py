import torch
import numpy as np
import pandas as pd
import molbart.util as util
from finetune_regression_modules import RegPropDataset, RegPropDataModule, FineTuneTransformerModel, EncoderOfBARTModel
from molbart.decoder import DecodeSampler 
from finetuneRegr import get_targs_preds


DEFAULT_BATCH_SIZE = 16
DEFAULT_ACC_BATCHES = 1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SCHEDULE = "cycle"
DEFAULT_AUGMENT = True
DEFAULT_WARM_UP_STEPS = 3000
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = 24
DEFAULT_LIMIT_VAL_BATCHES = 1.0
DEFAULT_EPOCHS = 150 
DEFAULT_GPUS = 1
DEFAULT_D_PREMODEL = 512
DEFAULT_MAX_SEQ_LEN = 300
DEFAULT_LR = 3e-4
DEFAULT_H_FEEDFORWARD = 2048
DEFAULT_drp = 0.2 
DEFAULT_Hdrp = 0.4 
DEFAULT_WEIGHT_DECAY = 0.0

def load_model(vocab_size, total_steps, pad_token_idx, tokeniser):

    sampler = DecodeSampler(tokeniser, DEFAULT_MAX_SEQ_LEN)
    # Pre-trained-combined model (Encoder+Decoder) but only using Encoder part
    premodel = EncoderOfBARTModel.load_from_checkpoint('D:/Jupyter/Chemformer_old/models/pre-trained/combined/step=1000000.ckpt',decode_sampler=sampler,pad_token_idx=pad_token_idx,vocab_size=vocab_size,num_steps=total_steps,lr=DEFAULT_LR,weight_decay=DEFAULT_WEIGHT_DECAY,schedule=DEFAULT_SCHEDULE,warm_up_steps=DEFAULT_WARM_UP_STEPS,dropout=DEFAULT_drp)

    premodel.decoder = torch.nn.Identity()
    premodel.token_fc = torch.nn.Identity()
    premodel.loss_fn = torch.nn.Identity()
    premodel.log_softmax = torch.nn.Identity()

    # Fine-tuned Encoder model on regression task
    model = FineTuneTransformerModel.load_from_checkpoint('D:/Jupyter/Chemformer/tb_logs/study_name/version_6/checkpoints/last.ckpt', d_premodel=DEFAULT_D_PREMODEL,vocab_size=vocab_size, premodel=premodel, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,h_feedforward=DEFAULT_H_FEEDFORWARD, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,activation='gelu', num_steps=total_steps, max_seq_len=DEFAULT_MAX_SEQ_LEN,dropout_p=DEFAULT_Hdrp, augment=DEFAULT_AUGMENT)

    return model

vocab_path = "D:/Jupyter/Chemformer/example_scripts/finetune_regression/prop_bart_vocab.txt"
chem_token_start_idx = util.DEFAULT_CHEM_TOKEN_START
tokeniser = util.load_tokeniser(vocab_path, chem_token_start_idx)
vocab_size = len(tokeniser)
pad_token_idx = tokeniser.vocab[tokeniser.pad_token]
train_steps = 42300

model = load_model(vocab_size, train_steps+1, pad_token_idx, tokeniser)

dataset = RegPropDataset(data_path='D:/Jupyter/Chemformer_old/MPP/ESOL.csv') 
dm = RegPropDataModule(dataset, tokeniser, DEFAULT_BATCH_SIZE*2, DEFAULT_MAX_SEQ_LEN)
dm.setup()
pred, gt = get_targs_preds(model=model, dl=dm.test_dataloader())
print(len(pred), len(gt))
df = pd.DataFrame()
df['pred'] = pred
df['gt'] = gt
df.to_csv('preds.csv')