import datetime
import argparse
import numpy as np
import pandas as pd
import itertools
import torch
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import molbart.util as util
from finetune_regression_modules import RegPropDataset, RegPropDataModule, RegrTransformerModel


# Default args
DEFAULT_STUDY_NAME = 'Regr_pXC50_' + str(datetime.datetime.now())
DEFAULT_data_path = 'all_TrainTestVal_pXC50.csv'
DEFAULT_vocab_path = "prop_bart_vocab.txt"
DEFAULT_genes_path = "/individual_files"
DEFAULT_BATCH_SIZE = 32
DEFAULT_ACC_BATCHES = 1
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SCHEDULE = "cycle"
DEFAULT_AUGMENT = True
DEFAULT_WARM_UP_STEPS = 3000
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = 24
DEFAULT_LIMIT_VAL_BATCHES = 1.0
DEFAULT_EPOCHS = 150
DEFAULT_GPUS = 1
DEFAULT_D_MODEL = 512 
DEFAULT_NUM_LAYERS = 6 
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FEEDFORWARD = 2048  
DEFAULT_H_FEEDFORWARD = 1024 
DEFAULT_lr = 3e-4
DEFAULT_ACTIVATION = 'relu' 
DEFAULT_drp = 0.15 
DEFAULT_Hdrp = 0.6


def load_model(args, vocab_size, total_steps, max_seq_len, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FEEDFORWARD,  H_FEEDFORWARD, lr, ACTIVATION, drp, Hdrp):

    model = RegrTransformerModel(
        vocab_size=vocab_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_feedforward=D_FEEDFORWARD,
        h_feedforward=H_FEEDFORWARD,
        lr=lr,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        activation=ACTIVATION,
        num_steps=total_steps,
        max_seq_len=max_seq_len,
        dropout=drp,
        dropout_p=Hdrp,     
        augment=True
    )
    return model


def build_trainer(args):
    """
    build the Trainer using pytorch_lightning
    """
    precision = 16 #32

    logger = TensorBoardLogger("tb_logs", name=args.study_name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", save_last=True, mode='min')

    trainer = Trainer(
        logger=logger, 
        gpus=args.gpus, 
        min_epochs=args.epochs, 
        max_epochs=args.epochs,
        precision=precision,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.clip_grad,
        callbacks=[lr_monitor, checkpoint_cb],
        progress_bar_refresh_rate=0,
        limit_val_batches=4
    )
    
    return trainer


def get_targs_preds(model, dl):
    """
    get the prediction and the targets
    Args: model, dataloader
    Returns: two lists with prediction and the targets 
    """
    preds = []
    targs = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for i, batch in enumerate(iter(dl)):
        batch['encoder_input'] = batch['encoder_input'].to(device) 
        batch['encoder_pad_mask'] = batch['encoder_pad_mask'].to(device) 
        batch['target'] = batch['target'].to(device) 
        
        batch_preds = model(batch).squeeze(dim=1).tolist()
        batch_targs = batch['target'].squeeze(dim=1).tolist()
        
        preds.append(batch_preds)
        targs.append(batch_targs)
        
    targs = list(itertools.chain.from_iterable(targs))
    preds = list(itertools.chain.from_iterable(preds))

    return preds, targs

def save_results_133QSAR(args, tokeniser, model):
    """
    compute the RMSE and R^2 for each gene symbol separately and save them to a .csv file
    Args: data_path, tokeniser, model
    Returns: two lists with prediction and the targets 
    """
    RMSEs_train = []
    R2s_train = []
    RMSEs = []
    R2s = []

    dff = pd.read_csv(args.data_path)
    gene_symbols = list(dff.Gene_Symbol.unique())

    for gene in gene_symbols:
        g_path =  f'{args.genes_path}/{gene}.csv'
        dataset1 = RegPropDataset(data_path=g_path)
        batch_size = 64 
        max_seq_len = 300 
        dm1 = RegPropDataModule(
            dataset1,
            tokeniser,
            batch_size,
            max_seq_len,
            forward_pred=True,
            augment=args.augment,
            val_idxs=dataset1.val_idxs,
            test_idxs=dataset1.test_idxs
        )
        dm1.setup()

        pred_train, y_train = get_targs_preds(model=model, dl=dm1.train_dataloader())
        R2_train = np.power(np.corrcoef(pred_train,y_train)[0,1], 2) 
        RMSE_train = np.sqrt(np.mean((np.array(pred_train) - np.array(y_train))**2))
        R2s_train.append(R2_train)
        RMSEs_train.append(RMSE_train)

        pred_test, y_test = get_targs_preds(model=model, dl=dm1.test_dataloader())
        R2_test = np.power(np.corrcoef(pred_test,y_test)[0,1], 2) 
        RMSE_test = np.sqrt(np.mean((np.array(pred_test) - np.array(y_test))**2))
        R2s.append(R2_test)
        RMSEs.append(RMSE_test)
        
    results = pd.DataFrame(
        {'Gene_Symbol':gene_symbols,
        'RMSE_train':RMSEs_train,
        'R2_train':R2s_train,
        'RMSE_test':RMSEs,
        'R2_test':R2s})
        
    save_path='results_NoTL_EncRegr133_pXC50.csv'
    results.to_csv(save_path, index=False)

def main(args):
    print("Building tokeniser...")
    tokeniser = util.load_tokeniser(args.vocab_path, args.chem_token_start_idx)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = RegPropDataset(data_path=args.data_path) 
    print("Finished dataset.")

    print("Building data module...")
    max_seq_len = 300 
    dm = RegPropDataModule(
        dataset,
        tokeniser,
        args.batch_size,
        max_seq_len,
        augment=args.augment, 
        val_idxs=dataset.val_idxs,
        test_idxs=dataset.test_idxs,
        num_buckets=args.num_buckets
    )
    print("Finished datamodule.")

    vocab_size = len(tokeniser)
    train_steps = util.calc_train_steps(args, dm)
    print(f"Train steps: {train_steps}")


    
    print("Loading model...")
    model = load_model(args, vocab_size, train_steps+1, max_seq_len, args.D_MODEL, args.NUM_LAYERS, args.NUM_HEADS,args.D_FEEDFORWARD,  args.H_FEEDFORWARD, args.lr, args.ACTIVATION, args.drp, args.Hdrp)
    print("Finished model.")

    print("Building trainer...")
    trainer = build_trainer(args)
    print("Finished trainer.")

    print("Fitting data module to trainer")
    trainer.fit(model, dm)
    print("Finished training.")
    
    if args.genes_path:
        print("Predict results for 133 QSAR")
        save_results_133QSAR(args, tokeniser, model)
        print("Finished results.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path", type=str, default= DEFAULT_data_path)
    parser.add_argument("--vocab_path", type=str, default=DEFAULT_vocab_path)
    parser.add_argument("--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START)
    parser.add_argument("--genes_path", type=str, default=DEFAULT_genes_path)
    parser.add_argument("--D_MODEL", type=float, default=DEFAULT_D_MODEL)
    parser.add_argument("--NUM_LAYERS", type=float, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--NUM_HEADS", type=float, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--D_FEEDFORWARD", type=float, default=DEFAULT_D_FEEDFORWARD)
    parser.add_argument("--H_FEEDFORWARD", type=float, default=DEFAULT_H_FEEDFORWARD)
    parser.add_argument("--lr", type=float, default=DEFAULT_lr)
    parser.add_argument("--ACTIVATION", type=str, default=DEFAULT_ACTIVATION)
    parser.add_argument("--drp", type=float, default=DEFAULT_drp)
    parser.add_argument("--Hdrp", type=float, default=DEFAULT_Hdrp)
    parser.add_argument("--gpus", type=float, default=DEFAULT_GPUS)
    parser.add_argument("--study_name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--augment", type=str, default=DEFAULT_AUGMENT)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--num_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument("--limit_val_batches", type=float, default=DEFAULT_LIMIT_VAL_BATCHES)

    args = parser.parse_args()
    main(args)