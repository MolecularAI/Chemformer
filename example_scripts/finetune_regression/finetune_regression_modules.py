import math
import torch
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from pysmilesutils.augment import SMILESAugmenter 

from molbart.tokeniser import MolEncTokeniser
from molbart.data.datasets import ReactionDataset 
from molbart.data.datamodules import _AbsDataModule
from molbart.models.pre_train import _AbsTransformerModel

from molbart.models.util import (
    PreNormEncoderLayer,
    PreNormDecoderLayer
)





# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Dataset ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class RegPropDataset(ReactionDataset):
    """
    code adapted from Irwin Ross
    for Encoder Regression
    
    Args: 
        data_path (str): the path of a csv file, 
            which must contain the columns [SMILES, pXC50, SET]
    """
    def __init__(self, data_path):
        path = Path(data_path)
        df = pd.read_csv(path)
        data_in = df["SMILES"].tolist()
        data_out = df["pXC50"].tolist()    

        super().__init__(data_in, data_out)

        self.train_idxs, self.val_idxs, self.test_idxs = self._save_idxs(df)

    def _save_idxs(self, df):
        val_idxs = df.index[df["SET"] == "valid"].tolist()
        test_idxs = df.index[df["SET"] == "test"].tolist()

        idxs_intersect = set(val_idxs).intersection(set(test_idxs))
        if len(idxs_intersect) > 0:
            raise ValueError(f"Val idxs and test idxs overlap")

        idxs = set(range(len(df.index)))
        train_idxs = idxs - set(val_idxs).union(set(test_idxs))

        return train_idxs, val_idxs, test_idxs

    
      


# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Data Modules ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class RegPropDataModule(_AbsDataModule):
    """
    code adapted from Irwin Ross
    """
    def __init__(
        self,
        dataset: RegPropDataset,
        tokeniser: MolEncTokeniser,
        batch_size: int,
        max_seq_len: int,
        train_token_batch_size: Optional[int] = None,
        num_buckets: Optional[int] = None,
        forward_pred: Optional[bool] = True,
        val_idxs: Optional[List[int]] = None, 
        test_idxs: Optional[List[int]] = None,
        split_perc: Optional[float] = 0.2,
        augment: Optional[bool] = True
    ):
        super().__init__(
            dataset,
            tokeniser,
            batch_size,
            max_seq_len,
            train_token_batch_size,
            num_buckets,
            val_idxs, 
            test_idxs,
            split_perc
        )

        if augment:
            print("Augmenting the SMILES strings.")
            self.aug = SMILESAugmenter()  #SMILESRandomizer() : List[str]->List[str]
        else:
            print("No data augmentation.")
            self.aug = None
    
        self.forward_pred = forward_pred

        

    def _collate(self, batch, train=True):

        token_output = self._prepare_tokens(batch)
        SMILES_tokens = token_output["SMILES_tokens"]
        SMILES_mask = token_output["SMILES_mask"]
        targets = token_output["props"]

        SMILES_token_ids = self.tokeniser.convert_tokens_to_ids(SMILES_tokens)
        SMILES_token_ids = torch.tensor(SMILES_token_ids).transpose(0, 1)
        SMILES_pad_mask = torch.tensor(SMILES_mask, dtype=torch.bool).transpose(0, 1)

        collate_output = {
            #stay consistent with Ross code for the dictionary keys
            "encoder_input": SMILES_token_ids,
            "encoder_pad_mask": SMILES_pad_mask,
            "target": targets
        }

        return collate_output

    def _prepare_tokens(self, batch):
        """ Prepare smiles strings for the model

        The smiles strings are prepared for the forward prediction task, no masking

        Args:
            batch (list[tuple(SMILES_str, target_str)]): Batched input to the model

        Output:
            Dictionary output from tokeniser: {
                "SMILES_tokens" (List[List[str]]): Molecule tokens from tokeniser,
                "props" (List[List[str]]): Property value,
                "SMILES_masks" (List[List[int]]): 0 refers to not padded, 1 refers to padded
            }
        """

        inpSMILES, props = tuple(zip(*batch))

        # split at this point: prefix_token + SMILES 
        # to augment only the SMILES string
        def pref_split(smiles_string):
            """
            splits a string on '|' into a list with two elements, starting from the right
            the prefix token must be in '<pref_token>'
            eg. '<pXC50><OPRM1>|N1[C@H](C(NCCCC[C@@H...=CC=C3'
            or  '<OPRM1>|N1[C@H](C(NCCCC[C@@H...=CC=C3'
            """
            part = smiles_string.rsplit('|',1) 
            return [part[0] , part[1]]

        if self.aug is not None:
            pref_SMILES = list(map(pref_split, inpSMILES)) #list[list[prefix_str, SMILES_str]]
            prefs, SMILES_str = tuple(zip(*pref_SMILES)) #tuple(prefix_str), tuple(SMILES_str)
            aug_SMILES_str = self.aug(SMILES_str) #list[aug_SMILES_str]
            inpSMILES = tuple([x+y for x,y in zip(prefs, aug_SMILES_str)]) #tuple(inpSMILES)
        else:
            pref_SMILES = list(map(pref_split, inpSMILES)) #list[list[prefix_str, SMILES_str]]
            prefs, SMILES_str = tuple(zip(*pref_SMILES)) #tuple(prefix_str), tuple(SMILES_str)
            inpSMILES = tuple([x+y for x,y in zip(prefs, SMILES_str)]) #tuple(inpSMILES)
            
            
        SMILES_output = self.tokeniser.tokenise(inpSMILES, pad=True)

        SMILES_tokens = SMILES_output["original_tokens"]
        SMILES_mask = SMILES_output["masked_pad_masks"]  # masked_pad_masks
        SMILES_tokens, SMILES_mask = self._check_seq_len(SMILES_tokens, SMILES_mask)


        token_output = {
            "SMILES_tokens": SMILES_tokens,
            "SMILES_mask": SMILES_mask,
            "props": torch.unsqueeze(torch.tensor(props),dim=1)
        }

        return token_output
        
        
        
# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------  Models ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
                
                
class RegrTransformerModel(pl.LightningModule):    
    """
    code adapted from Irwin Ross
    
    Encoder for Regression to train from scratch
    """
    def __init__(
        self, 
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        h_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        dropout,
        dropout_p,
        max_seq_len,
        batch_size,
        epochs,
        augment=None
    ):
        super(RegrTransformerModel, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.h_feedforward = h_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation
        self.num_steps = num_steps
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.max_seq_len = max_seq_len 
        self.batch_size = batch_size
        self.epochs = epochs 
        self.augment = augment

        self.save_hyperparameters()

        self.emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._positional_embs())
        
        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)
        
        
        # Regression part
        self.drpmem = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(d_model) 
        self.hidden_fc = nn.Linear(d_model, h_feedforward)
        
        self.drp = nn.Dropout(dropout_p)
        self.ln2 = nn.LayerNorm(h_feedforward) 
        
        self.predict_fc = nn.Linear(h_feedforward, 1) 
        self.loss_fn = nn.MSELoss()

        self._init_params()
        
        
    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "masked_tokens": tensor of token_ids of shape (seq_len, batch_size),
                "pad_masks": bool tensor of padded elems of shape (seq_len, batch_size),
                "sentence_masks" (optional): long tensor (0 or 1) of shape (seq_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output")
        """

        encoder_input = x["encoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)

        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)

        memory = memory[1,:,:] #in the 2rd element is the gene_symbol i.e. 1st in python
        #eg '^','<OPRD1>','O','=','C','1','N',...
        
#         memory = torch.mean(memory,dim=0)  #memory.reshape(memory.shape[1], -1) #to take the average from all last outputs of encoder

        
        x = self.drpmem(memory)
        x = self.ln(x)
        x = self.hidden_fc(x)
        
        x = F.relu(x)
        x = self.drp(x)
        x = self.ln2(x)
        model_output = self.predict_fc(x)

        return model_output
    
    
    def training_step(self, batch, batch_idx):
        self.train()
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output)
        self.log("train_loss", loss, on_step=True, logger=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        
        
    def test_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output)
        return loss
        

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.998), eps=1e-08, weight_decay=self.weight_decay)
    
        scheduler = OneCycleLR(optim, max_lr=self.lr, total_steps=self.num_steps)
        lr_dict = {
            "scheduler": scheduler,
            "interval": "step"
        }
        
        return [optim], [lr_dict]

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)  

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz):
        """ 
        Method from Pytorch transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode 
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        or Kaiming He uniform initialisation
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
                
                
                
                
##################################################################################################################




class FineTuneTransformerModel(pl.LightningModule):
    """
    code adapted from Irwin Ross
    Encoder for Regression to fine-tune
    """
    def __init__(
        self, 
        vocab_size, 
        d_premodel,
        premodel,
        h_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        dropout_p,
        max_seq_len,
        batch_size,
        epochs,
        augment=None
    ):
        super(FineTuneTransformerModel, self).__init__()

        self.premodel = premodel
        self.d_premodel = d_premodel
        self.h_feedforward = h_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation
        self.num_steps = num_steps
        self.dropout_p = dropout_p
        self.max_seq_len = max_seq_len 
        self.batch_size = batch_size
        self.epochs = epochs 
        self.augment = augment      
        
 
        # Regression part
        self.drpmem = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(self.d_premodel) 
        self.hidden_fc = nn.Linear(self.d_premodel, h_feedforward)
        self.drp = nn.Dropout(dropout_p)        
        self.predict_fc = nn.Linear(h_feedforward, 1)
        self.loss_fn = nn.MSELoss()

        
        
    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "masked_tokens": tensor of token_ids of shape (seq_len, batch_size),
                "pad_masks": bool tensor of padded elems of shape (seq_len, batch_size),
                "sentence_masks" (optional): long tensor (0 or 1) of shape (seq_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output")
        """

        memory = self.premodel(x)
        memory = memory[1,:,:] #in the 2rd element is the gene_symbol i.e. 1nd in python 
                                         # eg '^','<OPRD1>','O','=','C','1','N',...
#         memory = torch.mean(memory,dim=0) # take the average of the memory
        
            ## ADD A DROPOUT and try
        x = self.drpmem(memory)
        x = self.ln(x)
        x = self.hidden_fc(x)
        x = F.relu(x)
        x = self.drp(x)
        model_output = self.predict_fc(x)


        return model_output
    
    
    def training_step(self, batch, batch_idx):
        self.train()
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output) #loss_fct(logits.view(-1), labels.view(-1))
        self.log("train_loss", loss, on_step=True, logger=True)#, prog_bar=True

        return loss
    

    def validation_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        
        
    def test_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output)
        return loss
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_epoch_end_val', avg_loss)
        

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.998), eps=1e-08, weight_decay=self.weight_decay)
        scheduler = OneCycleLR(optim, max_lr=self.lr, total_steps=self.num_steps)
        lr_dict = {
            "scheduler": scheduler,
            "interval": "step"
        }
        return [optim], [lr_dict]

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_premodel)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_premodel) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_premodel for dim in range(0, self.d_premodel, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_premodel] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz):
        """ 
        Method from Pytorch transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode 
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        or Kaiming He uniform initialisation
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
                

                

    
####################################### EncoderOfBARTModel ####################################################


class EncoderOfBARTModel(_AbsTransformerModel):
    """
    code adapted from Irwin Ross
    This is the same BARTModel class from Ross but
    the Decoder part is erased
    This is needed just to load the pretrained model
    """
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(
            pad_token_idx,
            vocab_size, 
            d_model,
            num_layers, 
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            dropout,
            #schedule=schedule,
            warm_up_steps=warm_up_steps,
            **kwargs
        )

        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        self.schedule = schedule
        self.warm_up_steps = warm_up_steps

        if self.schedule == "transformer":
            assert warm_up_steps is not None, "A value for warm_up_steps is required for transformer LR schedule"

        enc_norm = nn.LayerNorm(d_model)
        dec_norm = nn.LayerNorm(d_model)

        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_layer = PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        encoder_input = x["encoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)

        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return memory

    def encode(self, batch):
        """ Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)
        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, batch):
        """ Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_embs.device)

        model_output = self.decoder(
            decoder_embs, 
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask
        )
        token_output = self.token_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def configure_optimizers(self):
        params = self.parameters()
        optim = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))

        if self.schedule == "const":
            print("Using constant LR schedule.")
            sch = LambdaLR(optim, lr_lambda=lambda epoch: 1)

        elif self.schedule == "cycle":
            print("Using cyclical LR schedule.")
            cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}

        elif self.schedule == "transformer":
            print("Using original transformer schedule.")
            trans_sch = FuncLR(optim, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}

        else:
            raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.d_model ** -0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step ** -0.5, step * (self.warm_up_steps ** -1.5))
        return self.lr * mult * lr

    def validation_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]

        loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        perplexity = self._calc_perplexity(batch, model_output)
        mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        mol_acc = torch.tensor(metrics["accuracy"], device=loss.device)
        invalid = torch.tensor(metrics["invalid"], device=loss.device)

        # Log for prog bar only
        self.log("mol_acc", mol_acc, prog_bar=True, logger=False, sync_dist=True)

        val_outputs = {
            "val_loss": loss,
            "val_token_acc": token_acc,
            "perplexity": perplexity,
            "val_molecular_accuracy": mol_acc,
            "val_invalid_smiles": invalid
        }
        return val_outputs

    def validation_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def test_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]

        loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        perplexity = self._calc_perplexity(batch, model_output)
        mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.test_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        test_outputs = {
            "test_loss": loss.item(),
            "test_token_acc": token_acc,
            "test_perplexity": perplexity,
            "test_invalid_smiles": metrics["invalid"]
        }

        if self.test_sampling_alg == "greedy":
            test_outputs["test_molecular_accuracy"] = metrics["accuracy"]

        elif self.test_sampling_alg == "beam":
            test_outputs["test_molecular_accuracy"] = metrics["top_1_accuracy"]
            test_outputs["test_molecular_top_1_accuracy"] = metrics["top_1_accuracy"]
            test_outputs["test_molecular_top_2_accuracy"] = metrics["top_2_accuracy"]
            test_outputs["test_molecular_top_3_accuracy"] = metrics["top_3_accuracy"]
            test_outputs["test_molecular_top_5_accuracy"] = metrics["top_5_accuracy"]
            test_outputs["test_molecular_top_10_accuracy"] = metrics["top_10_accuracy"]

        else:
            raise ValueError(f"Unknown test sampling algorithm, {self.test_sampling_alg}")

        return test_outputs

    def test_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        pad_mask = batch_input["target_pad_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """ Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input["target"]
        target_mask = batch_input["target_pad_mask"]
        vocab_dist_output = model_output["token_output"]

        inv_target_mask = ~(target_mask > 0)
        log_probs = vocab_dist_output.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)

        seq_lengths = inv_target_mask.sum(dim=0)
        exp = - (1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean()

    def _calc_token_acc(self, batch_input, model_output):
        token_ids = batch_input["target"]
        target_mask = batch_input["target_pad_mask"]
        token_output = model_output["token_output"]

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total
        return accuracy

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_input = batch_input["encoder_input"]
        enc_mask = batch_input["encoder_pad_mask"]

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        encode_input = {
            "encoder_input": enc_input,
            "encoder_pad_mask": enc_mask
        }
        memory = self.encode(encode_input)
        mem_mask = enc_mask.clone()

        _, batch_size, _ = tuple(memory.size())

        decode_fn = partial(self._decode_fn, memory=memory, mem_pad_mask=mem_mask)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size, memory.device)

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, memory.device, k=self.num_beams)

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask
        }
        model_output = self.decode(decode_input)
        return model_output
    