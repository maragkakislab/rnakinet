"""RNAkinet v3.0.

Network: raw signal (after the read's start is skipped) through three
convolution blocks, each two valid convolutions (kernel 9) and max pooling
3 (27x downsampling, ~1 step per nucleotide; 64, 128, 256 channels), then a
bidirectional GRU (128 units per direction). The read embedding is the max
and the mean of the GRU's outputs over the read's own positions plus its
final states (768 values); a small classifier turns it into the score.
Reads are batched padded to the longest read of the batch with their
lengths (input mode 'variable'); padding is never seen (valid convolutions,
packed GRU, masked pooling), so a read scores the same alone or in any
batch, and its score does not depend on how much padding it got.

Training (one run): plain BCE for loss_warmup_steps, then
- a non-negative positive-unlabeled (PU) label loss: reads of a labeling
  run are a mixture with a fraction labeled_fraction of labeled reads,
  control reads are clean negatives. With y = 1 for labeling-run reads (U)
  and y = 0 for control reads (N), U = pi P + (1 - pi) N, and the balanced
  risk of telling labeled reads (P) from N is
      R = 0.5 E_P[l(g, 1)] + 0.5 E_N[l(g, 0)],
      E_P[l(g, 1)] = (E_U[l(g, 1)] - (1 - pi) E_N[l(g, 1)]) / pi.
  The bracket can turn negative when the model overfits; training then
  steps along minus the bracket (Kiryo et al. 2017). valid_loss is R with
  the bracket clamped at 0; valid_bce is the plain BCE.
- a run adversary: two heads on the read embedding, behind a gradient
  reversal layer, predict which labeling run a labeling-run read came from
  and which control run a control read came from (per class, as the label
  is constant within a run). The reversed gradient pushes the network to
  make runs of a class indistinguishable. Its weight ramps from 0 to
  adv_weight over adv_ramp_steps (DANN schedule).
Checkpoint selection and early stopping (on valid_loss) start once the
adversary is fully on (selection_start_step).

Inference scores reads of any length above the minimum, cut to their first
200k samples (the longest training reads; inference_crop).

The adversary heads exist only in training (train.py passes the run counts,
needs_runs); inference builds the model without them and drops their
weights on loading."""
import math

import lightning.pytorch as pl
import torch
import torchmetrics
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from rnakinet.models.model import Permute

CHANNELS = (64, 128, 256)
KERNEL = 9
POOL = 3
HIDDEN = 128


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight * grad, None


class RNAkinetV3(pl.LightningModule):
    input_mode = 'variable'
    # inference defaults: no upper length limit; reads longer than the
    # longest training reads (200k samples) are scored on their first 200k
    inference_max_len = 0
    inference_crop = 200000
    # train.py passes the number of labeling and control runs
    needs_runs = True

    def __init__(self, lr=1e-4, warmup_steps=100, wd=0.01, logging_steps=1,
                 n_pos_runs=0, n_neg_runs=0, labeled_fraction=0.8,
                 loss_warmup_steps=150000, adv_weight=0.1,
                 adv_ramp_steps=20000, adv_hidden=64):
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.wd = wd
        self.logging_steps = logging_steps
        self.labeled_fraction = labeled_fraction
        self.loss_warmup_steps = loss_warmup_steps
        self.adv_weight = adv_weight
        self.adv_ramp_steps = adv_ramp_steps

        layers = []
        channels = [1, *CHANNELS]
        for c_in, c_out in zip(channels, channels[1:]):
            layers += [torch.nn.Conv1d(c_in, c_out, KERNEL), torch.nn.ReLU(),
                       torch.nn.Conv1d(c_out, c_out, KERNEL),
                       torch.nn.ReLU(), torch.nn.MaxPool1d(POOL)]
        self.convs = torch.nn.Sequential(*layers, Permute())
        self.gru = torch.nn.GRU(input_size=CHANNELS[-1], hidden_size=HIDDEN,
                                num_layers=1, batch_first=True,
                                bidirectional=True)
        embedding = 3 * 2 * HIDDEN
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embedding, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1))

        self.n_runs = {'pos': n_pos_runs, 'neg': n_neg_runs}
        self.adversary = torch.nn.ModuleDict({
            side: torch.nn.Sequential(
                torch.nn.Linear(embedding, adv_hidden), torch.nn.ReLU(),
                torch.nn.Linear(adv_hidden, n))
            for side, n in self.n_runs.items() if n >= 2
        })

        self.acc = torchmetrics.classification.Accuracy(task='binary')
        self.ce = torch.nn.BCELoss()
        self.adv_ce = torch.nn.CrossEntropyLoss()
        self.training_step_counter = 0
        self.cumulative_loss = 0
        self.cumulative_acc = 0

    # network

    @staticmethod
    def conv_lengths(lengths):
        """Positions per read after the blocks (two valid convolutions,
        then max pooling with stride POOL), at least 1."""
        for _ in CHANNELS:
            lengths = torch.div(lengths - 2 * (KERNEL - 1), POOL,
                                rounding_mode='floor')
        return lengths.clamp(min=1)

    def embed(self, x, lengths=None):
        """The read embedding: max and mean of the GRU's outputs over each
        read's own positions and its final states."""
        features = self.convs(x)
        steps = features.shape[1]
        if lengths is None:
            lengths = torch.full((len(x),), x.shape[-1])
        lengths = self.conv_lengths(lengths.cpu()).clamp(max=steps)
        packed = pack_padded_sequence(features, lengths, batch_first=True,
                                      enforce_sorted=False)
        output, h_n = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        total_length=steps)
        lengths = lengths.to(x.device)
        mask = (torch.arange(steps, device=x.device)[None, :]
                < lengths[:, None]).unsqueeze(-1)
        mean_pooled = (output * mask).sum(dim=1) / lengths[:, None]
        max_pooled = output.masked_fill(~mask, float('-inf')).max(dim=1)[0]
        last_states = h_n.permute(1, 0, 2).flatten(start_dim=1)
        return torch.cat([max_pooled, mean_pooled, last_states], dim=1)

    def forward(self, x, lengths=None):
        return torch.sigmoid(self.classifier(self.embed(x, lengths)))

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Adversary weights load only into heads of the same shape (the
        same run counts); others keep their own."""
        own = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith('adversary.')
                      or (k in own and own[k].shape == v.shape)}
        state_dict.update({k: v for k, v in own.items()
                           if k.startswith('adversary.')
                           and k not in state_dict})
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    # training

    @property
    def selection_start_step(self):
        """train.py picks the best checkpoint and stops early only from
        here, when the PU loss and the adversary are fully on (the
        validation loss changes at the switch)."""
        return self.loss_warmup_steps + self.adv_ramp_steps

    def adversary_lambda(self):
        """0 during the warm-up, then 2 / (1 + exp(-10 p)) - 1 times
        adv_weight, p going from 0 to 1 over adv_ramp_steps (DANN)."""
        step = self.global_step - self.loss_warmup_steps
        if step <= 0:
            return 0.0
        p = min(1.0, step / max(1, self.adv_ramp_steps))
        return self.adv_weight * (2 / (1 + math.exp(-10 * p)) - 1)

    def label_loss(self, preds, y, loss_type):
        """BCE during the warm-up, then the non-negative PU risk."""
        bce = self.ce(preds, y)
        self.log(f'{loss_type}_bce', bce, on_epoch=True,
                 sync_dist=loss_type == 'valid')
        if self.global_step < self.loss_warmup_steps:
            return bce
        p = preds[:, 0].clamp(1e-7, 1 - 1e-7)
        u, n = y[:, 0] > 0.5, y[:, 0] <= 0.5
        loss_1, loss_0 = -torch.log(p), -torch.log1p(-p)
        # a batch without one of the classes contributes nothing to it
        mean = (lambda v, m: v[m].mean() if m.any() else v.sum() * 0)
        bracket = (mean(loss_1, u)
                   - (1 - self.labeled_fraction) * mean(loss_1, n))
        if loss_type == 'train' and bracket < 0:
            return -bracket
        return (0.5 * bracket.clamp(min=0) / self.labeled_fraction
                + 0.5 * mean(loss_0, n))

    def adversary_loss(self, embedding, y, runs, weight, loss_type):
        reversed_ = GradientReversal.apply(embedding, weight)
        total = 0.0
        sync = loss_type == 'valid'
        for side, label in (('pos', True), ('neg', False)):
            if side not in self.adversary:
                continue
            mask = (y[:, 0] > 0.5) == label
            if mask.sum() == 0:
                continue
            logits = self.adversary[side](reversed_[mask])
            loss = self.adv_ce(logits, runs[mask])
            acc = (logits.argmax(dim=1) == runs[mask]).float().mean()
            self.log(f'{loss_type}_adv_loss_{side}', loss, on_epoch=True,
                     sync_dist=sync)
            self.log(f'{loss_type}_adv_acc_{side}', acc, on_epoch=True,
                     sync_dist=sync)
            # chance level, for reading the accuracy
            self.log(f'adv_chance_{side}', 1.0 / self.n_runs[side],
                     on_epoch=True, sync_dist=sync)
            total = total + loss
        return total

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr,
                                      weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, total_iters=self.warmup_steps)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx, dataloader_idx=None):
        x, lengths, y, runs = train_batch
        embedding = self.embed(x, lengths)
        preds = torch.sigmoid(self.classifier(embedding))
        label_loss = self.label_loss(preds, y, 'train')
        weight = self.adversary_lambda()
        # the adversary heads always learn; the reversed gradient reaching
        # the network is scaled by lambda (0 during the warm-up)
        loss = label_loss + self.adversary_loss(embedding, y, runs, weight,
                                                'train')
        self.log('train_loss', label_loss, on_epoch=True)
        self.log('adv_lambda', weight, on_step=True, on_epoch=False)
        self.log('train acc', self.acc(preds, y), on_epoch=True)
        self.lr_schedulers().step()
        self.training_step_counter += 1
        self.cumulative_loss += label_loss.item()
        self.cumulative_acc += self.acc(preds, y)
        if self.training_step_counter % self.logging_steps == 0:
            self.log('train_loss_cum',
                     self.cumulative_loss / self.logging_steps,
                     on_step=True, on_epoch=False)
            self.log('train_acc_cum',
                     self.cumulative_acc / self.logging_steps,
                     on_step=True, on_epoch=False)
            self.cumulative_acc = 0
            self.cumulative_loss = 0
        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x, lengths, y, runs = val_batch
        embedding = self.embed(x, lengths)
        preds = torch.sigmoid(self.classifier(embedding))
        # averaged over GPUs: checkpointing and early stopping act on it
        self.log('valid_loss', self.label_loss(preds, y, 'valid'),
                 on_epoch=True, sync_dist=True)
        self.log('valid acc', self.acc(preds, y), on_epoch=True,
                 sync_dist=True)
        with torch.no_grad():
            self.adversary_loss(embedding, y, runs, 0.0, 'valid')
