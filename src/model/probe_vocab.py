from .probe import BaseProbingGPT2
from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import GPT2LMHeadModel
import copy


class VocabProbingGPT2(BaseProbingGPT2):
    """Directly maps the hidden state of layer h to logits on vocab V"""
    
    def _create_probe(self, has_bias: bool):
        return nn.Linear(self.d_model, self.vocab_size, bias=has_bias)
    
    def __init__(self, base_model, tokenizer, num_layers=12, probing_layers=[], has_bias=True, loss_type="ce", freeze_backbone=False, device=None, add_layernorm=True):
        '''Freeze backbone: whether to train the backbone model and lens at the same time'''
        super().__init__(base_model, tokenizer, num_layers, probing_layers, has_bias, device=device)
        assert loss_type in ("ce", "kl")
        self.loss_type = loss_type
        self.freeze_backbone = freeze_backbone
        
        self.add_layernorm = add_layernorm
        if self.add_layernorm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.d_model) for _ in probing_layers
            ])
        
        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()
        print('loss type =', loss_type, ', probing layers =', probing_layers)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        assert labels is not None
        outputs = self.base_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True,
                                  labels=labels,
                                  return_dict=True)
        hidden_states = outputs.hidden_states
        total_probe_loss = 0.0
        probe_logits_ls = []
        
        if self.loss_type == "kl":
            final_logits = outputs.logits.detach()  # [B, S, V]
            final_probs = F.softmax(final_logits, dim=-1)

        for idx, layer in enumerate(self.probing_layers):
            h = hidden_states[layer + 1].detach()
            if self.add_layernorm:
                h = self.layer_norms[idx](h)
                
            # We don't hope the loss of probes to interfere with GPT2's loss,
            # so hidden state should be detached. If we use KL loss in native lens,
            # logits of the original GPT2 should also be detached.  
            probe_logits = self.probes[idx](h)
            probe_logits_ls.append(probe_logits)
            
            if labels is not None:
                if self.loss_type == "ce":
                    # Cross-entropy with next token labels
                    shifted_probe_logits = probe_logits[:, :-1, :]  # [8, 511, 89057]
                    shifted_labels = labels[:, 1:]  # [8, 511]
                    loss_i = self.loss_fn(shifted_probe_logits.reshape(-1, shifted_probe_logits.size(-1)), shifted_labels.reshape(-1))
                elif self.loss_type == "kl":
                    # KL divergence to final log probs
                    probe_logprobs = F.log_softmax(probe_logits, dim=-1)
                    loss_i = F.kl_div(probe_logprobs, final_probs, reduction="batchmean")
                total_probe_loss += loss_i

        if not self.freeze_backbone:
            return {'total_loss': outputs.loss + total_probe_loss, 'loss_main': outputs.loss, 'total_probe_loss': total_probe_loss, 'all_probe_logits': probe_logits_ls}
        else:
            return {'total_loss': total_probe_loss, 'total_probe_loss': total_probe_loss, 'all_probe_logits': probe_logits_ls}


if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("gpt2")