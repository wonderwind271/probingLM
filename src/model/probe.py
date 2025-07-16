from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


class ProbingOutput():
    def __init__(self, arg: dict):
        self.total_loss = arg['total_loss']
        self.total_probe_loss = arg['total_probe_loss']
        self.all_probe_logits = arg['all_probe_logits']
        if 'loss_main' in arg:
            self.loss_main = arg['loss_main']
        else:
            self.loss_main = None


class BaseProbingGPT2(nn.Module, ABC):
    def __init__(self, base_model: GPT2LMHeadModel, tokenizer, num_layers=12, has_bias=True):
        super().__init__()
        self.base_model = base_model
        self.num_layers = num_layers
        self.vocab_size = len(tokenizer)
        self.d_model = base_model.config.hidden_size
        self.device = base_model.device if hasattr(base_model, "device") else torch.device("cpu")

        self.probes = nn.ModuleList([
            nn.Linear(self.d_model, self.vocab_size, bias=has_bias)
            for _ in range(self.num_layers - 1)
        ])
        self.loss_fn = nn.CrossEntropyLoss()

    @abstractmethod
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Compute the probing loss and optionally the main loss"""
        pass

    def save_probing(self, path: str):
        torch.save(self.probes.state_dict(), path)

    def load_probing(self, path: str):
        self.probes.load_state_dict(torch.load(path, map_location=self.device))

    def save_base_model(self, dir_path: str):
        self.base_model.save_pretrained(dir_path)


class NaturalProbingGPT2(BaseProbingGPT2):
    """Main LLM + probing loss (natural probing, train both)"""
    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        outputs = self.base_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  labels=labels,
                                  output_hidden_states=True,
                                  return_dict=True)

        loss_main = outputs.loss
        hidden_states = outputs.hidden_states

        total_probe_loss = 0.0
        all_probe_logits = []
        for i in range(self.num_layers - 1):
            h = hidden_states[i + 1].detach()
            logits = self.probes[i](h)
            all_probe_logits.append(logits)
            if labels is not None:
                loss_i = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_probe_loss += loss_i

        total_loss = loss_main + total_probe_loss
        # return total_loss, loss_main, total_probe_loss, all_probe_logits
        return ProbingOutput({'total_loss': total_loss, 'loss_main': loss_main, 'total_probe_loss': total_probe_loss, 'all_probe_logits': all_probe_logits})


class LensProbingGPT2(BaseProbingGPT2):
    def __init__(self, base_model, tokenizer, num_layers=12, has_bias=True, loss_type="kl"):
        super().__init__(base_model, tokenizer, num_layers, has_bias)
        assert loss_type in ("ce", "kl")
        self.loss_type = loss_type

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True,
                                      return_dict=True)

        hidden_states = outputs.hidden_states
        final_logits = outputs.logits.detach()  # [B, S, V]

        total_probe_loss = 0.0
        all_probe_logits = []
        if self.loss_type == "kl":
            final_probs = F.softmax(final_logits, dim=-1)  # once
        
        for i in range(self.num_layers - 1):  # layers 1 to 11
            h = hidden_states[i + 1]  # [B, S, D]
            probe_logits = self.probes[i](h)  # [B, S, V]
            all_probe_logits.append(probe_logits)

            if labels is not None:
                if self.loss_type == "ce":
                    # Cross-entropy with next token labels
                    loss_i = self.loss_fn(probe_logits.view(-1, probe_logits.size(-1)), labels.view(-1))

                elif self.loss_type == "kl":
                    # KL divergence to final logits
                    # Apply softmax (teacher) and log_softmax (student)
                    probe_logprobs = F.log_softmax(probe_logits, dim=-1)

                    loss_i = F.kl_div(probe_logprobs, final_probs, reduction="batchmean")

                total_probe_loss += loss_i

        # return total_probe_loss, None, total_probe_loss, all_probe_logits
        return ProbingOutput({'total_loss': total_probe_loss, 'total_probe_loss': total_probe_loss, 'all_probe_logits': all_probe_logits})

