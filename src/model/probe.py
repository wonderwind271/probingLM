import torch.nn as nn
from transformers import GPT2LMHeadModel

class NaturalProbingGPT2(nn.Module):
    def __init__(self, base_model: GPT2LMHeadModel, tokenizer, num_layers=12):
        super().__init__()
        self.base_model = base_model
        self.num_layers = num_layers
        self.vocab_size = len(tokenizer)
        self.d_model = base_model.config.hidden_size

        # Independent unembedding heads for layer 0 to layer 10 (i=1 to 11)
        self.probes = nn.ModuleList([
            nn.Linear(self.d_model, self.vocab_size, bias=False)
            for _ in range(self.num_layers - 1)  # layers 0 to 10
        ])

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  labels=labels,
                                  output_hidden_states=True,
                                  return_dict=True)

        loss_main = outputs.loss
        hidden_states = outputs.hidden_states  # list of [batch, seq, hidden_size]

        total_probe_loss = 0.0
        for i in range(self.num_layers - 1):  # layers 0..10 (i=0..10)
            h = hidden_states[i + 1].detach()  # skip embedding, use layer i+1
            logits = self.probes[i](h)  # shape: [batch, seq, vocab]
            loss_i = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_probe_loss += loss_i

        total_loss = loss_main + total_probe_loss
        return total_loss, loss_main, total_probe_loss
