import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class AdapterLayer(nn.Module):
    def __init__(self, config, reduction_factor=16):
        super().__init__()
        self.input_dim = config.hidden_size
        self.down_size = self.input_dim // reduction_factor

        self.down_proj = nn.Linear(self.input_dim, self.down_size)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.input_dim)

    def forward(self, hidden_states):
        residual = hidden_states
        
        # Down projection
        down = self.down_proj(hidden_states)
        # Activation
        activated = self.activation(down)
        # Up projection
        up = self.up_proj(activated)
        
        # Add & Norm
        output = residual + up
        return output

class RobertaWithAdapter(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Load pre-trained RoBERTa
        self.roberta = RobertaModel(config)
        
        # Add adapter after each transformer layer
        self.adapters = nn.ModuleList([
            AdapterLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights
        self.post_init()
        
        # Freeze the original model parameters
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        # Only train the adapters and classifier
        for param in self.adapters.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,  # We need all hidden states
            return_dict=True,
        )

        # Convert tuple to list so we can modify it
        hidden_states = list(outputs.hidden_states)
        
        # Apply adapters to each layer's output
        for idx, adapter in enumerate(self.adapters):
            if idx > 0:  # Skip embedding layer
                hidden_states[idx] = adapter(hidden_states[idx])

        # Get sequence output (last hidden state)
        sequence_output = hidden_states[-1]
        
        # Pool the output (take [CLS] token representation)
        pooled_output = sequence_output[:, 0, :]
        
        # Classification
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=outputs.attentions,
        )
