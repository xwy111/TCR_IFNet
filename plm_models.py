import torch
import torch.nn as nn
from transformers import T5PreTrainedModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import ModelOutput


class Prott5(T5PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)

        config.use_cache = False
        config.is_decoder = False

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, self.shared)
        self.token_type_embeddings = nn.Embedding(2, config.d_model)

        self.feature_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.conv_extractor = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.d_model // 2, config.d_model // 4, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.d_model // 4, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, config.num_labels)
        )

        self.post_init()
        self.apply(self._init_custom_weights)
        self._debug_once = False

    def _init_custom_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        type_embeds = self.token_type_embeddings(token_type_ids)
        hidden_states = self.feature_projection(sequence_output + type_embeds)

        x = hidden_states.transpose(1, 2)
        conv_feats = self.conv_extractor(x)

        pooled_output, _ = torch.max(conv_feats, dim=2)
        logits = self.classifier(pooled_output)

        if self.training and not self._debug_once:
            print("\n" + "=" * 30)
            print(f"ProtT5 Mean: {sequence_output.mean().item():.4f}")
            print(f"Type Embedding Std: {type_embeds.std().item():.4f}")
            print(f"Conv Extractor Std: {conv_feats.std().item():.4f}")
            print(f"Logits Sample: {logits[0].detach().cpu().numpy()}")
            print("=" * 30 + "\n")
            self._debug_once = True

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return ModelOutput(loss=loss, logits=logits)