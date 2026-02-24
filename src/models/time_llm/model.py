"""
Time-LLM model adapted for FRED-MD macroeconomic forecasting.

This implementation preserves the original Time-LLM architecture (Jin et al.,
2024) while tailoring the prompt template and default hyper-parameters for
monthly macroeconomic data.  FRED-MD is the sole numerical data source.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import transformers
from transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
)

from src.models.time_llm.layers import (
    FlattenHead,
    Normalize,
    PatchEmbedding,
    ReprogrammingLayer,
)
from src.data.prompt_dictionary import build_domain_anchored_prompt

transformers.logging.set_verbosity_error()


def _load_llm(name: str, llm_layers: int):
    """Return (model, tokenizer, dim) for the requested LLM backbone."""
    if name == "LLAMA":
        hub = "huggyllama/llama-7b"
        cfg = LlamaConfig.from_pretrained(hub)
        cfg.num_hidden_layers = llm_layers
        cfg.output_attentions = True
        cfg.output_hidden_states = True
        try:
            model = LlamaModel.from_pretrained(
                hub, config=cfg, trust_remote_code=True, local_files_only=True
            )
        except EnvironmentError:
            model = LlamaModel.from_pretrained(
                hub, config=cfg, trust_remote_code=True, local_files_only=False
            )
        try:
            tok = LlamaTokenizer.from_pretrained(hub, trust_remote_code=True, local_files_only=True)
        except EnvironmentError:
            tok = LlamaTokenizer.from_pretrained(hub, trust_remote_code=True, local_files_only=False)
        return model, tok, 4096

    if name == "GPT2":
        hub = "openai-community/gpt2"
        cfg = GPT2Config.from_pretrained(hub)
        cfg.num_hidden_layers = llm_layers
        cfg.output_attentions = True
        cfg.output_hidden_states = True
        try:
            model = GPT2Model.from_pretrained(
                hub, config=cfg, trust_remote_code=True, local_files_only=True
            )
        except EnvironmentError:
            model = GPT2Model.from_pretrained(
                hub, config=cfg, trust_remote_code=True, local_files_only=False
            )
        try:
            tok = GPT2Tokenizer.from_pretrained(hub, trust_remote_code=True, local_files_only=True)
        except EnvironmentError:
            tok = GPT2Tokenizer.from_pretrained(hub, trust_remote_code=True, local_files_only=False)
        return model, tok, 768

    raise ValueError(f"Unsupported LLM: {name}")


FRED_MD_DESCRIPTION = (
    "The FRED-MD dataset is a large monthly panel of US macroeconomic "
    "indicators published by the Federal Reserve Bank of St. Louis. "
    "It covers output, income, employment, housing, consumption, money, "
    "interest rates, prices, and stock markets. "
    "All series are transformed to stationarity following McCracken & Ng (2016)."
)  # Legacy; Domain-Anchored PaP (DEC-007) replaces this in per-variable prompts.


class FedTimeLLM(nn.Module):
    """Time-LLM tailored for FRED-MD macroeconomic forecasting.

    Differences from the vanilla Time-LLM:
    * Default patch_len / stride are tuned for monthly frequency.
    * Prompt-as-Prefix includes a FRED-MD specific description.
    * The API is config-dict based (no argparse dependency).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.seq_len: int = config["seq_len"]
        self.pred_len: int = config["pred_len"]
        self.label_len: int = config.get("label_len", self.pred_len)
        self.d_model: int = config.get("d_model", 16)
        self.d_ff: int = config.get("d_ff", 32)
        self.n_heads: int = config.get("n_heads", 8)
        self.enc_in: int = config["enc_in"]
        self.dropout: float = config.get("dropout", 0.1)
        self.patch_len: int = config.get("patch_len", 6)
        self.stride: int = config.get("stride", 3)
        self.num_tokens: int = config.get("num_tokens", 1000)
        self.top_k: int = 5

        llm_name = config.get("llm_model", "GPT2")
        llm_layers = config.get("llm_layers", 6)
        self.llm_model, self.tokenizer, self.d_llm = _load_llm(llm_name, llm_layers)

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token = "[PAD]"

        for p in self.llm_model.parameters():
            p.requires_grad = False

        self.description: str = config.get("description", FRED_MD_DESCRIPTION)
        self.column_names: list[str] = config.get("column_names", [])

        self.patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(
            self.d_model, self.n_heads, self.d_ff, self.d_llm
        )

        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(
            self.enc_in, self.head_nf, self.pred_len, head_dropout=self.dropout
        )
        self.normalize_layers = Normalize(self.enc_in, affine=False)

        self._prompt_cache: torch.Tensor | None = None

    # ── forward ────────────────────────────────────────────────

    def _get_prompt_embeddings(self, device: torch.device) -> torch.Tensor:
        """Return cached (N, prompt_len, d_llm) prompt embeddings.

        Prompts depend only on column_names, pred_len, seq_len — all constant
        across batches — so we tokenize and embed once, then expand per batch.
        """
        if self._prompt_cache is not None and self._prompt_cache.device == device:
            return self._prompt_cache

        N = len(self.column_names) or self.enc_in
        prompts = [
            build_domain_anchored_prompt(
                self.column_names[i] if i < len(self.column_names) else "",
                self.pred_len,
                self.seq_len,
            )
            for i in range(N)
        ]
        ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).input_ids
        with torch.no_grad():
            emb = self.llm_model.get_input_embeddings()(ids.to(device))
        self._prompt_cache = emb
        return emb

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc)[:, -self.pred_len :, :]

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor):
        x_enc = self.normalize_layers(x_enc, "norm")
        B, T, N = x_enc.size()

        # (N, prompt_len, d_llm) → (B*N, prompt_len, d_llm)
        prompt_emb_n = self._get_prompt_embeddings(x_enc.device)
        prompt_emb = prompt_emb_n.unsqueeze(0).expand(B, -1, -1, -1)
        prompt_emb = prompt_emb.reshape(B * N, prompt_emb_n.shape[1], -1)

        source_emb = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # (B, T, N) → (B, N, T) for PatchEmbedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_emb, source_emb)

        llm_input = torch.cat([prompt_emb, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llm_input).last_hidden_state
        dec_out = dec_out[:, :, : self.d_ff]

        dec_out = dec_out.reshape(B, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums :])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, "denorm")
        return dec_out

    def _compute_lags(self, x: torch.Tensor) -> torch.Tensor:
        q_fft = torch.fft.rfft(x.permute(0, 2, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(q_fft), dim=-1)
        mean_corr = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_corr, self.top_k, dim=-1)
        return lags
