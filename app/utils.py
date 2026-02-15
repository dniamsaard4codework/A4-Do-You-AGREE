"""
Utility module for the NLI Web Application.

Contains model definitions, tokenization helpers, and prediction functions
used by the Flask app for Natural Language Inference.

Author: Dechathon Niamsa-ard [st126235]
"""

import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


# ==================== Model Definitions ====================

class Embedding(nn.Module):
    """Token + Position + Segment embedding layer for BERT."""

    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(x)
        return self.norm(
            self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        )


def get_attn_pad_mask(seq_q, seq_k, device):
    """Create attention padding mask."""
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, device):
        super().__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, device):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.fc = nn.Linear(n_heads * self.d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(self.d_k, self.device)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class BERT(nn.Module):
    """BERT model for pre-training and sentence encoding."""

    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, n_segments,
                 vocab_size, max_len, device):
        super().__init__()
        self.params = {
            'n_layers': n_layers, 'n_heads': n_heads, 'd_model': d_model,
            'd_ff': d_ff, 'd_k': d_k, 'n_segments': n_segments,
            'vocab_size': vocab_size, 'max_len': max_len
        }
        self.embedding = Embedding(vocab_size, max_len, n_segments, d_model, device)
        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, d_model, d_ff, d_k, device)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.device = device

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        h_pooled = self.activ(self.fc(output[:, 0]))
        logits_nsp = self.classifier(h_pooled)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, logits_nsp

    def get_last_hidden_state(self, input_ids, segment_ids):
        """Get the last hidden state of all tokens (used for sentence embedding)."""
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        return output


# ==================== Tokenization ====================

def tokenize_sentence(sentence, word2id, max_seq_length=128):
    """
    Tokenize a sentence using the custom vocabulary.

    Returns:
        Tuple of (input_ids, attention_mask, segment_ids)
    """
    cleaned = re.sub("[.,!?\\-]", '', sentence.lower())
    words = cleaned.split()
    token_ids = [word2id.get(w, word2id['[MASK]']) for w in words]
    input_ids = [word2id['[CLS]']] + token_ids + [word2id['[SEP]']]

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length - 1] + [word2id['[SEP]']]

    attention_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids += [word2id['[PAD]']] * padding_length
    attention_mask += [0] * padding_length
    segment_ids += [0] * padding_length

    return input_ids, attention_mask, segment_ids


# ==================== Pooling & Prediction ====================

# Label mappings
LABEL_MAP = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}
LABEL_COLORS = {0: '#2d6a4f', 1: '#b5851a', 2: '#c1121f'}


def mean_pool(token_embeds, attention_mask):
    """Mean pooling over token embeddings, excluding padding."""
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    return torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)


def predict_nli(premise, hypothesis, bert_model, classifier_head, word2id, device):
    """
    Run NLI prediction on a premise-hypothesis pair.

    Returns:
        dict with keys: label, confidence, similarity, color
    """
    with torch.no_grad():
        p_ids, p_mask, p_seg = tokenize_sentence(premise, word2id)
        h_ids, h_mask, h_seg = tokenize_sentence(hypothesis, word2id)

        p_ids_t = torch.LongTensor([p_ids]).to(device)
        p_mask_t = torch.LongTensor([p_mask]).to(device)
        p_seg_t = torch.LongTensor([p_seg]).to(device)
        h_ids_t = torch.LongTensor([h_ids]).to(device)
        h_mask_t = torch.LongTensor([h_mask]).to(device)
        h_seg_t = torch.LongTensor([h_seg]).to(device)

        u_hidden = bert_model.get_last_hidden_state(p_ids_t, p_seg_t)
        v_hidden = bert_model.get_last_hidden_state(h_ids_t, h_seg_t)

        u = mean_pool(u_hidden, p_mask_t)
        v = mean_pool(v_hidden, h_mask_t)

        sim = cosine_similarity(
            u.cpu().numpy().reshape(1, -1),
            v.cpu().numpy().reshape(1, -1)
        )[0, 0]

        uv_abs = torch.abs(u - v)
        x = torch.cat([u, v, uv_abs], dim=-1)
        logits = classifier_head(x)
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()

    return {
        "label": LABEL_MAP[pred_class],
        "confidence": float(confidence),
        "similarity": float(sim),
        "color": LABEL_COLORS[pred_class],
    }


# ==================== Model Loading ====================

def load_model(model_dir='../model', device=None):
    """
    Load the trained Sentence-BERT model and vocabulary.

    Args:
        model_dir: Path to directory containing sbert_nli.pth and vocab.pkl
        device: torch device (default: CPU)

    Returns:
        Tuple of (bert_model, classifier_head, word2id, device)
    """
    if device is None:
        device = torch.device('cpu')

    # Load vocabulary
    vocab_path = f'{model_dir}/vocab.pkl'
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
        word2id = vocab_data['word2id']

    # Load checkpoint
    ckpt_path = f'{model_dir}/sbert_nli.pth'
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    params = checkpoint['model_params']

    # Reconstruct BERT model
    bert_model = BERT(
        n_layers=params['n_layers'], n_heads=params['n_heads'],
        d_model=params['d_model'], d_ff=params['d_ff'],
        d_k=params['d_k'], n_segments=params['n_segments'],
        vocab_size=params['vocab_size'], max_len=params['max_len'],
        device=device,
    ).to(device)
    bert_model.load_state_dict(checkpoint['bert_state_dict'])
    bert_model.eval()

    # Reconstruct classifier head
    classifier_head = nn.Linear(params['d_model'] * 3, 3).to(device)
    classifier_head.load_state_dict(checkpoint['classifier_state_dict'])
    classifier_head.eval()

    return bert_model, classifier_head, word2id, device
