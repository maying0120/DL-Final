import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.activation import MultiheadAttention

from transformers import BertModel, BertConfig


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dim_hidden, h, prob_dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = MultiheadAttention(dim_hidden, h, prob_dropout)

    def forward(self, query, key, value, mask):
        # query, key value are arranged as seq * batch_size * dim_model
        # mask should be batch_size * num_heads * query_length * key/value_length
        return self.attention(query, key, value, attn_mask=mask)[0]


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, dim_model, dim_ff, prob_dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(dim_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(prob_dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TypeEncoding(nn.Module):

    def __init__(self, d_model):
        super(TypeEncoding, self).__init__()
        self.emb = nn.Embedding(2, d_model)

    def forward(self, x):
        return self.emb(x)


class ResidualLayer(nn.Module):

    def __init__(self):
        super(ResidualLayer, self).__init__()

    def forward(self, x, sublayer):
        return x + sublayer


class SourceLayer(nn.Module):

    def __init__(self, dim_hidden, dim_ff, n_head, dropout):
        super(SourceLayer, self).__init__()

        self.n_head = n_head

        self.attn = MultiHeadAttentionLayer(dim_hidden, n_head, dropout)
        self.ff = PositionWiseFeedForwardLayer(dim_hidden, dim_ff, dropout)
        self.residual = ResidualLayer()

    def forward(self, x, mask):
        out = x.transpose(0, 1)
        mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1)).repeat(self.n_head, 1, 1)
        out = self.residual(out, self.attn(out, out, out, mask))
        out = self.residual(out, self.ff(out))
        return out.transpose(0, 1)


class ContextLayer(nn.Module):

    def __init__(self, dim_hidden, dim_ff, n_head, dropout):
        super(ContextLayer, self).__init__()

        self.n_head = n_head

        self.attn = MultiHeadAttentionLayer(dim_hidden, n_head, dropout)
        self.ff = PositionWiseFeedForwardLayer(dim_hidden, dim_ff, dropout)
        self.residual = ResidualLayer()

    def forward(self, x, mem, x_mask, mem_mask):
        x = x.transpose(0, 1)
        mem = mem.transpose(0, 1)
        mask = torch.bmm(x_mask.unsqueeze(-1), mem_mask.unsqueeze(1)).repeat(self.n_head, 1, 1)
        out = self.residual(x, self.attn(x, mem, mem, mask))
        out = self.residual(out, self.ff(out))
        return out.transpose(0, 1)


class GateLayer(nn.Module):

    def __init__(self, dim_model):
        super(GateLayer, self).__init__()
        self.reset = nn.Linear(dim_model*3, dim_model)
        self.update = nn.Linear(dim_model*3, dim_model)
        self.proposal = nn.Linear(dim_model*3, dim_model)

    def forward(self, x_1, x_2):
        reset = torch.sigmoid(self.reset(torch.cat([x_1, x_2], -1)))
        update = torch.sigmoid(self.update(torch.cat([x_1, x_2], -1)))
        proposal = torch.tanh(self.proposal(torch.cat([reset * x_1, x_2], -1)))
        out = (1 - update) * x_1 + update * proposal
        return out


class TargetLayer(nn.Module):

    def __init__(self, dim_hidden, dim_ff, n_head, dropout):
        super(TargetLayer, self).__init__()

        self.n_head = n_head

        self.self_attn = MultiHeadAttentionLayer(dim_hidden, n_head, dropout)
        self.cross_attn = MultiHeadAttentionLayer(dim_hidden, n_head, dropout)
        self.ff = PositionWiseFeedForwardLayer(dim_hidden, dim_ff, dropout)
        self.residual = ResidualLayer()

    def forward(self, x, mem, x_mask, mem_mask):
        x = x.transpose(0, 1)
        mem = mem.transpose(0, 1)
        mem_mask = torch.bmm(x_mask.unsqueeze(-1), mem_mask.unsqueeze(1)).repeat(self.n_head, 1, 1)
        x_mask = torch.bmm(x_mask.unsqueeze(-1), x_mask.unsqueeze(1)).repeat(self.n_head, 1, 1)
        x_mask = x_mask & torch.tril(torch.ones(1, x_mask.size(1), x_mask.size(1)), 0).type_as(x_mask.data)
        out = self.residual(x, self.self_attn(x, x, x, x_mask))
        out = self.redidual(out, self.cross_attn(out, mem, mem, mem_mask))
        out = self.residual(out, self.ff(out))
        return out.transpose(0, 1)


class GenerationLayer(nn.Module):

    def __init__(self, dim_model, voc_size, dout_p):
        super(GenerationLayer, self).__init__()
        self.linear = nn.Linear(dim_model, voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(voc_size, voc_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))
        return F.log_softmax(x, dim=-1)


class ContextModel(nn.Module):

    def __init__(self, dim_text, dim_video, dim_hidden, dim_ff, n_head, n_layer, dropout):
        super(ContextModel, self).__init__()
        self.n_layer = n_layer

        self.bert_emb = BertModel.from_pretrained('bert-base-uncased')

        self.text_emb = nn.Linear(dim_text, dim_hidden)
        self.video_emb = nn.Linear(dim_video, dim_hidden)
        self.type_emb = TypeEncoding(dim_hidden)
        self.pos_emb = PositionalEncoding(dim_hidden, dropout)

        self.src = nn.ModuleList([SourceLayer(dim_hidden, dim_ff, n_head, dropout) for _ in range(n_layer)])
        self.ctx = ContextLayer(dim_hidden, dim_ff, n_head, dropout)
        self.gate = GateLayer(dim_hidden)
        self.tgt = nn.ModuleList([TargetLayer(dim_hidden, dim_ff, n_head, dropout) for _ in range(n_layer)])

        self.generation = GenerationLayer(dim_hidden, BertConfig().vocab_size)

    def forward(self, v, t, label, v_mask, t_mask, label_mask):

        # unpack date
        prev_v, v, next_v = v
        prev_t, t, next_t = t
        prev_v_mask, v_mask, next_v_mask = v_mask
        prev_t_mask, t_mask, next_t_mask = t_mask

        # mapping text features
        prev_t, _ = self.bert_emb(input_ids=prev_t, attention_mask=prev_t_mask)
        t, _ = self.bert_emb(input_ids=prev_t, attention_mask=t_mask)
        next_t, _ = self.bert_emb(input_ids=prev_t, attention_mask=next_t_mask)
        label, _ = self.bert_emb(input_ids=label, attention_mask=label_mask)

        prev_t = self.text_emb(prev_t)
        t = self.text_emb(t)
        next_t = self.text_emb(next_t)
        label = self.text_emb(label)

        # mapping video features
        prev_v = self.video_emb(prev_v)
        v = self.video_emb(v)
        next_v = self.video_emb(next_v)

        # previous features multi-modality fusion
        prev_input = torch.cat([prev_v, prev_t], 1)
        prev_mask = torch.cat([prev_v_mask, prev_t_mask], 1)
        prev_type = torch.cat([torch.ones_like(prev_v_mask).long(), torch.zeros_like(prev_v_mask).long()], 1)

        # current features multi-modality fusion
        curr_input = torch.cat([v, t], 1)
        curr_mask = torch.cat([v_mask, t_mask], 1)
        curr_type = torch.cat([torch.ones_like(v_mask).long(), torch.zeros_like(v_mask).long()], 1)

        # next features multi-modality fusion
        next_input = torch.cat([next_v, next_t], 1)
        next_mask = torch.cat([next_v_mask, next_t_mask], 1)
        next_type = torch.cat([torch.ones_like(next_v_mask).long(), torch.zeros_like(next_t_mask).long()], 1)

        prev_input = self.pos_emb(prev_input + self.type_emb(prev_type))
        curr_input = self.pos_emb(curr_input + self.type_emb(curr_type))
        next_input = self.pos_emb(next_input + self.type_emb(next_type))

        for i in range(self.n_layer):
            prev_input = self.src[i](prev_input, prev_mask)
            curr_input = self.src[i](curr_input, curr_mask)
            next_input = self.src[i](next_input, next_mask)

            # context cross attention features
            prev_out = self.ctx(curr_input, prev_input, curr_mask, prev_mask)
            next_out = self.ctx(curr_input, next_input, curr_mask, next_mask)

            # encoder features after gate selection
            curr_out = self.gate(torch.cat([prev_out, next_out], -1), curr_input)

            # decoder
            target_out = self.tgt[i](label, curr_out, label_mask, curr_mask)

        out = self.generation(target_out)

        return out







