import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.pytorch.cell import DCGRUCell, DecoderDCGRUCell
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
  if hard:
      shape = logits.size()
      _, k = y_soft.data.max(-1)
      y_hard = torch.zeros(*shape).to(device)
      y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
      y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
  else:
      y = y_soft
  return y

class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        #self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])
        self.zer = 1

    def forward(self, inputs, adj, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DecoderDCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class GTSModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, temperature, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        self.temperature = temperature
        self.dim_fc = int(model_kwargs.get('dim_fc', False))
        self.embedding_dim = 100
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.hidden_drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot
        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)


    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, label, inputs, node_feas, temp, gumbel_soft, labels=None, batches_seen=None):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        x = node_feas.transpose(1, 0).view(self.num_nodes, 1, -1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        # x = self.hidden_drop(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)

        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        adj = gumbel_softmax(x, temperature=temp, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)
        adj.masked_fill_(mask, 0)
        encoder_hidden_state = self.encoder(inputs, adj)

        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, adj, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )

        return outputs, x.softmax(-1)[:, 0].clone().reshape(self.num_nodes, -1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class GTSContribModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, temperature, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        # self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        self.temperature = temperature
        self.dim_fc = int(model_kwargs.get('dim_fc', False))
        self.embedding_dim = 100
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.hidden_drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.layers = nn.TransformerEncoderLayer(d_model=self.num_nodes * 2, nhead=9)
        self.transformer = nn.TransformerEncoder(self.layers, num_layers=1)
        self.pos_encoder = PositionalEncoding(self.num_nodes * 2)
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot
        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)


    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoded_input = self.pos_encoder(inputs)
        x1 = torch.mm(adj, encoded_input.reshape(207,-1))
        # x = self._concat(x, x1)

        for k in range(2, 3 + 1):
            x2 = 2 * torch.mm(adj, x1)
            x1, x0 = x2, x1
        diffusion = x2
        transformer_output = self.transformer(diffusion.reshape(12,64, 414))
        return transformer_output
                

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            # decoder_input -> 64 , 207
            # adj = 207, 207
            # decoder_hidden_state -> 12, 64, 414
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, label, inputs, node_feas, temp, gumbel_soft, labels=None, batches_seen=None):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        x = node_feas.transpose(1, 0).view(self.num_nodes, 1, -1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        # x = self.hidden_drop(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)

        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        adj = gumbel_softmax(x, temperature=temp, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)
        adj.masked_fill_(mask, 0)
        encoder_hidden_state = self.encoder(inputs, adj)

        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, adj, labels, batches_seen=batches_seen)
        
        self._logger.debug("Decoder complete")
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )

        return outputs, x.softmax(-1)[:, 0].clone().reshape(self.num_nodes, -1)




class SpatialAttention(nn.Module):
    def __init__(self, num_of_timesteps, num_of_features, num_of_vertices):
        super().__init__()
        self.w_1 = nn.Parameter(torch.randn((num_of_timesteps, )))
        self.w_2 = nn.Parameter(torch.randn((num_of_features, num_of_timesteps)))
        self.w_3 = nn.Parameter(torch.randn((num_of_features, )))
        self.b_s = nn.Parameter(torch.randn((1, num_of_vertices, num_of_vertices)))
        self.v_s = nn.Parameter(torch.randn((num_of_vertices, num_of_vertices)))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x.view(64, 207, 2, 12),self.w_1), self.w_2)
        # torch.matmul(torch.matmul((x.permute(0, 3, 2, 1).reshape(x.permute(0, 3, 2, 1).size()[2],-1)).T, self.w_2).reshape(64,13,2),
        #              self.U_2)
        # rhs = torch.matmul(self.w_3, x.permute(2,0,3,1))
        rhs = torch.einsum('n,nvlt->vlt', self.w_3 , x.view( 2,64, 12, 207))

        product = torch.matmul(lhs, rhs)
        S =torch.matmul(self.v_s,
                  F.sigmoid(product + self.b_s)
                     .permute(1, 2, 0)).permute(2, 0, 1)
        S = S - torch.max(S, axis=1, keepdims=True)[0]
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, axis=1, keepdims=True)
        return S_normalized

        # lhs = torch.matmul(torch.matmul((x.permute(0, 3, 2, 1).reshape(x.permute(0, 3, 2, 1).size()[2],-1)).T, self.U_1).reshape(64,13,2),
        #              self.U_2)
        # # shape is (N, V, T)
        # # rhs = torch.matmul(self.U_3, x.permute(2, 0, 1, 3))
        # # rhs = (self.U_3 * x.permute(2, 0, 1, 3)).squeeze(0)
        # rhs = torch.einsum('bnlv,v->bnl', (x.permute(2, 0, 1, 3).reshape(207, 64, 13, 2),self.U_3)).contiguous()

        # product = torch.matmul(lhs.reshape(64,13,207), rhs.reshape(64,207,13))

        # E = torch.matmul(self.V_e,
        #            F.sigmoid(product + self.b_e)
        #              .permute(1, 2, 0)).permute(2, 0, 1)

        # # normailzation
        # E = E - torch.max(E, axis=1, keepdims=True)[0]
        # exp = torch.exp(E)
        # E_normalized = exp / torch.sum(exp, axis=1, keepdims=True)
        # return E_normalized

# X = SpatialAttention(3,1,4)

# a = torch.randn(1,4,1,3)
# X(a)

class cheb_conv_with_SAt(nn.Module):
    '''
    K-order chebyshev graph convolution with Spatial Attention scores
    '''
    def __init__(self, num_of_filters, cheb_polynomials, num_of_features,K = 3, **kwargs):
        '''
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        '''
        super(cheb_conv_with_SAt, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        self.Theta = nn.Parameter(torch.randn(self.K, num_of_features, self.num_of_filters))

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        '''
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros((batch_size, num_of_vertices,
                                     self.num_of_filters), ctx=x.context)
            for k in range(self.K):

                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[k]

                # shape is (batch_size, V, F)
                rhs = torch.matmul(T_k_with_at.T((0, 2, 1)),
                                   graph_signal)

                output = output + torch.matmul(rhs, theta_k)
            outputs.append(output.expand_dims(-1))
        return F.relu(torch.concat(*outputs, dim=-1))



class Temporal_Attention_layer(nn.Module):
    '''
    compute temporal attention scores
    '''
    def __init__(self,num_of_vertices, num_of_features, num_of_timesteps, **kwargs):
        super(Temporal_Attention_layer, self).__init__(**kwargs)
        self.U_1 = nn.Parameter(torch.randn((num_of_vertices, )))
        self.U_2 = nn.Parameter(torch.randn((num_of_features, num_of_vertices)))
        self.U_3 = nn.Parameter(torch.randn((num_of_features, )))
        self.b_e = nn.Parameter(torch.randn((1, num_of_timesteps, num_of_timesteps)))
        self.V_e = nn.Parameter(torch.randn((num_of_timesteps, num_of_timesteps)))

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h
                       shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape


        # compute temporal attention scores
        # shape is (N, T, V)
        lhs = torch.matmul(torch.matmul((x.permute(0, 3, 2, 1).reshape(x.permute(0, 3, 2, 1).size()[2],-1)).T, self.U_1).reshape(64,12,2),
                     self.U_2)
        # shape is (N, V, T)
        # rhs = torch.matmul(self.U_3, x.permute(2, 0, 1, 3))
        # rhs = (self.U_3 * x.permute(2, 0, 1, 3)).squeeze(0)
        rhs = torch.einsum('bnlv,v->bnl', (x.permute(2, 0, 1, 3).reshape(207, 64, 12, 2),self.U_3)).contiguous()

        product = torch.matmul(lhs.reshape(64,12,207), rhs.reshape(64,207, 12))

        E = torch.matmul(self.V_e,
                   F.sigmoid(product + self.b_e)
                     .permute(1, 2, 0)).permute(2, 0, 1)

        # normailzation
        E = E - torch.max(E, axis=1, keepdims=True)[0]
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, axis=1, keepdims=True)
        return E_normalized


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, sp=None, centrality=None):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.spatial_attention = SpatialAttention(12,2, num_nodes)
        self.temporal_attention = Temporal_Attention_layer(num_nodes,2, 12)

        self.t_h = nn.Parameter(torch.empty((12)))
        nn.init.uniform_(self.t_h, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
        self.h_x = nn.Parameter(torch.empty(12, 256, 207))
        nn.init.uniform_(self.h_x,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

        self.t_1 = nn.Parameter(torch.empty(64,207,12))
        nn.init.uniform_(self.t_1,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
        self.t_2 = nn.Parameter(torch.empty(64,1,1,2))
        nn.init.uniform_(self.t_1,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

        # self.sa = nn.MultiheadAttention(32,8)
        # self.t_h_sp = nn.Parameter(torch.empty((207)))
        # nn.init.uniform_(self.t_h_sp, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
        self.h_x_sp = nn.Parameter(torch.empty(64,12,207))
        nn.init.uniform_(self.h_x_sp,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

        self.t_we = nn.Parameter(torch.empty((1)))

        self.s_we = nn.Parameter(torch.empty((1)))
        nn.init.uniform_(self.t_we,0,1)
        nn.init.uniform_(self.s_we,0,1)
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.linear = nn.Linear(220,256)
        self.receptive_field = receptive_field
        self.centrality_linear = nn.Parameter(F.softmax(torch.Tensor(centrality).to(device)).reshape(1,207,1), requires_grad=True)
        self.layer_norm = nn.LayerNorm([256, 207,1])
        # nn.init.constant_(self.centrality_linear, F.softmax(torch.Tensor(centrality)).reshape(1,207,1))



    def forward(self, input):
        tmp = torch.clone(input)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
# NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        x += self.centrality_linear
        # x, _ = self.sa(x.view(64,207*13,32), x.view(64,207*13,32), x.view(64,207*13,32))
        # x = x.view(64, 32, 207, 13)
        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + --> *input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + -------------> *skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)
        x = F.relu(skip)
        x = self.layer_norm(x)
        ttt = self.temporal_attention(tmp).matmul(self.h_x_sp)
        concat = torch.cat((self.spatial_attention(tmp),ttt),1).view(64,207,220)
        linear_out = self.linear(concat).view(64,256,207).unsqueeze(-1)
        # linear_handy = self.linear_hand(1 - F.softmax(self.shotest_path.view(1, 207, 207)))
        x = x + self.t_we * linear_out
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
