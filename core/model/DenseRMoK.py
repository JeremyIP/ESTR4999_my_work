import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layer.kanlayer import *
from core.layer.embedding import PatchEmbedding



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.min_val = torch.amin(x, dim=dim2reduce, keepdim=True).detach()
        self.max_val = torch.amax(x, dim=dim2reduce, keepdim=True).detach()

    def _normalize(self, x):
        x = (x - self.min_val) / (self.max_val - self.min_val + self.eps)
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        # Select the statistics for closing price which is at index 3 in the input.
        min_target = self.min_val[..., 3:4]  # Keepdim so that shape broadcasting works.
        max_target = self.max_val[..., 3:4]
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)

        x = x * (max_target - min_target + self.eps) + min_target
        if x.shape[-1] > 1:
            x = x[..., 3:4]

        print(self.min_val.shape)
        return x
        
    def set_statistics(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val


class DenseRMoK(nn.Module):
    def __init__(self, hist_len, pred_len, var_num, KAN_experts_list_01, drop, revin_affine):
        super(DenseRMoK, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.var_num = var_num 
        self.num_experts_selected = sum(KAN_experts_list_01)
        self.KAN_experts_list_01 = KAN_experts_list_01
        self.drop = drop
        self.revin_affine = revin_affine

        self.gate = nn.Linear(self.hist_len, self.num_experts_selected)
        self.softmax = nn.Softmax(dim=-1)

        self.experts = nn.ModuleList([])
        if self.KAN_experts_list_01[0]:
            self.experts.append(TaylorKANLayer(self.hist_len, self.pred_len, order=3, addbias=True))
        if self.KAN_experts_list_01[1]:
            self.experts.append(WaveKANLayer(self.hist_len, self.pred_len, wavelet_type="mexican_hat", device="cuda"))
        if self.KAN_experts_list_01[2]:
            self.experts.append(JacobiKANLayer(self.hist_len, self.pred_len, degree=5))
        if self.KAN_experts_list_01[3]:
            self.experts.append(ChebyKANLayer(self.hist_len, self.pred_len, degree=4))
        if self.KAN_experts_list_01[4]:
            self.experts.append(RBFKANLayer(self.hist_len, self.pred_len, num_centers=10))
        if self.KAN_experts_list_01[5]:
            self.experts.append(NaiveFourierKANLayer(self.hist_len, self.pred_len, gridsize=300))

        '''
            TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
            TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
            TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
            TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
            WaveKANLayer(hist_len, pred_len, wavelet_type="mexican_hat", device="cuda"),
            WaveKANLayer(hist_len, pred_len, wavelet_type="mexican_hat", device="cuda"),


            # TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
            # TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
            #KANInterface(hist_len, pred_len, layer_type="Linear"),
            #KANInterface(hist_len, pred_len, layer_type="Linear")

            #WaveKANLayer(hist_len, pred_len, wavelet_type="mexican_hat", device="cuda"),
            #WaveKANLayer(hist_len, pred_len, wavelet_type="mexican_hat", device="cuda"),
        ])
        '''

        # Modified Module combination 1 - Not good performance
        # self.experts = nn.ModuleList([
        #     TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
        #     JacobiKANLayer(hist_len, pred_len, degree=5),
        #     WaveKANLayer(hist_len, pred_len, wavelet_type="mexican_hat", device="cuda"),
        #     WaveKANLayer(hist_len, pred_len, wavelet_type="morlet", device="cuda"),
        # ])

        # Modified Module combination 2 - Not good performance
        #self.experts = nn.ModuleList([
        #     ChebyKANLayer(hist_len, pred_len, degree=4),
        #     ChebyKANLayer(hist_len, pred_len, degree=5),
        #     RBFKANLayer(hist_len, pred_len, num_centers=10),
        #     RBFKANLayer(hist_len, pred_len, num_centers=15),
        # ])

        # Modified Module combination 3 - Not good performance
        # self.experts = nn.ModuleList([
        #     NaiveFourierKANLayer(hist_len, pred_len, gridsize=300),
        #     WaveKANLayer(hist_len, pred_len, wavelet_type="mexican_hat", device="cuda"),
        #     WaveKANLayer(hist_len, pred_len, wavelet_type="morlet", device="cuda"),
        #     WaveKANLayer(hist_len, pred_len, wavelet_type="dog", device="cuda"),
        # ])

        # Modified Module combination 4 - Not good performance
        # self.experts = nn.ModuleList([
        #     TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
        #     JacobiKANLayer(hist_len, pred_len, degree=6),
        #     ChebyKANLayer(hist_len, pred_len, degree=4),
        #     RBFKANLayer(hist_len, pred_len, num_centers=10),
        # ])

        # Modified Module combination 5 - Crazy not good performance
        # self.experts = nn.ModuleList([
        #     KANLayer(in_dim=hist_len, out_dim=pred_len, num=10, k=3, noise_scale=0.1, scale_base=1.0,
        #         scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1],
        #         sp_trainable=True, sb_trainable=True, device="cuda")
        # ])


        self.dropout = nn.Dropout(self.drop)
        self.rev = RevIN(self.var_num, affine=self.revin_affine)
        self.final_layer = nn.Linear(in_features=self.var_num, out_features=1)

    def forward(self, var_x, marker_x):

        var_x = var_x[..., 0]  # x: [B, Li, N]
        B, L, N = var_x.shape
        var_x = self.rev(var_x, 'norm') if self.rev else var_x
        var_x = self.dropout(var_x).transpose(1, 2).reshape(B * N, L)

        score = F.softmax(self.gate(var_x), dim=-1)  # (BxN, E)
        expert_outputs = torch.stack([self.experts[i](var_x) for i in range(self.num_experts_selected)], dim=-1)  # (BxN, Lo, E)

        prediction = torch.einsum("BLE,BE->BL", expert_outputs, score).reshape(B, N, -1).permute(0, 2, 1)
        prediction = self.final_layer(prediction)

        prediction = self.rev(prediction, 'denorm')

        # ---------- Confidence Estimation ----------
        # Compute variance among expert outputs along the expert dimension.
        # expert_outputs: shape (B*N, pred_len, num_experts)
        expert_variance = torch.var(expert_outputs, dim=-1)  # Shape: (B*N, pred_len)
        # Reshape to (B, N, pred_len) then permute to (B, pred_len, N)
        expert_variance = expert_variance.reshape(B, N, -1).permute(0, 2, 1)
        # Average variance over the N dimension (across series/features) to get a per-time-step measure.
        mean_variance = torch.mean(expert_variance, dim=-1, keepdim=True)  # Shape: (B, pred_len, 1)
        # Transform variance to confidence: lower variance yields higher confidence.
        confidence = 1 / (1 + mean_variance)
        # print(f"confidence shape and value: {confidence.shape} and {confidence}")
        # ---------- End Confidence Estimation ----------


        # print("expert_variance shape: ", expert_variance.shape) 
        # print("expert_outputs shape: ", expert_outputs.shape)
        # print("prediction shape: ", prediction.shape)

        # Apply final layer to reduce output from [B, pred_len, var_num] to [B, pred_len, 1]
        return prediction, confidence

        
        
