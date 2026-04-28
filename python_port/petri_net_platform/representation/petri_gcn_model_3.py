from typing import List
import torch
import torch.nn.functional as f
from torch import nn

"""
    Petri 网上的 GCN 模型与状态编码组件.
    加入 LayerNorm 与 Residual 残差连接, 防止强化学习中的特征爆炸与 Batch 干扰.
    【全新升级】：引入 8 维高级特征，专为大型高约束 (HQ) 调度场景设计。
"""

class PetriNetGCNEnhanced(nn.Module):
    """增强版 PetriNet-GCN, 支持动态图拓扑输入与全局泛化。"""
    def __init__(self, pre: List[List[int]], post: List[List[int]], lambda_p: int, lambda_t: int, extra_p2t_rounds: int):
        super().__init__()
        self.pre = None
        self.post = None
        self.pre_t = None
        self.post_t = None
        
        # 【核心修改】：输入维度从 4 改为 8，接收更加丰富的特征
        self.p2p = nn.Linear(8, lambda_p)
        
        self.p2t_pre = nn.Linear(lambda_p, lambda_t, bias=False)
        self.p2t_post = nn.Linear(lambda_p, lambda_t, bias=False)
        self.t2p_post = nn.Linear(lambda_t, lambda_p, bias=False)
        self.t2p_pre = nn.Linear(lambda_t, lambda_p, bias=False)
        
        self.extra_t2p_post = nn.ModuleList([nn.Linear(lambda_t, lambda_p, bias=False) for _ in range(extra_p2t_rounds)])
        self.extra_t2p_pre = nn.ModuleList([nn.Linear(lambda_t, lambda_p, bias=False) for _ in range(extra_p2t_rounds)])
        self.extra_p2t_pre = nn.ModuleList([nn.Linear(lambda_p, lambda_t, bias=False) for _ in range(extra_p2t_rounds)])
        self.extra_p2t_post = nn.ModuleList([nn.Linear(lambda_p, lambda_t, bias=False) for _ in range(extra_p2t_rounds)])
        
        self.ln_p_init = nn.LayerNorm(lambda_p)
        self.ln_t_main = nn.LayerNorm(lambda_t)
        self.ln_p_main = nn.LayerNorm(lambda_p)
        self.extra_ln_p = nn.ModuleList([nn.LayerNorm(lambda_p) for _ in range(extra_p2t_rounds)])
        self.extra_ln_t = nn.ModuleList([nn.LayerNorm(lambda_t) for _ in range(extra_p2t_rounds)])
        
        self.out = nn.Linear(lambda_t, 1)

        if pre is not None and post is not None:
            self.set_graph(pre, post)

    def set_graph(self, pre: List[List[int]], post: List[List[int]]):
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.pre = torch.tensor(pre, dtype=torch.float32, device=device)
        self.post = torch.tensor(post, dtype=torch.float32, device=device)
        self.pre_t = self.pre.transpose(0, 1)
        self.post_t = self.post.transpose(0, 1)

    def _p2t(self, h_p: torch.Tensor, linear_pre: nn.Linear, linear_post: nn.Linear) -> torch.Tensor:
        h1 = linear_pre(h_p)
        h2 = linear_post(h_p)
        return torch.einsum("tp,bpl->btl", self.pre_t, h1) + torch.einsum("tp,bpl->btl", self.post_t, h2)

    def _t2p(self, h_t: torch.Tensor) -> torch.Tensor:
        h1 = self.t2p_post(h_t)
        h2 = self.t2p_pre(h_t)
        return torch.einsum("pt,btl->bpl", self.post, h1) + torch.einsum("pt,btl->bpl", self.pre, h2)

    def _t2p_with_layers(self, h_t: torch.Tensor, linear_post: nn.Linear, linear_pre: nn.Linear) -> torch.Tensor:
        h1 = linear_post(h_t)
        h2 = linear_pre(h_t)
        return torch.einsum("pt,btl->bpl", self.post, h1) + torch.einsum("pt,btl->bpl", self.pre, h2)

    def extract_features(self, x_p: torch.Tensor):
        single = x_p.dim() == 2
        if single:
            x_p = x_p.unsqueeze(0)
            
        h_p = f.relu(self.ln_p_init(self.p2p(x_p)))
        h_t = f.relu(self.ln_t_main(self._p2t(h_p, self.p2t_pre, self.p2t_post)))
        h_p = f.relu(self.ln_p_main(self._t2p(h_t)))
        
        for idx in range(len(self.extra_p2t_pre)):
            h_p_new = self._t2p_with_layers(h_t, self.extra_t2p_post[idx], self.extra_t2p_pre[idx])
            h_p = f.relu(self.extra_ln_p[idx](h_p_new + h_p)) 
            
            h_t_new = self._p2t(h_p, self.extra_p2t_pre[idx], self.extra_p2t_post[idx])
            h_t = f.relu(self.extra_ln_t[idx](h_t_new + h_t)) 
            
        return h_t, single

    def forward(self, x_p: torch.Tensor) -> torch.Tensor:
        h_t, single = self.extract_features(x_p)
        q = self.out(h_t).squeeze(-1)
        if single:
            return q.squeeze(0)
        return q


class PetriStateEncoderEnhanced:
    """
    提取 8 维状态特征，赋予网络对“危险死线”和“资源瓶颈”的强感知能力。
    """
    def __init__(self, end: List[int], min_delay_p: List[int], max_residence_time: List[int], capacity: List[int], device: torch.device):
        self.device = device
        self.set_context(end, min_delay_p, max_residence_time, capacity)
        
    def set_context(self, end: List[int], min_delay_p: List[int], max_residence_time: List[int], capacity: List[int]):
        """切换环境时更新底层环境参数"""
        self.end = end
        self.min_delay_p = min_delay_p
        self.max_residence_time = max_residence_time
        
        # 很多网文件可能没有容量限制，如果没有传入，则默认为无限大 (用 99.0 替代)
        if capacity is None or len(capacity) == 0:
            self.capacity = [99.0] * len(min_delay_p)
        else:
            self.capacity = capacity

    def encode(self, marking) -> torch.Tensor:
        p_info = list(marking.get_p_info())
        rows = []
        for idx, token in enumerate(p_info):
            goal = self.end[idx] if self.end[idx] != -1 else token
            timer = self._get_place_timer(marking, idx)
            min_t = self.min_delay_p[idx]
            max_t = self.max_residence_time[idx]
            cap = self.capacity[idx]

            # --------------------
            # 新增特征处理逻辑
            # --------------------
            # 1. 约束掩码：如果 max_t 小于 2^31-1，说明是危险腔室
            is_constrained = 1.0 if max_t < 2000000000 else 0.0
            
            # 2. 安全死线：过滤掉 2^31-1 这种会让神经网络梯度爆炸的天文数字
            safe_max_t = float(max_t) if is_constrained else 0.0
            
            # 3. 紧迫度 (Urgency Margin)：计算距离晶圆报废还剩多少时间。
            # 越小越危险！如果 unconstrained，则保持为 0，防止干扰。
            # 如果是有 token 并且受限的腔室，计算剩余安全时间。
            if is_constrained and token > 0:
                urgency_margin = float(max_t - timer)
            else:
                urgency_margin = 0.0
                
            # 4. 容量剩余率：如果库所满了，禁止往前走
            capacity_margin = float(cap - token)

            rows.append([
                float(token),           # F1: 当前数量
                float(goal),            # F2: 目标数量
                float(timer),           # F3: 已驻留时间
                float(min_t),           # F4: 最少加工时间
                safe_max_t,             # F5: 【新】最大容忍时间
                is_constrained,         # F6: 【新】是否是高危腔室 (Attention 掩码)
                capacity_margin,        # F7: 【新】剩余容量裕度
                urgency_margin          # F8: 【新】距离报废的紧迫程度
            ])
            
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _get_place_timer(self, marking, place: int) -> float:
        if hasattr(marking, "get_time"):
            try:
                return float(marking.get_time(place))
            except Exception: 
                pass
        if hasattr(marking, "t_info"):
            total = 0.0
            for item in marking.t_info[place]:
                if hasattr(item, "timer"):
                    total += float(item.timer)
                else:
                    total += float(item)
            return total
        return 0.0