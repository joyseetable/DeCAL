import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    一个经典的、带有瓶颈结构的 Adapter 模块。
    它包含一个降维线性层，一个非线性激活，和一个升维线性层。
    """
    def __init__(self,
                 input_dim: int,
                 bottleneck_dim: int,
                 output_dim: int = None,
                 activation: nn.Module = nn.GELU):
        """
        Args:
            input_dim (int): 输入特征的维度。
            bottleneck_dim (int): 瓶颈层的中间维度，这个值越小，参数量越少。
            output_dim (int, optional): 输出特征的维度。如果为 None，则默认为 input_dim。
            activation (nn.Module, optional): 非线性激活函数。默认为 nn.GELU。
        """
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
            
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.activation = activation()
        self.up_project = nn.Linear(bottleneck_dim, output_dim)
        
        # 对新添加的层进行初始化，这很重要
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.weight) # 关键：将 up_project 的权重初始化为0
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        # 原始特征 (残差连接)
        residual = x
        
        # 通过 Adapter
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        
        # 添加残差连接
        return x + residual
    

class AttentionFusion(nn.Module):
    """
        一个基于文本特征的注意力融合模块。
        它学习如何根据给定的文本查询，为多个图像特征源分配权重。
    """
    def __init__(self, embed_dim: int, num_sources: int = 3):
        """
            Args:
                embed_dim (int): 文本和图像特征的维度。
                num_sources (int): 要融合的图像特征源的数量 (例如, global, entity, background -> 3)。
        """
        super().__init__()
            # 这个小型的 MLP 将是可训练的，它就是我们“智能融合”的大脑
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), # 降维以减少参数
            nn.ReLU(),
            
            nn.Linear(embed_dim // 2, num_sources),
            nn.Softmax(dim=-1) # 输出一组和为 1 的权重
        )
        print(f"✨ 初始化 AttentionFusion 模块，将融合 {num_sources} 个特征源。")

    def forward(self, text_features, image_features_list):
        """
            Args:
                text_features (Tensor): 形状为 (Batch, Embed_Dim) 的文本特征。
                image_features_list (list[Tensor]): 一个包含多个图像特征张量的列表，
                                                    每个张量的形状都是 (Batch, Embed_Dim)。
        """
            # (Batch, Num_Sources, Embed_Dim)
        stacked_image_features = torch.stack(image_features_list, dim=1)
            
            # (Batch, Num_Sources) -> (Batch, 1, Num_Sources)
            # attention_net 根据文本内容，为每个图像特征源生成一个权重
        weights = self.attention_net(text_features).unsqueeze(1)
            
            # 加权平均: (Batch, 1, Num_Sources) @ (Batch, Num_Sources, Embed_Dim) -> (Batch, 1, Embed_Dim)
            # torch.bmm 是批量矩阵乘法
        fused_features = torch.bmm(weights, stacked_image_features)
            
            # (Batch, Embed_Dim)
        return fused_features.squeeze(1)