##############################
# model.py
##############################
import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from timm.models import swin_base_patch4_window7_224
import torch.nn.functional as F
from attention import SEBlock, CBAM, ECA

class LCA(nn.Module):
    """跨模态关联模块（支持多尺度）"""

    def __init__(self, conv_dim, swin_dim):
        super().__init__()
        # 通道对齐
        self.swin_adapt = nn.Sequential(
            nn.Conv2d(swin_dim, conv_dim, 1),
            nn.BatchNorm2d(conv_dim),
            nn.GELU()
        ) if swin_dim != conv_dim else nn.Identity()

        # 空间注意力机制
        self.conv_q = nn.Conv2d(conv_dim, conv_dim, 1)
        self.conv_kv = nn.Conv2d(conv_dim, conv_dim * 2, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Conv2d(conv_dim, conv_dim, 1)

        # 空间对齐
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, None))

    def forward(self, conv_feat, swin_feat):
        # 对齐处理
        swin_feat = self.adaptive_pool(self.swin_adapt(swin_feat))
        if swin_feat.shape[-2:] != conv_feat.shape[-2:]:
            swin_feat = F.interpolate(swin_feat, conv_feat.shape[-2:], mode='bilinear')

        # 注意力计算
        B, C, H, W = conv_feat.shape
        q = self.conv_q(conv_feat).view(B, C, H * W).permute(0, 2, 1)
        kv = self.conv_kv(swin_feat)
        k, v = torch.chunk(kv, 2, dim=1)

        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).permute(0, 2, 1)

        attn = torch.bmm(q, k) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)

        out = out.permute(0, 2, 1).view(B, C, H, W)
        return conv_feat + self.gamma * self.out_conv(out)

class MedicalNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 初始化ConvNeXt Base
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # 正确划分特征阶段（验证各阶段输出通道）
        self.conv_stages = nn.ModuleList([
            # Stage 0: 初始卷积层 [H/4, 128]
            nn.Sequential(self.convnext.features[0]),
            # Stage 1: 第一个下采样块 [H/8, 256]
            nn.Sequential(
                self.convnext.features[1],
                self.convnext.features[2]
            ),
            # Stage 2: 第二个下采样块 [H/16, 512]
            nn.Sequential(
                self.convnext.features[3],
                self.convnext.features[4]
            )
        ])

        # 初始化Swin Base
        self.swin = swin_base_patch4_window7_224(pretrained=False)
        self.swin.load_state_dict(torch.load("swin-models/swin_base_patch4_window7_224_22k.pth"), strict=False)

        # 调整Swin结构
        self.swin_layers = nn.ModuleList([
            self.swin.layers[0],
            self.swin.layers[1],
            self.swin.layers[2]
        ])

        # 各阶段通道配置（根据实际输出调整）
        self.conv_channels = [128, 256, 512]
        self.swin_channels = [128, 256, 512]

        # 多尺度融合模块（关键修正）
        self.fusions = nn.ModuleList([
            LCA(conv_dim=128, swin_dim=128),  # Stage0对应Swin0
            LCA(conv_dim=256, swin_dim=256),  # Stage1对应Swin1
            LCA(conv_dim=512, swin_dim=512)  # Stage2对应Swin2
        ])

        # 分类头（修正输入维度）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(128 + 256 + 512),  # 修正为实际通道总和
            nn.Linear(128 + 256 + 512, num_classes)
        )

        # 注册钩子保存特征图
        self.feature_maps = []

        def hook_fn(module, input,output):
            self.feature_maps.append(output)

        # 为三个融合层注册钩子
        self.fusions[0].register_forward_hook(hook_fn)
        self.fusions[1].register_forward_hook(hook_fn)
        self.fusions[2].register_forward_hook(hook_fn)

    def forward(self, x):
        self.feature_maps = []  # 清空历史特征

        # ConvNeXt特征提取
        conv_feats = []
        x_conv = x
        # print("\nConvNeXt特征阶段:")
        for i, stage in enumerate(self.conv_stages):
            x_conv = stage(x_conv)
            conv_feats.append(x_conv)
            # print(f"Stage {i}: {x_conv.shape}")

        # Swin特征提取
        x_swin = self.swin.patch_embed(x)
        swin_feats = []
        # print("\nSwin特征阶段:")
        for i, layer in enumerate(self.swin_layers):
            x_swin = layer(x_swin)
            feat = x_swin.permute(0, 3, 1, 2)
            target_size = conv_feats[i].shape[-2:]  # 对齐到对应Conv阶段
            feat = F.interpolate(feat, size=target_size, mode='bilinear')
            swin_feats.append(feat)
            # print(f"Stage {i}: {feat.shape}")

        # 多尺度融合
        fused_features = []
        # print("\n融合过程:")
        for i in range(3):
            # print(f"\nFusion {i + 1}:")
            # print(f"Conv特征尺寸: {conv_feats[i].shape}")
            # print(f"Swin特征尺寸: {swin_feats[i].shape}")

            fused = self.fusions[i](conv_feats[i], swin_feats[i])
            # print(f"融合后尺寸: {fused.shape}")
            fused_features.append(fused)

        # 分类处理
        x = torch.cat([F.adaptive_avg_pool2d(f, 1) for f in fused_features], dim=1)
        return self.classifier(x)


class MedicalNet_SE(nn.Module):
    """SE注意力增强版本"""

    def __init__(self, num_classes):
        super().__init__()
        # 初始化原始结构
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # 重新定义ConvNeXt阶段（插入SE模块）
        self.conv_stages = nn.ModuleList([
            nn.Sequential(
                self.convnext.features[0],
                SEBlock(128)  # 在每阶段后添加SE
            ),
            nn.Sequential(
                self.convnext.features[1],
                self.convnext.features[2],
                SEBlock(256)
            ),
            nn.Sequential(
                self.convnext.features[3],
                self.convnext.features[4],
                SEBlock(512)
            )
        ])

        # 初始化Swin Base
        self.swin = swin_base_patch4_window7_224(pretrained=False)
        self.swin.load_state_dict(torch.load("swin-models/swin_base_patch4_window7_224_22k.pth"), strict=False)

        # 调整Swin结构
        self.swin_layers = nn.ModuleList([
            self.swin.layers[0],
            self.swin.layers[1],
            self.swin.layers[2]
        ])

        # 各阶段通道配置（根据实际输出调整）
        self.conv_channels = [128, 256, 512]
        self.swin_channels = [128, 256, 512]

        # 多尺度融合模块（关键修正）
        self.fusions = nn.ModuleList([
            LCA(conv_dim=128, swin_dim=128),  # Stage0对应Swin0
            LCA(conv_dim=256, swin_dim=256),  # Stage1对应Swin1
            LCA(conv_dim=512, swin_dim=512)  # Stage2对应Swin2
        ])

        # 分类头（修正输入维度）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(128 + 256 + 512),  # 修正为实际通道总和
            nn.Linear(128 + 256 + 512, num_classes)
        )

        # 注册钩子保存特征图
        self.feature_maps = []

        def hook_fn(module, input, output):
            self.feature_maps.append(output)

        # 为三个融合层注册钩子
        self.fusions[0].register_forward_hook(hook_fn)
        self.fusions[1].register_forward_hook(hook_fn)
        self.fusions[2].register_forward_hook(hook_fn)

    def forward(self, x):
        self.feature_maps = []  # 清空历史特征

        # ConvNeXt特征提取
        conv_feats = []
        x_conv = x
        # print("\nConvNeXt特征阶段:")
        for i, stage in enumerate(self.conv_stages):
            x_conv = stage(x_conv)
            conv_feats.append(x_conv)
            # print(f"Stage {i}: {x_conv.shape}")

        # Swin特征提取
        x_swin = self.swin.patch_embed(x)
        swin_feats = []
        # print("\nSwin特征阶段:")
        for i, layer in enumerate(self.swin_layers):
            x_swin = layer(x_swin)
            feat = x_swin.permute(0, 3, 1, 2)
            target_size = conv_feats[i].shape[-2:]  # 对齐到对应Conv阶段
            feat = F.interpolate(feat, size=target_size, mode='bilinear')
            swin_feats.append(feat)
            # print(f"Stage {i}: {feat.shape}")

        # 多尺度融合
        fused_features = []
        # print("\n融合过程:")
        for i in range(3):
            # print(f"\nFusion {i + 1}:")
            # print(f"Conv特征尺寸: {conv_feats[i].shape}")
            # print(f"Swin特征尺寸: {swin_feats[i].shape}")

            fused = self.fusions[i](conv_feats[i], swin_feats[i])
            # print(f"融合后尺寸: {fused.shape}")
            fused_features.append(fused)

        # 分类处理
        x = torch.cat([F.adaptive_avg_pool2d(f, 1) for f in fused_features], dim=1)
        return self.classifier(x)


class MedicalNet_CBAM(nn.Module):
    """CBAM注意力增强版本"""

    def __init__(self, num_classes):
        super().__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # 重新定义ConvNeXt阶段（插入CBAM）
        self.conv_stages = nn.ModuleList([
            nn.Sequential(
                self.convnext.features[0],
                CBAM(128)  # 在每阶段后添加CBAM
            ),
            nn.Sequential(
                self.convnext.features[1],
                self.convnext.features[2],
                CBAM(256)
            ),
            nn.Sequential(
                self.convnext.features[3],
                self.convnext.features[4],
                CBAM(512)
            )
        ])

        # 初始化Swin Base
        self.swin = swin_base_patch4_window7_224(pretrained=False)
        self.swin.load_state_dict(torch.load("swin-models/swin_base_patch4_window7_224_22k.pth"), strict=False)

        # 调整Swin结构
        self.swin_layers = nn.ModuleList([
            self.swin.layers[0],
            self.swin.layers[1],
            self.swin.layers[2]
        ])

        # 各阶段通道配置（根据实际输出调整）
        self.conv_channels = [128, 256, 512]
        self.swin_channels = [128, 256, 512]

        # 多尺度融合模块（关键修正）
        self.fusions = nn.ModuleList([
            LCA(conv_dim=128, swin_dim=128),  # Stage0对应Swin0
            LCA(conv_dim=256, swin_dim=256),  # Stage1对应Swin1
            LCA(conv_dim=512, swin_dim=512)  # Stage2对应Swin2
        ])

        # 分类头（修正输入维度）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(128 + 256 + 512),  # 修正为实际通道总和
            nn.Linear(128 + 256 + 512, num_classes)
        )

        # 注册钩子保存特征图
        self.feature_maps = []

        def hook_fn(module, input, output):
            self.feature_maps.append(output)

        # 为三个融合层注册钩子
        self.fusions[0].register_forward_hook(hook_fn)
        self.fusions[1].register_forward_hook(hook_fn)
        self.fusions[2].register_forward_hook(hook_fn)

    def forward(self, x):
        self.feature_maps = []  # 清空历史特征

        # ConvNeXt特征提取
        conv_feats = []
        x_conv = x
        # print("\nConvNeXt特征阶段:")
        for i, stage in enumerate(self.conv_stages):
            x_conv = stage(x_conv)
            conv_feats.append(x_conv)
            # print(f"Stage {i}: {x_conv.shape}")

        # Swin特征提取
        x_swin = self.swin.patch_embed(x)
        swin_feats = []
        # print("\nSwin特征阶段:")
        for i, layer in enumerate(self.swin_layers):
            x_swin = layer(x_swin)
            feat = x_swin.permute(0, 3, 1, 2)
            target_size = conv_feats[i].shape[-2:]  # 对齐到对应Conv阶段
            feat = F.interpolate(feat, size=target_size, mode='bilinear')
            swin_feats.append(feat)
            # print(f"Stage {i}: {feat.shape}")

        # 多尺度融合
        fused_features = []
        # print("\n融合过程:")
        for i in range(3):
            # print(f"\nFusion {i + 1}:")
            # print(f"Conv特征尺寸: {conv_feats[i].shape}")
            # print(f"Swin特征尺寸: {swin_feats[i].shape}")

            fused = self.fusions[i](conv_feats[i], swin_feats[i])
            # print(f"融合后尺寸: {fused.shape}")
            fused_features.append(fused)

        # 分类处理
        x = torch.cat([F.adaptive_avg_pool2d(f, 1) for f in fused_features], dim=1)
        return self.classifier(x)

class MedicalNet_ECA(nn.Module):
    """ECA注意力增强版本"""

    def __init__(self, num_classes):
        super().__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # 重新定义ConvNeXt阶段（插入ECA）
        self.conv_stages = nn.ModuleList([
            nn.Sequential(
                self.convnext.features[0],
                ECA(128)  # 在每阶段后添加ECA
            ),
            nn.Sequential(
                self.convnext.features[1],
                self.convnext.features[2],
                ECA(256)
            ),
            nn.Sequential(
                self.convnext.features[3],
                self.convnext.features[4],
                ECA(512)
            )
        ])


        # 初始化Swin Base
        self.swin = swin_base_patch4_window7_224(pretrained=False)
        self.swin.load_state_dict(torch.load("swin-models/swin_base_patch4_window7_224_22k.pth"), strict=False)

        # 调整Swin结构
        self.swin_layers = nn.ModuleList([
            self.swin.layers[0],
            self.swin.layers[1],
            self.swin.layers[2]
        ])

        # 各阶段通道配置（根据实际输出调整）
        self.conv_channels = [128, 256, 512]
        self.swin_channels = [128, 256, 512]

        # 多尺度融合模块（关键修正）
        self.fusions = nn.ModuleList([
            LCA(conv_dim=128, swin_dim=128),  # Stage0对应Swin0
            LCA(conv_dim=256, swin_dim=256),  # Stage1对应Swin1
            LCA(conv_dim=512, swin_dim=512)  # Stage2对应Swin2
        ])

        # 分类头（修正输入维度）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(128 + 256 + 512),  # 修正为实际通道总和
            nn.Linear(128 + 256 + 512, num_classes)
        )

        # 注册钩子保存特征图
        self.feature_maps = []

        def hook_fn(module, input, output):
            self.feature_maps.append(output)

        # 为三个融合层注册钩子
        self.fusions[0].register_forward_hook(hook_fn)
        self.fusions[1].register_forward_hook(hook_fn)
        self.fusions[2].register_forward_hook(hook_fn)

    def forward(self, x):
        self.feature_maps = []  # 清空历史特征

        # ConvNeXt特征提取
        conv_feats = []
        x_conv = x
        # print("\nConvNeXt特征阶段:")
        for i, stage in enumerate(self.conv_stages):
            x_conv = stage(x_conv)
            conv_feats.append(x_conv)
            # print(f"Stage {i}: {x_conv.shape}")

        # Swin特征提取
        x_swin = self.swin.patch_embed(x)
        swin_feats = []
        # print("\nSwin特征阶段:")
        for i, layer in enumerate(self.swin_layers):
            x_swin = layer(x_swin)
            feat = x_swin.permute(0, 3, 1, 2)
            target_size = conv_feats[i].shape[-2:]  # 对齐到对应Conv阶段
            feat = F.interpolate(feat, size=target_size, mode='bilinear')
            swin_feats.append(feat)
            # print(f"Stage {i}: {feat.shape}")

        # 多尺度融合
        fused_features = []
        # print("\n融合过程:")
        for i in range(3):
            # print(f"\nFusion {i + 1}:")
            # print(f"Conv特征尺寸: {conv_feats[i].shape}")
            # print(f"Swin特征尺寸: {swin_feats[i].shape}")

            fused = self.fusions[i](conv_feats[i], swin_feats[i])
            # print(f"融合后尺寸: {fused.shape}")
            fused_features.append(fused)

        # 分类处理
        x = torch.cat([F.adaptive_avg_pool2d(f, 1) for f in fused_features], dim=1)
        return self.classifier(x)


class MedicalNet_Single_lca(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 初始化ConvNeXt Base
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # 正确划分特征阶段（验证各阶段输出通道）
        self.conv_stages = nn.ModuleList([
            # Stage 0: 初始卷积层 [H/4, 128]
            nn.Sequential(self.convnext.features[0]),
            # Stage 1: 第一个下采样块 [H/8, 256]
            nn.Sequential(
                self.convnext.features[1],
                self.convnext.features[2]
            ),
            # Stage 2: 第二个下采样块 [H/16, 512]
            nn.Sequential( 
                self.convnext.features[3],
                self.convnext.features[4]
            )
        ])

        # 初始化Swin Base
        self.swin = swin_base_patch4_window7_224(pretrained=False)
        self.swin.load_state_dict(torch.load("swin-models/swin_base_patch4_window7_224_22k.pth"), strict=False)

        # 调整Swin结构
        self.swin_layers = nn.ModuleList([
            self.swin.layers[0],
            self.swin.layers[1],
            self.swin.layers[2]
        ])

        # 各阶段通道配置（根据实际输出调整）
        self.conv_channels = [128, 256, 512]
        self.swin_channels = [128, 256, 512]

        # 多尺度融合模块（关键修正）- 只使用第1阶段
        self.fusions = nn.ModuleList([
            LCA(conv_dim=256, swin_dim=256),  # 只使用Stage1对应Swin1
        ])

        # 分类头（修正输入维度）- 只使用第1阶段的256通道
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(256),  # 修正为实际通道数：256
            nn.Linear(256, num_classes)
        )

        # 注册钩子保存特征图
        self.feature_maps = []

        def hook_fn(module, input, output):
            self.feature_maps.append(output)

        # 为融合层注册钩子
        self.fusions[0].register_forward_hook(hook_fn)

    def forward(self, x):
        self.feature_maps = []  # 清空历史特征

        # ConvNeXt特征提取
        conv_feats = []
        x_conv = x
        for i, stage in enumerate(self.conv_stages):
            x_conv = stage(x_conv)
            conv_feats.append(x_conv)

        # Swin特征提取
        x_swin = self.swin.patch_embed(x)
        swin_feats = []
        for i, layer in enumerate(self.swin_layers):
            x_swin = layer(x_swin)
            feat = x_swin.permute(0, 3, 1, 2)
            target_size = conv_feats[i].shape[-2:]  # 对齐到对应Conv阶段
            feat = F.interpolate(feat, size=target_size, mode='bilinear')
            swin_feats.append(feat)

        # 多尺度融合 - 只融合第1阶段
        fused_features = []
        fused = self.fusions[0](conv_feats[1], swin_feats[1])
        fused_features.append(fused)

        # 分类处理
        x = torch.cat([F.adaptive_avg_pool2d(f, 1) for f in fused_features], dim=1)
        return self.classifier(x)

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(SwinClassifier, self).__init__()
        self.swin = swin_base_patch4_window7_224(pretrained=False)
        # 加载预训练权重
        self.swin.load_state_dict(torch.load("swin-models/swin_base_patch4_window7_224_22k.pth"), strict=False)

        # 删除Swin Transformer的默认头部
        self.swin.head = nn.Identity()  # 将头部替换为恒等层

        # 获取Swin Transformer的输出特征维度
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.swin(dummy_input)
        feature_dim = features.view(-1).shape[0]

        # 添加自定义的分类头部
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 前向传播到Swin Transformer
        x = self.swin(x)
        # 压缩维度
        x = x.view(x.size(0), -1)
        # 分类
        x = self.classifier(x)
        return x

class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNeXtClassifier, self).__init__()
        # 初始化ConvNeXt模型并加载预训练权重
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # 替换分类头部
        # 获取原始头部的输入特征维度
        in_features = self.convnext.classifier[2].in_features
        # 替换为自定义的分类头部
        self.convnext.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.convnext(x)

class SwinClassifierBase(nn.Module):
    def __init__(self):
        super(SwinClassifierBase, self).__init__()
        self.swin = swin_base_patch4_window7_224(pretrained=False)
        self.swin.load_state_dict(torch.load("swin-models/swin_base_patch4_window7_224_22k.pth"), strict=False)
        self.swin.head = nn.Identity()  # 删除默认头部

    def forward(self, x):
        x = self.swin(x)
        x = x.view(x.size(0), -1)  # 压缩维度
        return x

class ConvNeXtClassifierBase(nn.Module):
    def __init__(self):
        super(ConvNeXtClassifierBase, self).__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.convnext.classifier[2] = nn.Identity()  # 删除默认头部

    def forward(self, x):
        x = self.convnext(x)
        return x

class FusionClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(FusionClassifier, self).__init__()
        # 初始化Swin和ConvNeXt模型
        self.swin = SwinClassifierBase()
        self.convnext = ConvNeXtClassifierBase()

        # 获取Swin和ConvNeXt的输出特征维度
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            swin_features = self.swin(dummy_input)
            convnext_features = self.convnext(dummy_input)

        # 获取特征维度
        swin_feature_dim = swin_features.shape[1]
        convnext_feature_dim = convnext_features.shape[1]
        total_feature_dim = swin_feature_dim + convnext_feature_dim

        # 添加自定义的分类头部
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 提取Swin和ConvNeXt的特征
        swin_features = self.swin(x)
        convnext_features = self.convnext(x)

        # 拼接特征
        fused_features = torch.cat((swin_features, convnext_features), dim=1)

        # 分类
        output = self.classifier(fused_features)
        return output


# 测试代码
if __name__ == "__main__":
    # model = MedicalNet_ECA(num_classes=5)
    # dummy_input = torch.randn(2, 3, 224, 224)
    # print("\n模型测试:")
    # output = model(dummy_input)
    # print("\n最终输出尺寸:", output.shape)

    # model = SwinClassifier(num_classes=5)
    # dummy_input = torch.randn(2, 3, 224, 224)
    # output = model(dummy_input)
    # print("\n最终输出尺寸:", output.shape)

    # model = ConvNeXtClassifier(num_classes=5)
    #
    # # 测试模型输出
    # dummy_input = torch.randn(2, 3, 224, 224)
    # output = model(dummy_input)
    # print("\n最终输出尺寸:", output.shape)

    model = FusionClassifier(num_classes=5)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("\n最终输出尺寸:", output.shape)

