import torch 
import torch.nn as nn
import sys 
import random
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import os
import math

sys.path.insert(0, "../../")
from pc_processor.layers import PositionEmbeddingNerf
from pc_processor.models.vision.resnet import ResNet
from pc_processor.models.vision.efficientnet import EfficientNet
from pc_processor.models.vision.swin import SwinTransformer
from pc_processor.models.salsanext import ResContextBlock

class RGBDecoder(nn.Module):
    def __init__(self, in_channels=[], nclasses=4, base_channels=32):
        super(RGBDecoder, self).__init__()
        self.up_4a = nn.Sequential(
            nn.Conv2d(in_channels[3], base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        
        self.up_3a = nn.Sequential(
            nn.Conv2d(in_channels[2] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")

        )
        self.up_2a = nn.Sequential(
            nn.Conv2d(in_channels[1] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.up_1a = nn.Sequential(
            nn.Conv2d(in_channels[0] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(base_channels, nclasses, kernel_size=3, padding=1)
        )

        self.depth_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        )

        self.rgb_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        up_4a = self.up_4a(inputs[3])
        up_3a = self.up_3a(torch.cat((up_4a, inputs[2]), dim=1))
        up_2a = self.up_2a(torch.cat((up_3a, inputs[1]), dim=1))
        up_1a = self.up_1a(torch.cat((up_2a, inputs[0]), dim=1))
        cls_pred = self.cls_head(up_1a).softmax(dim=1)
        depth_pred = self.depth_head(up_1a).sigmoid()
        rgb_pred = self.rgb_head(up_1a).sigmoid()
        
        return torch.cat((cls_pred, depth_pred, rgb_pred), dim=1) # num_classes + 1 + 3
        
        
class CrossAttention(nn.Module):
    def __init__(self, emb_dims=384, num_heads=4, dropout=0.2, mlp_ratio=1, use_mlp=True) -> None:
        super(CrossAttention, self).__init__()

        self.use_mlp = use_mlp
        self.attn_layer = nn.MultiheadAttention(
            emb_dims, num_heads, batch_first=True, dropout=0.2)
        self.attn_norm = nn.LayerNorm(emb_dims)
        if use_mlp:
            self.ffn = nn.Sequential(
                nn.Linear(emb_dims, emb_dims*mlp_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dims*mlp_ratio, emb_dims),
            )
            self.ffn_norm = nn.LayerNorm(emb_dims)
        
    def forward(self, q_feature, kv_feature):
        attn_out, _ = self.attn_layer(q_feature, kv_feature, value=kv_feature)
        attn_out = self.attn_norm(attn_out + q_feature)
        if self.use_mlp:
            attn_out = self.ffn_norm(attn_out + self.ffn(attn_out))
        return attn_out


class BevContextBlock(nn.Module):
    def __init__(self, internel_channels=64, emb_dims=384, voxels_per_pillar=32) -> None:
        super(BevContextBlock, self).__init__()
        self.emb_dims = emb_dims
        self.voxels_per_pillar = voxels_per_pillar
        self.internel_channels = internel_channels
        self.mid_channels = internel_channels*voxels_per_pillar

        self.bev2voxels = nn.Sequential(
            ResContextBlock(emb_dims, emb_dims),
            ResContextBlock(emb_dims, emb_dims),
            nn.Conv2d(emb_dims, self.mid_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, bev_feature):
        # B C H W
        bev_feature = F.interpolate(bev_feature, scale_factor=2)
        voxel_feature = self.bev2voxels(bev_feature)
        B, C, H, W = voxel_feature.size()
        voxel_feature = voxel_feature.view(
            B, C//self.voxels_per_pillar, self.voxels_per_pillar, H, W) # return B C L H W
        return voxel_feature

class QueryDecoder(nn.Module):
    def __init__(self, num_classes, pc_range, 
                 internal_dims=64, use_rgb=False) -> None:
        super(QueryDecoder, self).__init__()
        
        self.internal_dims = internal_dims
        self.pc_range = pc_range 
        self.use_rgb = use_rgb
        if use_rgb:
            self.head = nn.Conv3d(in_channels=internal_dims, out_channels=3+num_classes, kernel_size=1, padding=0)
        else:
            self.head = nn.Conv3d(in_channels=internal_dims, out_channels=num_classes, kernel_size=1, padding=0)

        
    def forward(self, voxel_feature, query_pose):
        # voxel_feature: B C L, H, W
        voxel_preds = self.head(voxel_feature)
        grid = query_pose.clone() # x,y,z -> l, w, h
        grid[..., 0] = (grid[..., 0] - self.pc_range[0])/(self.pc_range[1] - self.pc_range[0]) # normalize to [0, 1] x l
        grid[..., 1] = (grid[..., 1] - self.pc_range[2])/(self.pc_range[3] - self.pc_range[2]) # normalize to [0, 1] y w
        grid[..., 2] = (grid[..., 2] - self.pc_range[4])/(self.pc_range[5] - self.pc_range[4]) # normalize to [0, 1] z h
        grid = grid.clamp(min=0, max=1)
        grid = grid * 2.0 - 1.0 # normalize to [-1, 1] 

        # B, C, L, H, W = voxel_feature.size()
        voxel_feature = voxel_preds.permute(0, 1, 2, 4, 3).contiguous() # B, C, L, W, H # x y z == L, W, H
        grid = grid.unsqueeze(2).unsqueeze(2) # B, N, 1, 1, 3
        query_preds= F.grid_sample(
            voxel_feature, grid=grid, align_corners=False
            ).squeeze(3).squeeze(3).permute(0, 2, 1).contiguous() # B, C, N, 1, 1 -> B, N, C
        
        if self.use_rgb:
            query_preds[..., :3] = query_preds[..., :3].sigmoid()
            return query_preds, voxel_preds[:, 3:]
        else:
            return query_preds, voxel_preds
    
class PV2BEVProjection(nn.Module):
    def __init__(self, bev_range=[0, 51.2, -25.6, 25.6], 
                 grid_size=0.4, emb_dims=384, num_heads=4, 
                 num_layers=1):
        super(PV2BEVProjection, self).__init__()

        self.bev_range = bev_range
        self.grid_size = grid_size

        self.bev_h = int((bev_range[1] - bev_range[0])/grid_size)
        self.bev_w = int((bev_range[3] - bev_range[2])/grid_size)
        
        self.xy_query = nn.Parameter(
            self.getBevPosEmb(bev_h=self.bev_h, bev_w=self.bev_w, grid_size=grid_size), requires_grad=False)

        self.pos_embed = nn.Parameter(torch.randn(1, self.bev_h*self.bev_w, emb_dims))
        trunc_normal_(self.pos_embed, std=.02)

        self.embed_layer = nn.Sequential(
            PositionEmbeddingNerf(
                num_pos_feats=emb_dims//2, temperature=1000),
            nn.Flatten(2),
            nn.Linear(emb_dims, emb_dims),
            nn.LayerNorm(emb_dims),
        )

        self.self_attn_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.self_attn_layers.append(CrossAttention(
                emb_dims=emb_dims, num_heads=num_heads, use_mlp=True
            ))
            self.cross_attn_layers.append(
                CrossAttention(emb_dims=emb_dims, num_heads=num_heads, use_mlp=True)
            )

    def getBevPosEmb(self, bev_h, bev_w, grid_size):
        x_query = torch.arange(0, bev_h, step=1) * grid_size + self.bev_range[0] + grid_size/2.
        y_query = torch.arange(0, bev_w, step=1) * grid_size + self.bev_range[2] + grid_size/2.
        xy_query = torch.cat(
            (x_query.unsqueeze(1).repeat(1, bev_w).unsqueeze(2),
             y_query.unsqueeze(0).repeat(bev_h, 1).unsqueeze(2)),
             dim=2)
        xy_query = xy_query.view(-1, 2) # HW, 2
        return xy_query

    def forward(self, pv_feature, bev_feature=None):
        B = pv_feature.size(0)
        
        if bev_feature is not None:
            B, C, _, _ = bev_feature.size()
            bev_feature = F.interpolate(bev_feature, size=(self.bev_h, self.bev_w), mode="bilinear")
            query = bev_feature.view(B, C, self.bev_h*self.bev_w).permute(0, 2, 1) + self.pos_embed
        else:
            xy_query = self.xy_query.unsqueeze(0).repeat(B, 1, 1)
            query = self.embed_layer(xy_query) + self.pos_embed

        for i in range(len(self.self_attn_layers)):
            query = self.self_attn_layers[i](q_feature=query, kv_feature=query)
            query = self.cross_attn_layers[i](q_feature=query, kv_feature=pv_feature)
        
        out = query
        B, _, C = out.size()
        out_bev = out.view(B, self.bev_h, self.bev_w, C).permute(0, 3, 1, 2).contiguous()
        return out_bev
       
class PointcloudEncoder(nn.Module):
    def __init__(self, bev_range, grid_size, emb_dims=384, num_heads=4) -> None:
        super(PointcloudEncoder, self).__init__()

        pcd_channels = 6
        num_pos_feats = (emb_dims//pcd_channels)*pcd_channels
        self.embed_layer = nn.Sequential(
            PositionEmbeddingNerf(num_pos_feats=num_pos_feats//pcd_channels, temperature=1000),
            nn.Flatten(2), 
            nn.Linear(emb_dims, emb_dims),
            nn.LayerNorm(emb_dims),
        )
        self.projection_voxels_layer = PV2BEVProjection(
            bev_range=bev_range,
            grid_size=grid_size,
            emb_dims=emb_dims, 
            num_heads=num_heads
        )
    
    def forward(self, pcd_voxels, bev_feature=None):
        # pcd features: xyzid
        pcd_voxel_feature = self.embed_layer(pcd_voxels)
        out_bev = self.projection_voxels_layer(pcd_voxel_feature, bev_feature) 
        return out_bev

class RadarEncoder(nn.Module):
    def __init__(self, bev_range, grid_size, emb_dims=384, num_heads=4) -> None:
        super(RadarEncoder, self).__init__()

        self.num_radar = 5
        pcd_channels = 4
        num_pos_feats = (emb_dims//pcd_channels)*pcd_channels
        self.embed_layer = nn.Sequential(
            PositionEmbeddingNerf(num_pos_feats=num_pos_feats//pcd_channels, temperature=1000),
            nn.Flatten(2), 
            nn.Linear(emb_dims, emb_dims),
            nn.LayerNorm(emb_dims),
        )
        self.projection_voxels_layer = PV2BEVProjection(
            bev_range=bev_range,
            grid_size=grid_size,
            emb_dims=emb_dims, 
            num_heads=num_heads
        )
        self.pos_embed = nn.Parameter(
        torch.randn(1, self.num_radar, 1, emb_dims) * .02)
        trunc_normal_(self.pos_embed, std=.02)

    
    def forward(self, radar_points, bev_feature=None):
        # pcd features: x y vx vy
        B, L, N, C = radar_points.size()
        radar_points = radar_points.view(B*L, N, C)
        radar_feature = self.embed_layer(radar_points)
        radar_feature = radar_feature.view(B, L, N, -1) + self.pos_embed.repeat(B, 1, N, 1)
        radar_feature = radar_feature.view(B, L*N, -1)
        out_bev = self.projection_voxels_layer(radar_feature, bev_feature) 
        return out_bev

class AttentionFeatureMerge(nn.Module):
    def __init__(self, 
                 in_channels=2048, 
                 out_channels=384, 
                 num_patches=None,
                 drop_rate=0.2,
                 emb_dims=384,
                 num_heads=4) -> None:
        super(AttentionFeatureMerge, self).__init__()
        assert num_patches is not None

        self.dropout = nn.Dropout(p=drop_rate)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        ) 
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        self.self_attn = CrossAttention(emb_dims=emb_dims, num_heads=num_heads)    

        self.single_view_embed = nn.Parameter(
                torch.randn(1, num_patches, emb_dims) * .02)
        
        trunc_normal_(self.single_view_embed, std=.02)

    def forward(self, features):
        out = self.conv(self.dropout(features[-1]))
        out = out*self.attention(out)
        out = out.flatten(2).permute(0, 2, 1).contiguous()
        out = self.self_attn(out+self.single_view_embed, out)
        return out
    
class REONet(nn.Module):
    def __init__(self, backbone="resnet34", 
        img_size=(640, 960), patch_size=16, num_classes=20, 
        num_sem_classes=20,
        pc_range=[0, 51.2, -25.6, 25.6, -2, 4.4],
        grid_size=0.2,
        voxel_downscales=4,
        num_images=1, 
        empty_idx=0,
        imgnet_pretrained=True,
        pretrained_model_root="/path/to/pretrained_model/",
        use_pcd=False, 
        use_radar=False,
        use_image=True) -> None:
        super(REONet, self).__init__()

        self.backbone = backbone
        self.num_images = num_images
        self.use_pcd = use_pcd
        self.use_image = use_image
        self.use_radar = use_radar
        self.empty_idx = empty_idx
        
        self.num_classes = num_classes
        self.num_sem_classes = num_sem_classes
        self.grid_size = grid_size
        if (not self.use_image) and (not self.use_pcd):
            raise ValueError("both use_image and use_pcd are False, please check the code")
        
        emb_dims = 384
        num_heads = 4
        patch_size = 32
        self.pc_range = pc_range
        self.voxel_size = [
            int((pc_range[1] - pc_range[0])/grid_size),
            int((pc_range[3] - pc_range[2])/grid_size),
            int((pc_range[5] - pc_range[4])/grid_size)
        ]
        self.voxels_per_pillar = self.voxel_size[2] // 2

        if imgnet_pretrained and use_image:
            assert os.path.isdir(pretrained_model_root), "path not found: {}".format(pretrained_model_root)
        if self.backbone == "resnet18":
            self.feature_extractor = ResNet(backbone="resnet18", pretrained=False)
            pretrained_model_pth = os.path.join(pretrained_model_root, "resnet18-f37072fd.pth")

            img_channels = [64, 128, 256, 512]
        elif self.backbone == "resnet34":
            self.feature_extractor = ResNet(backbone="resnet34", pretrained=False)
            pretrained_model_pth = os.path.join(pretrained_model_root, "resnet34-b627a593.pth")
            img_channels = [64, 128, 256, 512]

        elif self.backbone == "resnet50":
            self.feature_extractor = ResNet(backbone="resnet50", pretrained=False)
            pretrained_model_pth = os.path.join(pretrained_model_root, "resnet50-0676ba61.pth")
            img_channels = [256, 512, 1024, 2048]

        elif self.backbone == "resnet50_wide":
            self.feature_extractor = ResNet(backbone="resnet50_wide", pretrained=False)
            pretrained_model_pth = os.path.join(pretrained_model_root, "wide_resnet50_2-95faca4d.pth")
            img_channels = [256, 512, 1024, 2048]

        elif self.backbone == "resnet101":
            self.feature_extractor = ResNet(backbone="resnet101", pretrained=False)
            pretrained_model_pth = os.path.join(pretrained_model_root, "resnet101-63fe2227.pth")
            img_channels = [256, 512, 1024, 2048]

        elif self.backbone == "resnet152":
            self.feature_extractor = ResNet(backbone="resnet152", pretrained=False)
            pretrained_model_pth = os.path.join(pretrained_model_root, "resnet152-394f9c45.pth")
            img_channels = [256, 512, 1024, 2048]
        
        elif self.backbone == "efficientnet_b7":
            pretrained_model_pth = os.path.join(pretrained_model_root, "efficientnet_b7_lukemelas-dcc49843.pth")
            self.feature_extractor = EfficientNet(
                backbone="b7", weights=pretrained_model_pth)
            pretrained_model_pth = None
            img_channels = [48, 80, 224, 640]
        elif self.backbone == "swin_t":
            pretrained_model_pth = os.path.join(pretrained_model_root, "swin_t-704ceda3.pth")
            self.feature_extractor = SwinTransformer(
                backbone="swin_t", weights=pretrained_model_pth)
            pretrained_model_pth = None
            img_channels = [96, 192, 384, 768]
        else:
            self.feature_extractor = None
            pretrained_model_pth = None
        
        if imgnet_pretrained and self.feature_extractor is not None and pretrained_model_pth is not None:
            self.feature_extractor.load_state_dict(
                torch.load(pretrained_model_pth, map_location="cpu"), strict=False)
        
        if self.use_image:
            num_patches = (img_size[0]//patch_size)*(img_size[1]//patch_size)
            
            self.feature_merge = AttentionFeatureMerge(
                in_channels=img_channels[-1], out_channels=emb_dims, 
                emb_dims=emb_dims, num_patches=num_patches, num_heads=num_heads)
            
            self.img_projection = PV2BEVProjection(
                bev_range=pc_range[:4], grid_size=grid_size*voxel_downscales,
                num_heads=num_heads, emb_dims=emb_dims,
                num_layers=1)
            
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches*self.num_images, emb_dims) * .02)
        
            trunc_normal_(self.pos_embed, std=.02)

            self.aux_decoder = RGBDecoder(in_channels=img_channels, nclasses=num_sem_classes, base_channels=32)

        self.bev_context_block = BevContextBlock(
            emb_dims=emb_dims, voxels_per_pillar=self.voxels_per_pillar, internel_channels=32
        )

        self.sem_decoder = QueryDecoder(
            num_classes=num_classes, internal_dims=32, pc_range=pc_range, use_rgb=True)
       
        if self.use_pcd:
            self.pcd_encoder = PointcloudEncoder(
                bev_range=pc_range[:4], grid_size=grid_size*voxel_downscales,
                emb_dims=emb_dims, num_heads=num_heads)

        if self.use_radar:
            self.radar_encoder = RadarEncoder(
                bev_range=pc_range[:4], grid_size=grid_size*voxel_downscales,
                emb_dims=emb_dims, num_heads=num_heads
            )

    def forward(self, x, query_pose, 
                aug_imgs=None,
                pcd_voxels=None, 
                radar_points=None):
        if aug_imgs is not None and self.use_image:
            proj_pred = self.forwardAugFeature(aug_imgs)
        else:
            proj_pred = None    
        features = self.forwardFeature(x, pcd_voxels, radar_points=radar_points)
        
        voxel_feature = self.bev_context_block(features)
        sem_preds, voxel_preds = self.sem_decoder(voxel_feature, query_pose)
        geo_rgb = sem_preds[..., :3]
        sem_preds = sem_preds[..., 3:]
        if self.empty_idx == 0:
            geo_cls = torch.cat([
                sem_preds[..., self.empty_idx:self.empty_idx+1], 
                sem_preds[..., self.empty_idx+1:].max(dim=-1, keepdim=True)[0]], 
                dim=-1)
        else:
            geo_cls = torch.cat([
                sem_preds[..., self.empty_idx:self.empty_idx+1], 
                sem_preds[..., :self.empty_idx].max(dim=-1, keepdim=True)[0]], 
                dim=-1)

        geo_preds = torch.cat([geo_rgb, geo_cls], dim=-1)  
        return sem_preds, geo_preds, proj_pred, voxel_preds

    def forwardAugFeature(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, H, W)
        img_feature = self.feature_extractor(x)
        proj_pred = self.aux_decoder(img_feature)
        return proj_pred
        
    def forwardFeature(self, x, pcd_voxels, radar_points=None):
        B, N, C, H, W = x.size()
        if self.use_pcd:
            bev_feature = self.pcd_encoder(pcd_voxels=pcd_voxels)
        else:
            bev_feature = None

        if self.use_image:
            input_imgs = x.view(B*N, C, H, W)
            img_feature = self.feature_extractor(input_imgs)
            # parallel single view processing
            img_feature = self.feature_merge(img_feature)
            _, L, C = img_feature.size()
            img_feature = img_feature.view(B, N*L, C).contiguous()+self.pos_embed
            bev_feature = self.img_projection(pv_feature=img_feature, bev_feature=bev_feature)
        
        if self.use_radar and radar_points is not None:
            assert bev_feature is not None
            bev_feature = self.radar_encoder(radar_points=radar_points, bev_feature=bev_feature)

        return bev_feature 

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with torch.no_grad():
        model = REONet(
            backbone="resnet50", img_size=(352, 1184), 
            use_pcd=True, num_images=2, use_radar=True,
            num_classes=20).cuda()
        model.train()
        test_aug_input = torch.zeros(1, 3, 256, 1024).cuda() # B C H W
        test_input = torch.zeros(1, 2, 3, 352, 1184).cuda() # B N C H W
        test_pcd_voxels = torch.zeros(1, 10240, 6).cuda()
        test_query = torch.zeros(1, 5000, 3).cuda()
        test_radar_points = torch.zeros(1, 5, 100, 4).cuda()
        test_fine_query = torch.rand(1, 256*256*32, 7).cuda()
        test_fine_query[..., 3] = torch.randint(low=0, high=18, size=(1, 256*256*32))
        outputs = model(test_input, test_query,
                        aug_imgs=test_aug_input,
                        pcd_voxels=test_pcd_voxels,
                        radar_points=test_radar_points)
        for out in outputs:
            if out is not None:
                print(out.size())
