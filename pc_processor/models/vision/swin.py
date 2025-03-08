import torch
import torch.nn as nn
from torchvision.models.swin_transformer import swin_b, swin_s, swin_t

class SwinTransformer(nn.Module):
    def __init__(self, backbone="swin_b", weights=None) -> None:
        super(SwinTransformer, self).__init__()

        if backbone == "swin_b":
            self.model = swin_b()
        elif backbone == "swin_s":
            self.model = swin_s()
        elif backbone == "swin_t":
            self.model = swin_t()
        else:
            raise NotImplementedError("invalid net type: {}".format(backbone))    
        if weights is not None:
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.out_index = [1, 3, 5, 7]
        self.model.head = None
        
    def forward(self, x):
        outs = []
        max_layer_idx = max(self.out_index)
        for i, layer in enumerate(self.model.features):
            x = layer(x)
            if i in self.out_index:
                outs.append(x.permute(0, 3, 1, 2))
            if i == max_layer_idx:
                break
        return outs

if __name__ == "__main__":
    test_input = torch.ones(1, 3, 352, 1184).cuda()
    model = SwinTransformer("swin_t").cuda()
    test_output = model(test_input)
    for out in test_output:
        print(out.size())

