import torch
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_b7, efficientnet_b6, efficientnet_b5

class EfficientNet(nn.Module):
    def __init__(self, backbone="b7", weights=None, out_index=[2, 3, 5, 7]) -> None:
        super(EfficientNet, self).__init__()

        if backbone == "b7":
            self.model = efficientnet_b7()
        elif backbone == "b6":
            self.model = efficientnet_b6()
        elif backbone == "b5":
            self.model = efficientnet_b5()
        else:
            raise NotImplementedError("invalid net type: {}".format(backbone))    
        if weights is not None:
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.out_index = out_index

    def forward(self, x):
        outs = []
        max_layer_idx = max(self.out_index)
        for i, layer in enumerate(self.model.features):
            x = layer(x)
            if i in self.out_index:
                outs.append(x)
            if i == max_layer_idx:
                break
        return outs



if __name__ == "__main__":
    test_input = torch.ones(1, 3, 352, 1184).cuda()
    model = EfficientNet("b7").cuda()
    test_output = model(test_input)
    for out in test_output:
        print(out.size())

