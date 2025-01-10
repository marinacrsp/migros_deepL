import torchvision.models as models
import torch.nn as nn


def build_model(pretrained=True, fine_tune=True, num_classes=2):
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif not pretrained:
        print("[INFO]: Not loading pre-trained weights")
        model = models.resnet18(weights=None)
    if fine_tune:
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad_(False)
        print("[INFO]: Fine-tuning the following layers...")
        for module, _ in model.named_children():
            layer_block = False
            if 'layer' in module:
                if 'layer1' in module:
                    layer_block = model.layer1
                elif 'layer2' in module:
                    layer_block = model.layer2
                # if 'layer3' in module:
                #     layer_block = model.layer3
                # elif 'layer4' in module:
                #     layer_block = model.layer4
                
                if layer_block:
                    for buffer in layer_block:
                        buffer.conv1.requires_grad_(True)
                        print(buffer.conv2)
        # for params in model.parameters():
        #     params.requires_grad = True
    elif not fine_tune:
        print("[INFO]: Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head, it is trainable.
    model.fc = nn.Linear(512, num_classes)
    return model


if __name__ == "__main__":
    model = build_model(num_classes=1000)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")

