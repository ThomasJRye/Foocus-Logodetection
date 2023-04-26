import torchvision
import tensorflow as tf

def get_model(device='cpu', model_name='v2', tensorboard_callback=None):
    # Load the model.
    if model_name == 'v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            pretrained=True,
            num_classes=33
        )
    elif model_name == 'v1':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            num_classes=33
        )
    # Load the model onto the computation device.
    model = model.eval().to(device)

    if tensorboard_callback is not None:
        # Add the TensorBoard callback to the model's list of callbacks
        callbacks = [tensorboard_callback]
        if model.training:
            model.train_callback_list = callbacks
        else:
            model.eval_callback_list = callbacks

    return model
