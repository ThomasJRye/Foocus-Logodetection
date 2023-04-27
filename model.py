import torchvision

def get_model(device='cuda', model_name='v2', tensorboard_callback=None):
    # Load the model.
    if model_name == 'v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights='DEFAULT',
        )
    elif model_name == 'v1':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights='DEFAULT',
        )
    # Load the model onto the computation device.
    model = model.eval().to(device)

    # if tensorboard_callback is not None:
    #     # Add the TensorBoard callback to the modesl's list of callbacks
    #     callbacks = [tensorboard_callback]
    #     if model.training:
    #         model.train_callback_list = callbacks
    #     else:
    #         model.eval_callback_list = callbacks

    return model
