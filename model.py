import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from coco_names import COCO_INSTANCE_CATEGORY_NAMES

def get_model(device='cpu', model_name="v1"):
    # Load the model.
    if model_name == 'v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    elif model_name == 'v1':
        # load model pretrained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # replace the classifier with a new one, with correct nr of classes
    # pretrained model has 91 classes as default, we only need 31!
    num_classes = len(COCO_INSTANCE_CATEGORY_NAMES) + 1 # nr of logos + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the model onto the computation device.
    model = model.eval().to(device)

    return model
