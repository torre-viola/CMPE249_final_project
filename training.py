import os
from pathlib import Path
import pandas as pd
import torchvision
from torchvision import transforms
from torchvision.io import read_image
import numpy as np 
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from av2.utils.io import read_feather, read_img
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader

DATASET_PATH = "/data/cmpe249-fa22/Argoverse_Sensor_Data/train"
IN_CHANNELS = 27
TIME_LIMIT = 80
BATCH_SIZE = 16
NUM_EPOCHS = 10

LABELS_NUMS2WORDS_MAP = {
    0: "OTHER",
    1: "REGULAR_VEHICLE",
    2: "PEDESTRIAN",
    3: "BOLLARD",
    4: "CONSTRUCTION_CONE",
    5: "CONSTRUCTION_BARREL",
    6: "STOP_SIGN",
    7: "BICYCLE",
    8: "LARGE_VEHICLE",
    9: "WHEELED_DEVICE",
    10: "BUS",
}

LABELS_WORDS2NUM_MAP = {
    "REGULAR_VEHICLE": 1,
    "PEDESTRIAN": 2,
    "BOLLARD": 3,
    "CONSTRUCTION_CONE": 4,
    "CONSTRUCTION_BARREL": 5,
    "STOP_SIGN": 6,
    "BICYCLE": 7,
    "LARGE_VEHICLE": 8,
    "WHEELED_DEVICE": 9,
    "BUS": 10,
}

class ArgoverseDataset(Dataset):
    """ class object which inherits from the pytorch Dataset class to 
        injest the dataset and prepare it for iteration during training """
    def __init__(
        self, 
        img_dir: str=f"{DATASET_PATH}/01bb304d-7bd8-35f8-bbef-7086b688e35e/sensors/", 
        annotations_file: str=f"{DATASET_PATH}/01bb304d-7bd8-35f8-bbef-7086b688e35e/annotations.feather", 
    ):
        # read_feather returns a pandas dataframe
        self.img_labels = read_feather(annotations_file)
        self.img_dir = img_dir

        self.argo_data = AV2SensorDataLoader(
            data_dir=Path(f"{DATASET_PATH}"), 
            labels_dir=Path(f"{DATASET_PATH}"),
        )

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(299)])

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> tuple:

        timestamp = self.img_labels.iloc[idx, 0]
        rfc_path = self.argo_data.get_closest_img_fpath("01bb304d-7bd8-35f8-bbef-7086b688e35e", "ring_front_center", timestamp)
        rfl_path = self.argo_data.get_closest_img_fpath("01bb304d-7bd8-35f8-bbef-7086b688e35e", "ring_front_left", timestamp)
        rfr_path = self.argo_data.get_closest_img_fpath("01bb304d-7bd8-35f8-bbef-7086b688e35e", "ring_front_right", timestamp)
        rrl_path = self.argo_data.get_closest_img_fpath("01bb304d-7bd8-35f8-bbef-7086b688e35e", "ring_rear_left", timestamp)
        rrr_path = self.argo_data.get_closest_img_fpath("01bb304d-7bd8-35f8-bbef-7086b688e35e", "ring_rear_right", timestamp)
        rsl_path = self.argo_data.get_closest_img_fpath("01bb304d-7bd8-35f8-bbef-7086b688e35e", "ring_side_left", timestamp)
        rsr_path = self.argo_data.get_closest_img_fpath("01bb304d-7bd8-35f8-bbef-7086b688e35e", "ring_side_right", timestamp)
        sfl_path = self.argo_data.get_closest_img_fpath("01bb304d-7bd8-35f8-bbef-7086b688e35e", "stereo_front_left", timestamp)
        sfr_path = self.argo_data.get_closest_img_fpath("01bb304d-7bd8-35f8-bbef-7086b688e35e", "stereo_front_right", timestamp)

        agg_view = []
        for view_path in [rfc_path, rfl_path, rfr_path, rrl_path, rrr_path, rsl_path, rsr_path, sfl_path, sfr_path]:
            if os.path.exists(view_path):
                image = read_img(view_path)
                # if the image is the opposite orientation than the majority, rotate it to be correct so we can stack them.
                if image.shape == (2048, 1550, 3):
                    image = np.rot90(image).copy()
                # print(view_path, "has shape", image.shape)
                tensor_image = torch.as_tensor(image)
                resized_img = transforms.Resize(299)(tensor_image)
                agg_view.append(resized_img.to_numpy)
        
        ret_label = self.img_labels.loc[self.img_labels['timestamp_ns'] == timestamp].copy()
        ret_label['category'] = ret_label.category.map(LABELS_WORDS2NUM_MAP).fillna(0).astype(int)
        ret_label.drop("track_uuid", axis=1, inplace=True)

        x = np.swapaxes(np.dstack(agg_view),0,2)

        return torch.as_tensor(x), torch.as_tensor(ret_label.to_numpy()[:10])
    
def get_dataLoader(data: ArgoverseDataset) -> DataLoader :
    return DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
    )

argo_data = ArgoverseDataset()
argo_loader = get_dataLoader(argo_data)

def loss_func(pred_label, true_label):
    """This is the loss function for the model. The labels 
       contain a model class as well as bounding box locations."""
    return pred_label - true_label

# Create model 
proj_model = timm.create_model(
    "xception71",
    pretrained=True,
    in_chans=IN_CHANNELS,
    num_classes=11,
)
print(proj_model.default_cfg)
# proj_model.cuda()

optimizer = torch.optim.AdamW(proj_model.parameters(), lr=0.01)
# print(proj_model)

for epoch in range(NUM_EPOCHS):
    print(f"Epoch number: {epoch}")
    for x, true_label in argo_loader:
        print(x.shape)
        break;
        pred_label = proj_model(x)
        loss = loss_func(pred_label, true_label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
