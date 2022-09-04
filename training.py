import os
from pathlib import Path

import pandas as pd
import numpy as np 
import timm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchvision.io import read_image
from torchmetrics import JaccardIndex

import av
from av2.utils.io import read_feather, read_img
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.rendering import video
from av2.rendering.video import tile_cameras, write_video

DATASET_PATH = "D:\dummy_data"
IN_CHANNELS = 27
TIME_LIMIT = 80
BATCH_SIZE = 4
NUM_EPOCHS = 2

LABELS_NUMS2WORDS_MAP = {
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

def data_processing(img_list):
    """ This function prepares the x values to be used in model training, inference, or 
        simply to run through the model (for example to produce visualizations)"""
    agg_view = []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    for view_path in img_list:
        if os.path.exists(view_path):
            image = read_img(view_path)
            # if the image is the opposite orientation than the majority, rotate it to be correct so we can stack them.
            if image.shape == (2048, 1550, 3):
                image = np.rot90(image).copy()
            resized_img = transform(image)
            agg_view.append(resized_img)
        
    x = np.vstack(agg_view)
    return torch.as_tensor(x)

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
        self.argo_data = AV2SensorDataLoader(data_dir=Path(f"{DATASET_PATH}"), labels_dir=Path(f"{DATASET_PATH}"))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> tuple:
        ## Data Ingestion 
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

        ## Data Preprocessing
        img_list = [rfc_path, rfl_path, rfr_path, rrl_path, rrr_path, rsl_path, rsr_path, sfl_path, sfr_path]
        x = data_processing(img_list)

        ret_label = self.img_labels.loc[self.img_labels['timestamp_ns'] == timestamp].copy()
        # replace word labels with numbers and remove rows of categories we aren't classifying
        ret_label['category'] = ret_label.category.map(LABELS_WORDS2NUM_MAP).fillna(0).astype(int)
        ret_label = ret_label.drop(ret_label[ret_label.category == 0].index)
        # drop unnecessary rows
        ret_label.drop(["track_uuid", "timestamp_ns", "num_interior_pts"], axis=1, inplace=True)

        return x, torch.as_tensor(ret_label.to_numpy()[:10])
    
def get_dataLoader(data: ArgoverseDataset) -> DataLoader :
    return DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
    )

def proj_loss_func(pred_label, true_label):
    """This is the loss function for the model. The labels 
       contain a model class as well as bounding box locations."""
    #print(f"pred_label: {pred_label}")
    #print(f"true_label: {true_label}")
    finds = []
    
    pred_label = torch.argmax(pred_label, axis=1)
    for label in true_label:
        for pred in pred_label:
            if label in pred:
                finds.append(1)
            else: 
                finds.append(0)

    return torch.sum(torch.as_tensor(finds)/len(true_label))

### BEGIN THE SCRIPT ###
# instantiate the training data and the data loader
argo_data = ArgoverseDataset()
argo_loader = get_dataLoader(argo_data)

# create model 
proj_model = timm.create_model(
    "mobilenetv3_large_100",
    pretrained=True,
    in_chans=IN_CHANNELS,
    num_classes=10,
)
print(proj_model.default_cfg)
#proj_model.cuda()

optimizer = torch.optim.AdamW(proj_model.parameters(), lr=0.01)
# print(proj_model)


### TRAIN ###
for epoch in range(NUM_EPOCHS):
    print(f"Epoch number: {epoch}")
    counter = 0 
    for x, true_label in argo_loader:
        optimizer.zero_grad()
        pred_label = proj_model(x)
        # print(f"losses: {pred_label}")
        # loss_func = JaccardIndex(10, multilabel=True)
        loss = proj_loss_func(pred_label, true_label).requires_grad_()

        loss.backward()
        optimizer.step()
        counter += 1
        torch.cuda.empty_cache()

        if counter == 50:
            break;
        if counter % 5 == 0:
            print(counter)


### EVALUATE ###
proj_model.eval()

# instantiate the testing data and the data loader
test_argo_data = ArgoverseDataset()
#    f"{DATASET_PATH}/022af476-9937-3e70-be52-f65420d52703/sensors/", 
#    f"{DATASET_PATH}/022af476-9937-3e70-be52-f65420d52703/annotations.feather",
#)
test_argo_loader = get_dataLoader(test_argo_data)

counter = 0
for x, true_label in test_argo_loader:
    confidences_logits = proj_model(x)
    counter += 1
    if counter > 10:
        break;


### VISUALIZATION ###
# Right now, the visualization just visualizes the image data. It does not add any info gleaned from 
# running the images through the model. 

# video parameter definition
fps = 24
video = av.open("C:/Users/viola/Documents/MSAI/CMPE249/final_proj_vid.mp4", mode="w")
stream = video.add_stream("mpeg4", rate=fps)
stream.width = 1550
stream.height = 2048
stream.pix_fmt = "yuv420p"

# initialize images that will be used in video
vis_data = AV2SensorDataLoader(data_dir=Path(f"{DATASET_PATH}"), labels_dir=Path(f"{DATASET_PATH}"))
vis_folder = "01bb304d-7bd8-35f8-bbef-7086b688e35e"

# iterate through one of the image directories to get all the timestamps.  
for filename in os.listdir("D:/dummy_data/01bb304d-7bd8-35f8-bbef-7086b688e35e/sensors/cameras/ring_front_center"):
    timestamp = int(os.path.splitext(filename)[0])

    # get file paths for each camera in tiling function
    ring_front_center_fpath = vis_data.get_closest_img_fpath(vis_folder, "ring_front_center", timestamp)
    ring_front_left_fpath = vis_data.get_closest_img_fpath(vis_folder, "ring_front_left", timestamp)
    ring_front_right_fpath = vis_data.get_closest_img_fpath(vis_folder, "ring_front_right", timestamp)
    ring_rear_left_fpath = vis_data.get_closest_img_fpath(vis_folder, "ring_rear_left", timestamp)
    ring_rear_right_fpath = vis_data.get_closest_img_fpath(vis_folder, "ring_rear_right", timestamp)
    ring_side_left_fpath = vis_data.get_closest_img_fpath(vis_folder, "ring_side_left", timestamp)
    ring_side_right_fpath = vis_data.get_closest_img_fpath(vis_folder, "ring_side_right", timestamp)
    stereo_front_left_fpath = vis_data.get_closest_img_fpath(vis_folder, "stereo_front_left", timestamp)
    stereo_front_right_fpath = vis_data.get_closest_img_fpath(vis_folder, "stereo_front_right", timestamp)
    # make image dict for tile_cameras function
    named_sensor_data = {}
    named_sensor_data["ring_front_center"] = read_img(ring_front_center_fpath)
    named_sensor_data["ring_front_left"] = read_img(ring_front_left_fpath)
    named_sensor_data["ring_front_right"] = read_img(ring_front_right_fpath)
    named_sensor_data["ring_rear_left"] = read_img(ring_rear_left_fpath)
    named_sensor_data["ring_rear_right"] = read_img(ring_rear_right_fpath)
    named_sensor_data["ring_side_left"] = read_img(ring_side_left_fpath)
    named_sensor_data["ring_side_right"] = read_img(ring_side_right_fpath)

    # put together image list for data processing and run model inference 
    # even though the stereo images are not included in the visualization, 
    # they still need to be including in inference
    img_list = [
        ring_front_center_fpath,
        ring_front_left_fpath,
        ring_front_right_fpath,
        ring_rear_left_fpath,
        ring_rear_right_fpath,
        ring_side_left_fpath,
        ring_side_right_fpath,
        stereo_front_left_fpath,
        stereo_front_right_fpath,
    ]
    x = data_processing(img_list)
    boxes = proj_model(np.expand_dims(x, axis=0))
    print(f"boxes: {boxes}")

    # add images from all cameras to one frame
    frame = av.VideoFrame.from_ndarray(np.array(tile_cameras(named_sensor_data)), format="rgb24")

    # add frame to video
    for packet in stream.encode(frame):
        video.mux(packet)

# flush stream
for packet in stream.encode():
    video.mux(packet)

video.close()
