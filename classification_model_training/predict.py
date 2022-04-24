import torch

import numpy as np
import os
import pandas as pd
import albumentations
import argparse
import glob
import csv
import sklearn

from efficientnet_pytorch import EfficientNet
from utils.data import ClassificationDataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from utils.calc_mean_std import get_mean_std

from torchvision import transforms


def load_data(test_data_path, gt):

    df_test = pd.read_csv(gt)

    test_images = df_test.image.values.tolist()
    test_images = [os.path.join(test_data_path, i) for i in test_images]
    test_targets = df_test.target.values

    return test_images, test_targets


def predict_multiclass(model, device, test_loader, ensemble):

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            
            target = target.type(torch.DoubleTensor)
            data= data.to(device)
            output = model(data)
            output = torch.sigmoid(output)
            output_noargmax = output.cpu().numpy()
            output = np.argmax(output_noargmax, axis=1)

            if i==0:
                predictions_no_argmax = output_noargmax
                predictions = output
                targets = target.data.cpu().numpy()

            else:
                predictions_no_argmax = np.concatenate((predictions_no_argmax, output_noargmax))
                predictions = np.concatenate((predictions, output))
                targets = np.concatenate((targets, target.data.cpu().numpy()))

    if ensemble:
        return targets, predictions_no_argmax

    f1_score = sklearn.metrics.f1_score(targets, predictions, average='micro')

    return f1_score


def predict(model, device, test_loader, ensemble):

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):

            target = target.type(torch.DoubleTensor)
            data= data.to(device)
            output = model(data)

            output = torch.sigmoid(output)
            output = output.cpu().numpy()

            for j in range(len(output)):
                if output[j] > 0.5:
                    output[j] = 1
                else:
                    output[j] = 0

            if i==0:
                predictions = output
                targets = target.data.cpu().numpy()
            else: 
                predictions = np.concatenate((predictions,output))
                targets = np.concatenate((targets, target.data.cpu().numpy()))

    predictions = predictions.flatten()

    if ensemble:
        return targets, predictions
    
    f1_score = sklearn.metrics.f1_score(targets, predictions)
    accuracy = sklearn.metrics.accuracy_score(targets, predictions)

    return f1_score, accuracy

    


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="path to the folder containing the images size: 260x260")
    parser.add_argument("--gt", type=str, help="path to the file that contains the gt csv file - see examples under /examples (needs to be added)")
    parser.add_argument("--num_classes", type=int, help="The number of classes in the test dataset")

    parser.add_argument("--single_model_path", default=None, type=str, help="Use only for single model prediction: path to the folder containing the model file: .bin")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_bs", type=int, default=16)


    opt = parser.parse_args()
    
    test_images, test_targets = load_data(opt.dataset, opt.gt)




    prediction_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = ClassificationDataset(
            image_paths=test_images,
            targets=test_targets,
            augmentations=prediction_aug)



    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.val_bs,
        shuffle=False,
        num_workers=2
    )

    if opt.num_classes == 1:

        if opt.single_model_path is not None:
            
            model = EfficientNet.from_name("efficientnet-b2", in_channels = 3, num_classes = 1)
            model.load_state_dict(torch.load(opt.single_model_path))
            model.to(opt.device)

            f1_score, accuracy = predict(model, opt.device, test_loader, ensemble=False)

            print("f1_score: ", f1_score, "accuracy: ", accuracy)
            
        else:
            print("No model path specified - read opt.arguments")

