#!/usr/bin/env python
# coding: utf-8

from fastai.vision import *
import fastai
from PIL import Image
import os
import torch
#fastai.torch_core.defaults.device = 'cpu'
#defaults.device = torch.device('cpu')
fastai.torch_core.defaults.device = 'cpu'
#defaults.device = 'cpu'
#torch.cuda.device("cpu")
#torch.device("cpu")

# path to load exported model and to save csv
path = "ybns-v2/"
# Path to test.csv
csv_path = Path("/mnt/data/landmark2019/test.csv")
# Path to test images
img_path = Path("/mnt/data/landmark2019/test/")

# Load test.csv into dataframe
test_df = pd.read_csv(csv_path)

test_df["file_path"] =  test_df["id"].astype(str) + ".jpg"
test_df["file_exists"] = test_df["file_path"].apply(lambda x: True if Path(os.path.join(img_path, x)).is_file() else False)
test_df.set_index("id", inplace=True)

# create a new df to pass to imagelist but only from files that exist 
files_df = test_df.loc[test_df["file_exists"] == True][["file_path"]]
test_images = ImageList.from_df(df=files_df, path=img_path)


def get_pred_to_filename():
  # Create a dictionary with the "<category/label> <confidence score>" as value 
  # and image id (= image filename without file extension) as key 
  pred_to_filename = {}
  prob_to_filename = {}
  predstring_to_filename = {}

  i = 0
  for pred in preds:
    filename = os.path.basename(os.path.splitext(test_images.items[i])[0])
    # get the predicted category via the preds tensor
    category = learn.data.classes[preds[i].argmax().item()]
    # and its probabilty 
    probability = preds[i][preds[i].argmax().item()].item()
    # add an element to the dictionary with "<category> <probability>" with 
    # filename as key
    predstring_to_filename[filename] = str(category)+" "+str(round(probability, 5))
    pred_to_filename[filename] = category
    prob_to_filename[filename] = probability
    i += 1
    
  return predstring_to_filename, pred_to_filename, prob_to_filename  

submission_fname = "faoy-auto-1-6000"

imodels = ["auto-resnet101-0-1000-model", 
           "auto-resnet101-1000-2000-model",
           "auto-resnet101-2000-3000-model",
           "auto-resnet101-3000-4000-model",
           "auto-resnet101-4000-5000-model",
           "auto-resnet101-5000-6000-model"
          ]

i = 0
for imodel in imodels:
    print("Running on model: {}".format(imodel))
    
    predstring_to_filename = {}
    pred_to_filename = {}
    prob_to_filename = {}
    pickling = True
    i += 1
    # check if the model had been previously used (and we have a pickle of it)
    if os.path.isfile(path+"pickle/"+imodel+"_preds.pkl"):
        # Bingo ... just read the pickle
        print("{} pickle found - loading predictions from file".format(imodel))
        # skip the creation of the temp_df pickle further down
        pickling = False
        temp_df = pd.read_pickle(path+"pickle/"+imodel+"_preds.pkl")
        predstring_to_filename  = dict(zip(temp_df.index, temp_df.predstring))
        pred_to_filename  = dict(zip(temp_df.index, temp_df.landmarks))
        prob_to_filename  = dict(zip(temp_df.index, temp_df.prob))     
    else:
        # new model -> run the learner to get the predictions
        learn = load_learner(path, imodel+".pkl", test=test_images).to_fp16().to_fp32()
        # learn.data.batch_size = 128
        preds, y = learn.get_preds(ds_type=DatasetType.Test)
        predstring_to_filename, pred_to_filename, prob_to_filename  = get_pred_to_filename()
        
    if i == 1:
        # first iteration -> load ALL the data from the dictonary into the dataframe
        test_df["landmarks"] = test_df.index.map(pred_to_filename) 
        test_df["prob"] = test_df.index.map(prob_to_filename) 
        test_df["predstring"] = test_df.index.map(predstring_to_filename)
    else:
        # only load predictions that have higher probability then previous ones into dataframe
        test_df["landmarks"] = np.where(test_df.index.map(prob_to_filename) > test_df["prob"], test_df.index.map(pred_to_filename), test_df["landmarks"] )
        test_df["predstring"] = np.where(test_df.index.map(prob_to_filename) > test_df["prob"], test_df.index.map(predstring_to_filename), test_df["predstring"] )
        test_df["prob"] = np.where(test_df.index.map(prob_to_filename) > test_df["prob"], test_df.index.map(prob_to_filename), test_df["prob"] )

    if pickling:
        # store the predictions for this model in pickle
        # also ouput as csv for debugging
        temp_df = test_df.copy() 
        temp_df["landmarks"] = temp_df.index.map(pred_to_filename)
        temp_df["prob"] = temp_df.index.map(prob_to_filename)
        temp_df["predstring"] = temp_df.index.map(predstring_to_filename)
        #temp_df = temp_df[['predstring']].rename(columns={'predstring':'landmarks'})
        # this will create a seperate csv for each model with only the values predicted by that model
        temp_df.to_csv(path+"csv/"+imodel+"_preds.csv", sep=",", encoding="utf-8")
        # and also save as pickle in case we want to reimport it
        temp_df.to_pickle(path+"pickle/"+imodel+"_preds.pkl")

    # store everything for debugging in the test_df also ...
    test_df["landmarks"+str(i)] = test_df.index.map(pred_to_filename)
    test_df["prob"+str(i)] = test_df.index.map(prob_to_filename)
    test_df["predstring"+str(i)] = test_df.index.map(predstring_to_filename)

submission_df = test_df[['predstring']].rename(columns={'predstring':'landmarks'})
# save it in the local folder
submission_df.to_csv(submission_fname+".csv", sep=",", encoding="utf-8")
# save pickle and csv for debugging    
test_df.to_csv(path+"csv/"+submission_fname+"_with_all_data.csv")
