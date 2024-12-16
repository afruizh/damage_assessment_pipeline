import cv2 as cv
import numpy as np
from PIL import Image
import glob
import pathlib

import sys

import u2net_utils

import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

from u2net_utils.data_loader import RescaleT
from u2net_utils.data_loader import ToTensor
from u2net_utils.data_loader import ToTensorLab
from u2net_utils.data_loader import SalObjDataset

from u2net_utils.model import U2NET # full size version 173.6 MB
from u2net_utils.model import U2NETP # small version u2net 4.7 MB

from torchvision import models


import onnxruntime as ort
import cv2 as cv
import numpy as np
from torchvision.transforms import v2 as transforms

# MODEL_PATH = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_gpu\models\u2net.pth"
# MODEL_PATH = r"D:\CIAT\catalogue\AcceleratedBreedingInitiative\1.Data\16. Spidermites_AdrianK\best_models"
# MODEL_PATH = r"D:\local_mydata\models\spidermites\best_models"

MODEL_PATH = "./models"

#************************
# from loguru import logger
# from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
# import subprocess

# # Grounding DINO
# import GroundingDINO.groundingdino.datasets.transforms as T
# from GroundingDINO.groundingdino.models import build_model
# from GroundingDINO.groundingdino.util import box_ops
# from GroundingDINO.groundingdino.util.slconfig import SLConfig
# from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# from huggingface_hub import hf_hub_download

import gc

def clear():    
    gc.collect()
    torch.cuda.empty_cache() 

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

class BackgroundRemover():

    def __init__(self):
        

        #Load model
        #model_dir = "/workspace/u2net.pth"
        #model_dir = "D:/local_mydata/models/u2net.pth"
        model_dir = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_gpu\models\u2net.pth"
        model_dir = os.path.join(MODEL_PATH, "u2net.pth")

        ## Load model
        net = U2NET(3,1)

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        net.eval()

        self.net = net

    def remove_background(self, filepath_image):

        img_name_list = [filepath_image]

        test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                            lbl_name_list = [],
                                            transform=transforms.Compose([RescaleT(320),
                                                                        ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)    

        net = self.net

        for i_test, data_test in enumerate(test_salobj_dataloader):

            print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

            # normalization
            pred = d1[:,0,:,:]
            pred = normPRED(pred)

            # save results to test_results folder
            #if not os.path.exists(prediction_dir):
            #    os.makedirs(prediction_dir, exist_ok=True)
            #save_output(img_name_list[i_test],pred,prediction_dir)

            predict = pred
            predict = predict.squeeze()
            #mask_torch.permute(1, 2, 0).detach().cpu().numpy()
            predict_np = predict.cpu().data.numpy()

            img = cv.imread(filepath_image)
            w = img.shape[1]
            h = img.shape[0]

            #im = Image.fromarray(predict_np*255).convert('RGB')
            #image = io.imread(filepath_image)
            #imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

            imo = cv.resize(predict_np, (w,h), cv.INTER_LINEAR )

            #del d1,d2,d3,d4,d5,d6,d7
            return imo
        
    def remove_background_save(self, path_in, path_out, path_out_mask = None):

        print("remove_background_save")

        mask_torch = self.remove_background(path_in)
        mask = mask_torch*255
        mask = mask.astype(np.uint8)

        img = cv.imread(path_in)
        mask0 = mask#cv.UMat(cv.imread(mask,0))
        #127
        #200
        ret,binary_mask = cv.threshold(mask0,80,255,cv.THRESH_BINARY)
        binary_mask = np.uint8(binary_mask)
        res = cv.bitwise_and(img,img, mask = binary_mask)

        cv.imwrite(path_out, res)

        if not (path_out_mask == None):
            cv.imwrite(path_out_mask, mask)

    def remove_background_dir(self, path_in, path_out):

        img_name_list = glob.glob(os.path.join(path_in, "*.jpg"))

        for img_name in img_name_list:

            img_name_output = img_name.replace(path_in, path_out)

            if not os.path.exists(img_name_output):
                self.remove_background_save(img_name, img_name_output)
                print(img_name.replace(path_in, path_out))

    def remove_background_gradio(self, np_image):

        w = np_image.shape[1]
        h = np_image.shape[0]

        #image = torch.tensor(np_image)
        #image = image.permute(2,0,1)

        image = np_image#Image.fromarray(np_image)
        imidx = np.array([0])
        #label = "test"

        #***
        label_3 = np.zeros(image.shape)
        
        label = np.zeros(label_3.shape[0:2])
        if(3==len(label_3.shape)):
            label = label_3[:,:,0]
        elif(2==len(label_3.shape)):
            label = label_3

        if(3==len(image.shape) and 2==len(label.shape)):
            label = label[:,:,np.newaxis]
        elif(2==len(image.shape) and 2==len(label.shape)):
            image = image[:,:,np.newaxis]
            label = label[:,:,np.newaxis]
        #***
        
        sample = {'imidx':imidx, 'image':image, 'label':label}
        print(image.shape)
        print(label.shape)

        
        eval_transform = transforms.Compose([RescaleT(320),ToTensorLab(flag=0)])
        #eval_transform = transforms.Compose([RescaleT(320)])
        #eval_transform = transforms.Compose([RescaleT(320)])
        #eval_transform = transforms.Compose([ToTensorLab(flag=0)])
        #eval_transform = transforms.Compose([transforms.Resize(320)
        #                                     , transforms.ToTensor()])
        #eval_transform = transforms.Compose([transforms.Resize(320)])

        test_salobj_dataloader = DataLoader(sample,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
        
        sample = eval_transform(sample)

        net = self.net

        #for i_test, data_test in enumerate(test_salobj_dataloader):

        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        #x = eval_transform(sample)
        #x = x[:3, ...].to(device)

        inputs_test = sample['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = inputs_test.unsqueeze(0)

        print(inputs_test.shape)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)


        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        predict = pred
        predict = predict.squeeze()
        #mask_torch.permute(1, 2, 0).detach().cpu().numpy()
        predict_np = predict.cpu().data.numpy()

        imo = cv.resize(predict_np, (w,h), cv.INTER_LINEAR )

        mask = imo*255
        mask = mask.astype(np.uint8)
        mask0 = mask#cv.UMat(cv.imread(mask,0))
        #127
        #200
        ret,binary_mask = cv.threshold(mask0,80,255,cv.THRESH_BINARY)
        #ret,binary_mask = cv.threshold(mask0,233,255,cv.THRESH_BINARY)
        binary_mask = np.uint8(binary_mask)
        res = cv.bitwise_and(np_image,np_image, mask = binary_mask)

        return mask, res
    
    def apply_mask(self, input, mask, threshold):

        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        ret,binary_mask = cv.threshold(mask,threshold,255,cv.THRESH_BINARY)
        #binary_mask = np.uint8(binary_mask)
        #binary_mask = mask
        print("apply mask")
        print(input.shape)
        print(input.dtype)
        print(binary_mask.shape)
        print(binary_mask.dtype)
        res = cv.bitwise_and(input,input, mask = binary_mask)

        # foreground_alpha = mask.astype(np.float32) / 255.0 
        # # Create a new image to store the result with same size and type as foreground
        # blended_image = np.zeros_like(input)

        # # Loop through each pixel and apply alpha based on mask value
        # for channel in range(3):  # Loop through BGR channels
        #     blended_image[:, :, channel] = input[:, :, channel] * foreground_alpha


        return res, binary_mask


def get_transform(train = True):
    transforms_list = []
    #if train:
    #    transforms.append(T.RandomHorizontalFlip(0.5))
    transforms_list.append(transforms.Resize(256))
    transforms_list.append(transforms.CenterCrop(256))
    #transforms_list.append(transforms.ToDtype(torch.float, scale=True))
    transforms_list.append(transforms.ToTensor())
    #transforms_list.append(transforms.ToDtype(torch.float32, scale=True))    
    transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transforms_list)
    
class DamageClassifier():

    def __init__(self):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_name =""
        

    def initialize(self, model_name):

        #Load model

        if model_name == "Resnet18":

            model_filepath = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\16. Spidermites_AdrianK\best_models\resnet18_SpidermitesModel.pth"
            model_filepath = os.path.join(MODEL_PATH, "resnet18_SpidermitesModel.pth")
            model = models.resnet18(weights='IMAGENET1K_V1')

        if model_name == "Resnet152":

            model_filepath = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\16. Spidermites_AdrianK\best_models\short_resnet152_SpidermitesModel_44_44.pth"
            model_filepath = os.path.join(MODEL_PATH, "short_resnet152_SpidermitesModel_44_44.pth")
            model = models.resnet152(weights='IMAGENET1K_V1')

        if model_name == "Googlenet":

            model_filepath = r"\\catalogue.cgiarad.org\AcceleratedBreedingInitiative\1.Data\16. Spidermites_AdrianK\best_models\regnet_x_32gf_SpidermitesModel.pth"
            model_filepath = model_filepath = os.path.join(MODEL_PATH, "regnet_x_32gf_SpidermitesModel.pth")
            model = models.regnet_x_32gf(weights='IMAGENET1K_V1')       
            
        if model_name == "Regnet32":

            model_filepath = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\16. Spidermites_AdrianK\best_models\short_resnet18_SpidermitesModel.pth"
            model_filepath = model_filepath = os.path.join(MODEL_PATH, "short_resnet18_SpidermitesModel.pth")
            model = models.resnet18(weights='IMAGENET1K_V1')
        
        #Add fully connected layer at the end with num_classes as output
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_filepath))
            model.cuda()
        else:
            model.load_state_dict(torch.load(model_filepath, map_location='cpu'))
        model.eval()

        self.model = model
        self.model_name = model_name

        return


    def inference(self, np_image, model_name):

        if model_name == "Regnet":

            model_filepath = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\16. Spidermites_AdrianK\best_models\regnet_x_32gf_SpidermitesModel.onnx"
            model_filepath = model_filepath = os.path.join(MODEL_PATH, "regnet_x_32gf_SpidermitesModel.onnx")
            ort_sess = ort.InferenceSession(model_filepath
                                ,providers=ort.get_available_providers()
                                )
            
            transforms_list = []
            transforms_list.append(transforms.ToTensor())
            transforms_list.append(transforms.Resize(512))
            transforms_list.append(transforms.CenterCrop(512))
            transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

            apply_t =  transforms.Compose(transforms_list) 

            img = apply_t(np_image)

            imgs = np.array([img.numpy()])

            outputs = ort_sess.run(None, {'input': [img.numpy()]})

            np_res = outputs[0][0]


            final_res = {'0-(No damage)': np_res[0]
                            ,'1-3-(Moderately damaged)': np_res[1]
                            ,'4-7-(Damaged)': np_res[2]
                            ,'8-10-(Severely damaged)': np_res[3]}
                
            return final_res

        else:

            if self.model_name != model_name:
                self.initialize(model_name)

            with torch.no_grad():

                print("inference")
                print(np_image.shape)

                pil_image = Image.fromarray(np_image.astype('uint8'))
                data_transforms = get_transform(train = False)

                img = data_transforms(pil_image)

                inputs = img.to(self.device)

                outputs = self.model(inputs.unsqueeze(0))
                #_, preds = torch.max(outputs, 1)

                print(outputs)

                _, preds = torch.max(outputs, 1)
                print(preds)

                m = nn.Softmax(dim=1)
                res = m(outputs)
                print(res)

                np_res = res[0].cpu().numpy()
                print(np_res)

                final_res = {'0-(No damage)': np_res[0]
                            ,'1-3-(Moderately damaged)': np_res[1]
                            ,'4-7-(Damaged)': np_res[2]
                            ,'8-10-(Severely damaged)': np_res[3]}
                
                return final_res

class ColorCheckerDetector():

    def __init__(self):

        return
    
    def process(self, np_image_mask, np_image):

        ret,binary_mask = cv.threshold(np_image_mask,80,255,cv.THRESH_BINARY)
        binary_mask_C = cv.cvtColor(binary_mask, cv.COLOR_BGR2GRAY) #change to single channel
        (contours, hierarchy) = cv.findContours(binary_mask_C, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        main_contour = contours[0]

        # compute the center of the contour
        moments = cv.moments(main_contour)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Bounding rect
        bb_x,bb_y,bb_w,bb_h = cv.boundingRect(binary_mask_C)

        # Min Bounding rect
        rect = cv.minAreaRect(main_contour)
        box = cv.boxPoints(rect)
        box = np.int64(box)

        # Fitting line
        rows,cols = binary_mask_C.shape[:2]
        #[vx,vy,x,y] = cv.fitLine(main_contour, cv.DIST_L2,0,0.01,0.01)
        [vx,vy,x,y] = cv.fitLine(box, cv.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        point1 = (cols-1,righty)
        point2 = (0,lefty)
        angle = np.arctan2(np.abs(righty-lefty),cols)

        # rotation matrix
        M_rot = cv.getRotationMatrix2D((cx, cy), -angle*180.0/np.pi, 1.0)
        rotated = cv.warpAffine(np_image, M_rot, (binary_mask.shape[1], binary_mask.shape[0]))

        #perspective transform
        input_pts = box.astype(np.float32)
        maxHeight = 200
        maxWidth = 290
        output_pts = np.float32([[0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1] ,                       
                        [0, maxHeight - 1]]
                        )
        M_per = cv.getPerspectiveTransform(input_pts,output_pts)
        corrected = cv.warpPerspective(np_image,M_per,(maxWidth, maxHeight),flags=cv.INTER_LINEAR)

        res = cv.drawContours(np_image, main_contour, -1, (255,255,0), 5)
        res = cv.rectangle(res,(bb_x,bb_y),(bb_x+bb_w,bb_y+bb_h),(0,255,0),5)
        res = cv.drawContours(res,[box],0,(0,0,255),5)
        res = cv.line(res,(cols-1,righty),(0,lefty),(0,0,255),5)

        return [res, rotated, corrected]




class BatchProcessor():

    def __init__(self):
        return
    
    def batch_process(self, input_dir, output_dir, output_suffixes = ["output"], format="jpg", pattern='**/*.tiff', processing_fc=None, output_format = None):

        if processing_fc == None:
            print("Processing function is None")
            return
        else:

            if output_format == None:
                output_format = format

            # Get list of files in folder and subfolders
            pattern = '**/*.'  + format
            files = glob.glob(pattern, root_dir=input_dir, recursive=True)

            for file in files:

                filepath = os.path.join(input_dir, file)
                basename = os.path.basename(filepath)
                parent_dir = os.path.dirname(filepath)
                extra_path = file.replace(basename,"")
                output_dir = os.path.join(output_dir, extra_path)

                # Create output filepath list
                output_filepaths = []
                for suffix in output_suffixes:
                    output_filepaths.append(os.path.join(output_dir, basename.replace("." + format, "_" + suffix + "." + output_format)))

                if not os.path.exists(output_filepaths[0]):# Process only if first output file does not exist

                    if not os.path.exists(output_dir): # Create subfolders if necessary
                        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)


                    processing_fc(filepath, output_filepaths) # Process and save file

                    print(file)
                    print(output_filepaths[0])
                    print("****")


class Segmentor():

    def __init__(self):

        self.sam_predictor = None
        self.groundingdino_model = None
        #self.sam_checkpoint = './sam_vit_h_4b8939.pth'
        #self.sam_checkpoint = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\4.Scripts\AndresRuiz\local_mydata_backup\model\sam_vit_h_4b8939.pth"
        self.sam_checkpoint = r"D:\local_mydev\Grounded-Segment-Anything\sam_vit_h_4b8939.pth"


        # self.config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        # self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        # self.ckpt_filename = "groundingdino_swint_ogc.pth"

        self.config_file = r"D:\local_mydev\gsam\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.ckpt_filename = "groundingdino_swint_ogc.pth"

        self.device ='cpu'

        self.load_sam_model(self.device)
        self.load_groundingdino_model(self.device)

        return
    
    def get_sam_vit_h_4b8939(self):
        return
        # if not os.path.exists('./sam_vit_h_4b8939.pth'):
        #     logger.info(f"get sam_vit_h_4b8939.pth...")
        #     result = subprocess.run(['wget', '-nv', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'], check=True)
        #     print(f'wget sam_vit_h_4b8939.pth result = {result}')
    
    def load_sam_model(self, device):

        sam_checkpoint = self.sam_checkpoint

        # initialize SAM
        self.get_sam_vit_h_4b8939()
        logger.info(f"initialize SAM model...")
        sam_device = device
        sam_model = build_sam(checkpoint=sam_checkpoint).to(sam_device)
        self.sam_predictor = SamPredictor(sam_model)
        self.sam_mask_generator = SamAutomaticMaskGenerator(sam_model)

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases
    
    def load_model_hf(self, model_config_path, repo_id, filename, device='cpu'):
        args = SLConfig.fromfile(model_config_path) 
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        print(checkpoint['model'])
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model 
    
    def load_groundingdino_model(self, device):

        config_file = self.config_file
        ckpt_repo_id = self.ckpt_repo_id
        ckpt_filename = self.ckpt_filename


        # initialize groundingdino model
        logger.info(f"initialize groundingdino model...")
        self.groundingdino_model = self.load_model_hf(config_file, ckpt_repo_id, ckpt_filename, device=device) #'cpu')
        logger.info(f"initialize groundingdino model...{type(self.groundingdino_model)}")

    def show_mask(self, mask, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        color = np.array([1.0, 0, 0, 1.0])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        return mask_image
        
    
    def process(self, np_image, text_prompt):

        results = []
        results.append(np_image)
        #results.append(np_image)

        sam_predictor = self.sam_predictor
        groundingdino_model = self.groundingdino_model

        image = np_image
        #text_prompt = text_prompt.strip()

        box_threshold = 0.3
        text_threshold = 0.25
        size = image.shape
        H, W = size[1], size[0]

        # RUN grounding dino model
        groundingdino_device = 'cpu'

        #image_dino = torch.from_numpy(image)
        image_dino = Image.fromarray(image)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        print(image.shape)
        image_dino, _ = transform(image_dino, None)  # 3, h, w

        boxes_filt, pred_phrases =self.get_grounding_output(
            groundingdino_model, image_dino, text_prompt, box_threshold, text_threshold, device=groundingdino_device
        )

        if sam_predictor:
            sam_predictor.set_image(image)

        if sam_predictor:


            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])


            masks, _, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )

            print("RESULTS*************")
            print(len(masks))

            # results = []

            for mask in masks:
                print(type(mask))
                print(mask.shape)
                #mask_img = mask.cpu().data.numpy()
                mask_img =self.show_mask(mask.cpu().numpy())
                print(type(mask_img))
                print(mask_img.shape)
                results.append(mask_img)
            #     results.append(mask.cpu().numpy())

            return results
            #assert sam_checkpoint, 'sam_checkpoint is not found!'

        return None