import cv2 as cv
import numpy as np
import glob
import pathlib

import os

import pandas as pd
#from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
#from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.table import Table, TableStyleInfo

import onnxruntime as ort
import cv2 as cv
import numpy as np

from skimage import transform

MODEL_PATH = "./models"
#MODEL_PATH = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\16. Spidermites_AdrianK\models\onnx"


def rescale_t_classification(image, target_size=320):
    """
    Rescale image to have its shorter side equal to target_size while maintaining aspect ratio.
    """
    h, w = image.shape[:2]
    if w < h:
        new_w = target_size
        new_h = int(target_size * h / w)
    else:
        new_h = target_size
        new_w = int(target_size * w / h)
    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
    return resized

def rescale_t(image, target_size=320):
    """
    Rescale image to have its shorter side equal to target_size while maintaining aspect ratio.
    """
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = target_size*h/w,target_size
        #new_w = target_size
        #new_h = int(target_size * h / w)
    else:
        #new_h = target_size
        #new_w = int(target_size * w / h)
        new_h, new_w = target_size,target_size*w/h

    new_h, new_w = int(new_h), int(new_w)

    print(new_w)
    print(new_h)

    # resized = cv.resize(image, (target_size, target_size))
    # resized = resized / 255.0

    resized = transform.resize(image,(target_size,target_size),mode='constant')

    return resized

# def to_tensor_lab(image, flag=0):
#     """
#     Process an image and its corresponding label like the original ToTensorLab,
#     but without converting to torch.Tensor.
    
#     Parameters:
#       imidx (any): an index/identifier.
#       image (np.ndarray): input image as a numpy array (assumed to be in RGB).
#       label (np.ndarray): corresponding label map.
#       flag (int): determines which processing branch to follow:
#                   2: use both RGB and LAB channels (6 channels),
#                   1: use LAB only,
#                   0: use RGB only.
    
#     Returns:
#       dict: a dictionary with keys 'imidx', 'image', and 'label', where
#             - image is transposed to shape (C, H, W),
#             - label is similarly transposed if it has a channel dimension.
#     """
    
#     if flag == 2:
#         # With RGB and LAB channels (6 channels)
#         H, W = image.shape[:2]
#         tmpImg = np.zeros((H, W, 6), dtype=np.float32)
        
#         # Ensure we have a 3-channel image: if single channel, replicate it.
#         if image.ndim == 2 or image.shape[2] == 1:
#             tmpImgt = np.repeat(image, 3, axis=2) if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2RGB)
#         else:
#             tmpImgt = image.copy()
        
#         # Convert the RGB image to LAB (scikit-image expects float images in [0,1])
#         # (Assumes image is in RGB; if using OpenCV, convert BGR to RGB before calling this.)
#         tmpImgt = tmpImgt.astype(np.float32) / 255.0
#         tmpImgtl = color.rgb2lab(tmpImgt)
        
#         # Normalize each channel of the RGB part to [0, 1]
#         for i in range(3):
#             ch = tmpImgt[:, :, i]
#             tmpImg[:, :, i] = (ch - np.min(ch)) / (np.max(ch) - np.min(ch) + 1e-8)
#         # Normalize each channel of the LAB part to [0, 1]
#         for i in range(3):
#             ch = tmpImgtl[:, :, i]
#             tmpImg[:, :, i+3] = (ch - np.min(ch)) / (np.max(ch) - np.min(ch) + 1e-8)
        
#         # Then standardize each channel (subtract mean and divide by std)
#         for i in range(6):
#             ch = tmpImg[:, :, i]
#             tmpImg[:, :, i] = (ch - np.mean(ch)) / (np.std(ch) + 1e-8)
    
#     elif flag == 1:
#         # With LAB color only (3 channels)
#         # Ensure 3 channels in input image:
#         if image.ndim == 2 or image.shape[2] == 1:
#             tmpImg = np.repeat(image, 3, axis=2) if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2RGB)
#         else:
#             tmpImg = image.copy()
#         tmpImg = tmpImg.astype(np.float32) / 255.0
#         tmpImg = color.rgb2lab(tmpImg)
        
#         # Normalize and standardize each channel independently
#         for i in range(3):
#             ch = tmpImg[:, :, i]
#             ch_norm = (ch - np.min(ch)) / (np.max(ch) - np.min(ch) + 1e-8)
#             tmpImg[:, :, i] = (ch_norm - np.mean(ch_norm)) / (np.std(ch_norm) + 1e-8)
    
#     else:
        
#         # With RGB color only (flag == 0)
#         image_norm = image.astype(np.float32) / 255.0
#         # Create a 3-channel image if not already 3-channel.
#         if image.ndim == 2 or image.shape[2] == 1:
#             tmpImg = np.repeat(image_norm, 3, axis=2) if image.ndim == 3 else cv.cvtColor(image_norm, cv.COLOR_GRAY2RGB)
#         else:
#             tmpImg = image_norm.copy()
        
#         # Normalize using fixed mean and std (typical for many pretrained networks)
#         # Note: These constants assume the image is in RGB.
#         tmpImg[:, :, 0] = (tmpImg[:, :, 0] - 0.485) / 0.229
#         tmpImg[:, :, 1] = (tmpImg[:, :, 1] - 0.456) / 0.224
#         tmpImg[:, :, 2] = (tmpImg[:, :, 2] - 0.406) / 0.225

    
#     # Transpose image from H x W x C to C x H x W (like PyTorch expects)
#     tmpImg = tmpImg.transpose((2, 0, 1))

#     #return {'imidx': imidx, 'image': tmpImg, 'label': tmpLbl}
#     return tmpImg

def to_tensor_lab(image, flag=0):
    """
    Process an image and its corresponding label like the original ToTensorLab,
    but without converting to torch.Tensor.
    
    Parameters:
      imidx (any): an index/identifier.
      image (np.ndarray): input image as a numpy array (assumed to be in RGB).
      label (np.ndarray): corresponding label map.
      flag (int): determines which processing branch to follow:
                  2: use both RGB and LAB channels (6 channels),
                  1: use LAB only,
                  0: use RGB only.
    
    Returns:
      dict: a dictionary with keys 'imidx', 'image', and 'label', where
            - image is transposed to shape (C, H, W),
            - label is similarly transposed if it has a channel dimension.
    """
    
    tmpImg = np.zeros((image.shape[0],image.shape[1],3))
    print("****")
    print(tmpImg.shape)
    print(image.shape)
    print("****")
    image = image/np.max(image)
    if image.shape[2]==1:
        tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
    else:
        tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
        tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

    
    # Transpose image from H x W x C to C x H x W (like PyTorch expects)
    tmpImg = tmpImg.transpose((2, 0, 1))
    print(tmpImg.shape)
    #return {'imidx': imidx, 'image': tmpImg, 'label': tmpLbl}
    return tmpImg

def preprocess_image(image_path, target_size=320, flag=0):
    """
    Read an image from disk, apply rescaling and LAB conversion.
    """
    image = cv.imread(image_path)
    w = image.shape[1]
    h = image.shape[0]
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    # image_rescaled = rescaleT(image, target_size)
    # image_processed = toTensorLab(image_rescaled, flag)

    image_rescaled = rescale_t(image, target_size)
    image_processed = to_tensor_lab(image_rescaled, flag)

    return image_processed, w, h

def center_crop(image, crop_size):
    """
    Crop the center region from an image.
    
    Parameters:
      image (np.ndarray): Image in H x W x C format.
      crop_size (int or tuple): If int, a square crop of size (crop_size, crop_size)
                                is extracted; if tuple, it is (crop_h, crop_w).
    
    Returns:
      np.ndarray: The center-cropped image.
    """
    if isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        crop_h, crop_w = crop_size
    
    h, w = image.shape[:2]
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    return image[start_y:start_y+crop_h, start_x:start_x+crop_w]

def normalize(image, mean, std):
    """
    Normalize an image given per-channel mean and std.
    
    The image is assumed to be in CHW format.
    """
    # Create an output array
    normed = np.empty_like(image)
    # For each channel, subtract mean and divide by std.
    for c in range(image.shape[-1]):
        normed[...,c] = (image[...,c] - mean[c]) / std[c]
    return normed

def softmax(x, axis=0):
    """
    Compute softmax values for each set of scores along the specified axis.
    
    Parameters:
      x (np.ndarray): Input array.
      axis (int): The axis along which to compute softmax.
      
    Returns:
      np.ndarray: An array the same shape as x with softmax computed along the given axis.
    """
    # Subtract the maximum for numerical stability.
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_e_x

def normPRED(d):
    """
    Normalize the predicted SOD probability map (numpy array) to the range [0, 1].

    Parameters:
    d (numpy.ndarray): The input probability map.

    Returns:
    numpy.ndarray: The normalized probability map.
    """
    ma = np.max(d)
    mi = np.min(d)
    # Prevent division by zero in case ma equals mi
    if ma - mi == 0:
        return d
    dn = (d - mi) / (ma - mi)
    return dn


# class BackgroundRemover():

#     def __init__(self):


#     def remove_background(self, filepath_image):


#             return imo
        
#     def remove_background_save(self, path_in, path_out, path_out_mask = None):

#         print("remove_background_save")

#         mask_torch = self.remove_background(path_in)
#         mask = mask_torch*255
#         mask = mask.astype(np.uint8)

#         img = cv.imread(path_in)
#         mask0 = mask#cv.UMat(cv.imread(mask,0))
#         #127
#         #200
#         ret,binary_mask = cv.threshold(mask0,80,255,cv.THRESH_BINARY)
#         binary_mask = np.uint8(binary_mask)
#         res = cv.bitwise_and(img,img, mask = binary_mask)

#         cv.imwrite(path_out, res)

#         if not (path_out_mask == None):
#             cv.imwrite(path_out_mask, mask)

#     def remove_background_dir(self, path_in, path_out):

#         img_name_list = glob.glob(os.path.join(path_in, "*.jpg"))

#         for img_name in img_name_list:

#             img_name_output = img_name.replace(path_in, path_out)

#             if not os.path.exists(img_name_output):
#                 self.remove_background_save(img_name, img_name_output)
#                 print(img_name.replace(path_in, path_out))

#     def apply_mask(self, input, mask, threshold):

#         mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
#         ret,binary_mask = cv.threshold(mask,threshold,255,cv.THRESH_BINARY)
#         #binary_mask = np.uint8(binary_mask)
#         #binary_mask = mask
#         print("apply mask")
#         print(input.shape)
#         print(input.dtype)
#         print(binary_mask.shape)
#         print(binary_mask.dtype)
#         res = cv.bitwise_and(input,input, mask = binary_mask)

#         # foreground_alpha = mask.astype(np.float32) / 255.0 
#         # # Create a new image to store the result with same size and type as foreground
#         # blended_image = np.zeros_like(input)

#         # # Loop through each pixel and apply alpha based on mask value
#         # for channel in range(3):  # Loop through BGR channels
#         #     blended_image[:, :, channel] = input[:, :, channel] * foreground_alpha


#         return res, binary_mask

class BackgroundRemover():

    def __init__(self):
        self.ort_sess = None

    def initialize(self, ):

        if self.ort_sess is None:
            # Load model
            model_filepath = os.path.join(MODEL_PATH, "custom_u2net.onnx")
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                })
            ]
            self.ort_sess = ort.InferenceSession(model_filepath, providers=providers)

    def inference(self, np_image, threshold=80):

        self.initialize()

        img_prec = np_image

        print("**OMTO")
        print(img_prec.dtype)
        print(img_prec.shape)
        print(np.min(img_prec))
        print(np.max(img_prec))
        print("**OMTO")

        w = np_image.shape[1]
        h = np_image.shape[0]
        img_prec = rescale_t(img_prec, 320)
        print("**rescale")
        print(img_prec.dtype)
        print(img_prec.shape)
        print(np.min(img_prec))
        print(np.max(img_prec))
        print("**rescale")
        img_prec = to_tensor_lab(img_prec, 0)
        img_prec = img_prec.astype(np.float32)
        print(img_prec.dtype)
        print(img_prec.shape)
        print(np.min(img_prec))
        print(np.max(img_prec))
        img_prec = np.expand_dims(img_prec, axis=0)

        #outputs = ort_sess.run(None, {'input': [img.numpy()]})

        outputs = self.ort_sess.run(None, {'input': img_prec})

        print(len(outputs))
        print(outputs[0].shape)

        pred = outputs[0][:,0,:,:]
        pred = normPRED(pred)
        pred = np.squeeze(pred)

        imo = cv.resize(pred, (w,h), cv.INTER_LINEAR )
        mask = imo*255
        mask = mask.astype(np.uint8)
        mask0 = mask

        ret,binary_mask = cv.threshold(mask0,threshold,255,cv.THRESH_BINARY)
        binary_mask = np.uint8(binary_mask)
        segmented = cv.bitwise_and(np_image,np_image, mask = binary_mask)
            
        return mask, binary_mask, segmented

    def inference_file(self, filename):

        np_image = cv.imread(filename)
        np_image = cv.cvtColor(np_image, cv.COLOR_BGR2RGB)

        mask, binary_mask, segmented = self.inference(np_image)

        return mask, binary_mask, segmented
    
    def inference_file_save(self, filename, output_folder
                            , progress_callback=None
                        , interruption_check=None):

        mask, binary_mask, segmented = self.inference_file(filename)

        # Save the images
        basename = os.path.basename(filename)
        print(output_folder)
        print(basename)
        output_mask_path = os.path.join(output_folder, basename.replace(".jpg", "_mask.jpg"))
        output_binary_mask_path = os.path.join(output_folder, basename.replace(".jpg", "_binary_mask.jpg"))
        output_segmented_path = os.path.join(output_folder, basename.replace(".jpg", "_segmented.jpg"))

        cv.imwrite(output_mask_path, mask)
        cv.imwrite(output_binary_mask_path, binary_mask)
        cv.imwrite(output_segmented_path, cv.cvtColor(segmented, cv.COLOR_RGB2BGR))

    def batch_processing(self, folder, output_folder, format="tiff"
                        , progress_callback=None
                        , interruption_check=None):

        processor = BatchProcessor()

        def processFunction(filepath, output_files):

            if os.path.exists(output_files[0]):
                print(f"File already exists {output_files[0]}")
            else:
                mask, binary_mask, segmented = self.inference_file(filepath)
                cv.imwrite(output_files[0], mask)
                print(f"File saved {output_files[0]}")
                cv.imwrite(output_files[1], binary_mask)
                print(f"File saved {output_files[1]}")
                cv.imwrite(output_files[2], cv.cvtColor(segmented, cv.COLOR_RGB2BGR))
                print(f"File saved {output_files[2]}")
                

        processor.batch_process(input_dir=folder
                                , output_dir=output_folder
                                , processing_fc=processFunction
                                , pattern = '**/*.' + format
                                , output_suffixes = ["mask", "binary_mask", "segmented"]
                                , progress_callback=progress_callback
                                , interruption_check=interruption_check
                                )

    # def remove_background(self, filepath_image):

    #         return imo
        
    # def remove_background_save(self, path_in, path_out, path_out_mask = None):

    #     print("remove_background_save")

    #     mask_torch = self.remove_background(path_in)
    #     mask = mask_torch*255
    #     mask = mask.astype(np.uint8)

    #     img = cv.imread(path_in)
    #     mask0 = mask#cv.UMat(cv.imread(mask,0))
    #     #127
    #     #200
    #     ret,binary_mask = cv.threshold(mask0,80,255,cv.THRESH_BINARY)
    #     binary_mask = np.uint8(binary_mask)
    #     res = cv.bitwise_and(img,img, mask = binary_mask)

    #     cv.imwrite(path_out, res)

    #     if not (path_out_mask == None):
    #         cv.imwrite(path_out_mask, mask)

    # def remove_background_dir(self, path_in, path_out):

    #     img_name_list = glob.glob(os.path.join(path_in, "*.jpg"))

    #     for img_name in img_name_list:

    #         img_name_output = img_name.replace(path_in, path_out)

    #         if not os.path.exists(img_name_output):
    #             self.remove_background_save(img_name, img_name_output)
    #             print(img_name.replace(path_in, path_out))

    # def apply_mask(self, input, mask, threshold):

    #     mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    #     ret,binary_mask = cv.threshold(mask,threshold,255,cv.THRESH_BINARY)
    #     #binary_mask = np.uint8(binary_mask)
    #     #binary_mask = mask
    #     print("apply mask")
    #     print(input.shape)
    #     print(input.dtype)
    #     print(binary_mask.shape)
    #     print(binary_mask.dtype)
    #     res = cv.bitwise_and(input,input, mask = binary_mask)

    #     # foreground_alpha = mask.astype(np.float32) / 255.0 
    #     # # Create a new image to store the result with same size and type as foreground
    #     # blended_image = np.zeros_like(input)

    #     # # Loop through each pixel and apply alpha based on mask value
    #     # for channel in range(3):  # Loop through BGR channels
    #     #     blended_image[:, :, channel] = input[:, :, channel] * foreground_alpha


    #     return res, binary_mask


class DamageClassifier():

    def __init__(self):

        self.model_name = ""
        self.results = []
        

    def initialize(self, model_name):

        #Load model

        if self.model_name != model_name:

            self.model_name = model_name

            if model_name == "Regnet":            
                model_filepath = model_filepath = os.path.join(MODEL_PATH, "regnet_x_32gf_SpidermitesModel.onnx")
            if model_name == "Resnet18":
                model_filepath = model_filepath = os.path.join(MODEL_PATH, "resnet18_SpidermitesModel.onnx")
            if model_name == "Resnet152":
                model_filepath = model_filepath = os.path.join(MODEL_PATH, "short_resnet152_SpidermitesModel_44_44.onnx")               
            if model_name == "Googlenet":
                model_filepath = model_filepath = os.path.join(MODEL_PATH, "googlenet_SpidermitesModel.onnx")

            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    # Optional: additional options can be provided, e.g.
                    #"gpu_mem_limit":  * 1024 * 1024 * 1024,
                    #"gpu_mem_limit":  6 * 1024,
                    # "cudnn_conv_algo_search": "EXHAUSTIVE",
                    # "do_copy_in_default_stream": True,
                })
            ]

            print("************** MODEL")

            print(model_name)
            print(model_filepath)

            self.ort_sess = ort.InferenceSession(model_filepath, providers=providers)

            # self.ort_sess = ort.InferenceSession(model_filepath
            #                     ,providers=ort.get_available_providers()
            #                     )

        return


    def inference(self, np_image, model_name):

        self.initialize(model_name)

        img_prec = np_image
        img_prec = img_prec.astype(np.float32)/255.0
        img_prec = rescale_t_classification(img_prec, 512)        
        img_prec = center_crop(img_prec, 512)
        img_prec = normalize(img_prec, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img_prec = img_prec.astype(np.float32)

        # img_prec = cv.imread(r"D:\local_mydev\HuggingFace\phenotyping_pipeline\prec.png")
        # img_prec = cv.cvtColor(img_prec, cv.COLOR_BGR2RGB)
        # img_prec = img_prec.astype(np.float32)/255.0

        

        img_prec = np.transpose(img_prec, (2,0,1))

        
        # save = img_prec.transpose(1,2,0)*255
        # save = cv.cvtColor(save, cv.COLOR_RGB2BGR)

        # cv.imwrite("prec.png", save)

        #img_prec = np.expand_dims(img_prec, axis=0)

        #outputs = ort_sess.run(None, {'input': [img.numpy()]})

        #outputs = self.ort_sess.run(None, {'input': img_prec})

        outputs = self.ort_sess.run(None, {'input': [img_prec]})

        np_res = outputs[0][0]
        print(np_res)
        
        if model_name != "Regnet":
            np_res = softmax(np_res)

        final_res = {'0-(No damage)': np_res[0]
                        ,'1-3-(Moderately damaged)': np_res[1]
                        ,'4-7-(Damaged)': np_res[2]
                        ,'8-10-(Severely damaged)': np_res[3]}
            
        return final_res


    def inference_file(self, filename, model_name
                    ,progress_callback=None
                    ,interruption_check=None):

        np_image = cv.imread(filename)
        np_image = cv.cvtColor(np_image, cv.COLOR_BGR2RGB)

        final_res = self.inference(np_image, model_name)

        max_key = max(final_res, key=final_res.get)

        full_path = os.path.abspath(filename)
        basename = os.path.basename(filename)
        final_res["filepath"] = full_path.replace("\\","/")
        final_res["basename"] = basename
        final_res["damage"] = max_key
        

        self.results.append(final_res)

        return final_res
    
    
    def batch_processing(self, folder, model_name, output_filename, format="tiff"
                        , progress_callback=None
                        , interruption_check=None):

        processor = BatchProcessor()

        def processFunction(filepath, output_files):

            self.inference_file(filepath, model_name)

            #self.results.append({"filepath":filepath, "class":"class1"})


        processor.batch_process(input_dir=folder
                                , output_dir=folder
                                , processing_fc=processFunction
                                , pattern = '**/*.' + format
                                , progress_callback=progress_callback
                                , interruption_check=interruption_check
                                )

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(self.results)
        df = df[['filepath','basename','damage','0-(No damage)','1-3-(Moderately damaged)','4-7-(Damaged)','8-10-(Severely damaged)']]

        # Save the DataFrame to an Excel file with formatting
        excel_file_path = output_filename
        
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            worksheet = writer.sheets['Sheet1']

            # Define the table range
            table_range = f"A1:{chr(65 + len(df.columns) - 1)}{len(df) + 1}"

            # Create a table object
            table = Table(displayName="Table1", ref=table_range)

            # Apply default table style
            style = TableStyleInfo(
                name="TableStyleMedium9",  # Choose a table style
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=True
            )
            table.tableStyleInfo = style
            worksheet.add_table(table)

            # Apply header formatting
            header_font = Font(bold=True)
            center_align = Alignment(horizontal='center', vertical='center')
            for cell in worksheet[1]:
                cell.font = header_font
                cell.alignment = center_align

            # Apply center alignment to all data cells
            for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=1, max_col=worksheet.max_column):
                for cell in row:
                    cell.alignment = center_align

        print(f"Excel file saved at: {excel_file_path}")


class BatchProcessor():

    def __init__(self):
        return
    
    def batch_process(self, input_dir, output_dir
                      , output_suffixes = ["output"]
                      , format="jpg"
                      , pattern='**/*.tiff'
                      , processing_fc=None
                      , output_format = None
                      , progress_callback=None
                      , interruption_check=None
                      ):

        if processing_fc == None:
            print("Processing function is None")
            return
        else:

            logs = []

            if output_format == None:
                output_format = format

            # Get list of files in folder and subfolders
            pattern = '**/*.'  + format
            files = glob.glob(pattern, root_dir=input_dir, recursive=True)
            total_files = len(files)
            processed_count = 0

            # Emit initial progress if needed
            if progress_callback:
                progress_callback({"processed_count":processed_count
                                    , "total_files":total_files
                                    , "status":"Initializing..."
                                    , "logs":logs
                                    , "percent": processed_count/total_files*100
                                    })

            for file in files:

                # Check for interruption request before processing each file
                if interruption_check and interruption_check():
                    print("Interruption requested, stopping batch process.")
                    break # Exit the loop

                filepath = os.path.join(input_dir, file)
                basename = os.path.basename(filepath)
                parent_dir = os.path.dirname(file)  # Fix to use the correct relative path for the file
                output_sub_dir = os.path.join(output_dir, parent_dir)  # Ensure output directory keeps the same structure

                # Create output filepath list
                output_filepaths = []
                for suffix in output_suffixes:
                    output_filepaths.append(os.path.join(output_sub_dir, basename.replace("." + format, "_" + suffix + "." + output_format)))

                if not os.path.exists(output_filepaths[0]):  # Process only if first output file does not exist

                    if not os.path.exists(output_sub_dir):  # Create subfolders if necessary
                        pathlib.Path(output_sub_dir).mkdir(parents=True, exist_ok=True)

                    logs.append(f"Processing {file}")
                    
                    processing_fc(filepath, output_filepaths) # Process and save file

                    print(file)
                    print(output_filepaths[0])
                    print("****")
                    logs.append(f"Saved {output_filepaths[0]}")

                processed_count += 1
                # Emit progress after attempting to process (or skip) each file
                if progress_callback:
                    progress_callback({"processed_count":processed_count
                                       , "total_files":total_files
                                       , "status":"Processing"
                                       , "logs": logs
                                       , "percent": processed_count/total_files*100
                                       })

            print(f"Batch process loop finished. Processed {processed_count}/{total_files} files.")

#********************************************************

def apply_lut(mask_resized):

    # Create a lookup table with shape (256, 1, 3)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    # Define colors for specific classes
    lut[1] = [255, 0, 0]   # class 1 -> red
    lut[2] = [0, 255, 0]   # class 2 -> green
    lut[3] = [0, 0, 255]   # class 3 -> blue
    # Other values will remain black (or you can define them)

    # Apply the custom colormap using applyColorMap
    colored_img = cv.applyColorMap(mask_resized, lut)

    return colored_img

def mm_inference(img_path, model_input_size, ort_sess, use_logits=True):

    if not isinstance(img_path, str):
        img = img_path.copy()
        orig_img = img.copy()
    else:
        img = cv.imread(img_path)
        orig_img = img
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Preprocess
    
    img = cv.resize(img, model_input_size, interpolation=cv.INTER_LINEAR)
    img = img.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img = np.expand_dims(img, 0)  # (1, C, H, W)

    img_prec = img.astype(np.float32)

    outputs = ort_sess.run(None, {'input': img_prec})

    orig_h, orig_w = orig_img.shape[:2]

    # Postprocess
    if use_logits:
        logits = outputs[0][0]  # shape: (4, 128, 128)
        # Upsample logits to (4, 512, 512) using bilinear interpolation
        logits_upsampled = np.stack([
            cv.resize(logits[c], (orig_w, orig_h), interpolation=cv.INTER_LINEAR_EXACT)
            for c in range(logits.shape[0])
        ])
        mask_resized = np.argmax(logits_upsampled, axis=0)  # shape: (512, 512)
        mask_resized = mask_resized.astype(np.uint8)
    else:
        mask = np.argmax(outputs[0], axis=1)[0]  # (128, 128)
        # If you want to match the original image size:        
        mask_resized = cv.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv.INTER_NEAREST)

    colored_img = apply_lut(mask_resized)

    print(np.unique(colored_img))

    return colored_img, mask_resized

# Tiled inference: split image into tiles, run inference on each, then stitch back together
def mm_inference_tiled(img_path, ort_sess, tile_size=(512, 512), overlap=64):

    if isinstance(img_path, str):
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    else:
        img = img_path.copy()
    orig_h, orig_w = img.shape[:2]

    stride_y = tile_size[0] - overlap
    stride_x = tile_size[1] - overlap

    mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
    count_map = np.zeros((orig_h, orig_w), dtype=np.uint8)

    for y in range(0, orig_h, stride_y):
        for x in range(0, orig_w, stride_x):
            y1 = y
            x1 = x
            y2 = min(y1 + tile_size[0], orig_h)
            x2 = min(x1 + tile_size[1], orig_w)
            tile = img[y1:y2, x1:x2]

            # Pad tile if needed
            pad_bottom = tile_size[0] - (y2 - y1)
            pad_right = tile_size[1] - (x2 - x1)
            if pad_bottom > 0 or pad_right > 0:
                tile = cv.copyMakeBorder(tile, 0, pad_bottom, 0, pad_right, cv.BORDER_REFLECT_101)

            # Preprocess
            colored_img, mask_tile = mm_inference(tile, tile_size, ort_sess)

            # Remove padding
            mask_tile = mask_tile[:y2 - y1, :x2 - x1]

            mask_full[y1:y2, x1:x2] += mask_tile
            count_map[y1:y2, x1:x2] += 1

    # Average overlapping regions
    mask_full = mask_full // np.maximum(count_map, 1)

    colored_img = apply_lut(mask_full)
    # plt.imshow(colored_img)
    # plt.title("Tiled Segmentation Result")
    # plt.axis('off')
    # plt.show()
    return colored_img, mask_full


class DamageSegmentor():

    def __init__(self):
        self.ort_sess = None

    def initialize(self, ):

        if self.ort_sess is None:
            # Load model
            model_filepath = os.path.join(MODEL_PATH, "phenobox_damage_segmentation.onnx")
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                })
            ]
            self.ort_sess = ort.InferenceSession(model_filepath, providers=providers)


    # def inference(self, np_image, threshold=80):


    def inference_file(self, filename):

        self.initialize()

        model_input_size = (512, 512)
        colored_img, mask = mm_inference_tiled(filename, self.ort_sess, tile_size=model_input_size, overlap=64)

        return colored_img

    def batch_processing(self, folder, output_folder, format="tiff"
                        , progress_callback=None
                        , interruption_check=None):
        
        processor = BatchProcessor()

        def processFunction(filepath, output_files):

            if os.path.exists(output_files[0]):
                print(f"File already exists {output_files[0]}")
            else:
                segmented = self.inference_file(filepath)
                # cv.imwrite(output_files[0], mask)
                # print(f"File saved {output_files[0]}")
                # cv.imwrite(output_files[1], binary_mask)
                # print(f"File saved {output_files[1]}")
                #cv.imwrite(output_files[0], cv.cvtColor(segmented, cv.COLOR_RGB2BGR))
                cv.imwrite(output_files[0], segmented)
                print(f"File saved {output_files[0]}")
                

        processor.batch_process(input_dir=folder
                                , output_dir=output_folder
                                , processing_fc=processFunction
                                , pattern = '**/*.' + format
                                , output_suffixes = ["segmented"]
                                , progress_callback=progress_callback
                                , interruption_check=interruption_check
                                )





