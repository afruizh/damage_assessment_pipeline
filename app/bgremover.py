import cv2 as cv
import numpy as np
from PIL import Image
import glob
import pathlib

import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.table import Table, TableStyleInfo

import onnxruntime as ort
import cv2 as cv
import numpy as np

MODEL_PATH = "./models"


def rescale_t(image, target_size=320):
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
    resized = cv.resize(image, (new_w, new_h))
    return resized

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
    
    if flag == 2:
        # With RGB and LAB channels (6 channels)
        H, W = image.shape[:2]
        tmpImg = np.zeros((H, W, 6), dtype=np.float32)
        
        # Ensure we have a 3-channel image: if single channel, replicate it.
        if image.ndim == 2 or image.shape[2] == 1:
            tmpImgt = np.repeat(image, 3, axis=2) if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        else:
            tmpImgt = image.copy()
        
        # Convert the RGB image to LAB (scikit-image expects float images in [0,1])
        # (Assumes image is in RGB; if using OpenCV, convert BGR to RGB before calling this.)
        tmpImgt = tmpImgt.astype(np.float32) / 255.0
        tmpImgtl = color.rgb2lab(tmpImgt)
        
        # Normalize each channel of the RGB part to [0, 1]
        for i in range(3):
            ch = tmpImgt[:, :, i]
            tmpImg[:, :, i] = (ch - np.min(ch)) / (np.max(ch) - np.min(ch) + 1e-8)
        # Normalize each channel of the LAB part to [0, 1]
        for i in range(3):
            ch = tmpImgtl[:, :, i]
            tmpImg[:, :, i+3] = (ch - np.min(ch)) / (np.max(ch) - np.min(ch) + 1e-8)
        
        # Then standardize each channel (subtract mean and divide by std)
        for i in range(6):
            ch = tmpImg[:, :, i]
            tmpImg[:, :, i] = (ch - np.mean(ch)) / (np.std(ch) + 1e-8)
    
    elif flag == 1:
        # With LAB color only (3 channels)
        # Ensure 3 channels in input image:
        if image.ndim == 2 or image.shape[2] == 1:
            tmpImg = np.repeat(image, 3, axis=2) if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        else:
            tmpImg = image.copy()
        tmpImg = tmpImg.astype(np.float32) / 255.0
        tmpImg = color.rgb2lab(tmpImg)
        
        # Normalize and standardize each channel independently
        for i in range(3):
            ch = tmpImg[:, :, i]
            ch_norm = (ch - np.min(ch)) / (np.max(ch) - np.min(ch) + 1e-8)
            tmpImg[:, :, i] = (ch_norm - np.mean(ch_norm)) / (np.std(ch_norm) + 1e-8)
    
    else:
        # With RGB color only (flag == 0)
        image_norm = image.astype(np.float32) / 255.0
        # Create a 3-channel image if not already 3-channel.
        if image.ndim == 2 or image.shape[2] == 1:
            tmpImg = np.repeat(image_norm, 3, axis=2) if image.ndim == 3 else cv.cvtColor(image_norm, cv.COLOR_GRAY2RGB)
        else:
            tmpImg = image_norm.copy()
        
        # Normalize using fixed mean and std (typical for many pretrained networks)
        # Note: These constants assume the image is in RGB.
        tmpImg[:, :, 0] = (tmpImg[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (tmpImg[:, :, 1] - 0.456) / 0.224
        tmpImg[:, :, 2] = (tmpImg[:, :, 2] - 0.406) / 0.225

    
    # Transpose image from H x W x C to C x H x W (like PyTorch expects)
    tmpImg = tmpImg.transpose((2, 0, 1))

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

            self.ort_sess = ort.InferenceSession(model_filepath, providers=providers)

            # self.ort_sess = ort.InferenceSession(model_filepath
            #                     ,providers=ort.get_available_providers()
            #                     )

        return


    def inference(self, np_image, model_name):

        self.initialize(model_name)

        img_prec = np_image
        img_prec = rescale_t(img_prec, 512)
        img_prec = center_crop(img_prec, 512)
        img_prec = normalize(img_prec, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img_prec = img_prec.astype(np.float32)

        img_prec = np.transpose(img_prec, (2,0,1))
        img_prec = np.expand_dims(img_prec, axis=0)

        #outputs = ort_sess.run(None, {'input': [img.numpy()]})

        outputs = self.ort_sess.run(None, {'input': img_prec})

        np_res = outputs[0][0]
        
        if model_name != "Regnet":
            np_res = softmax(np_res)

        final_res = {'0-(No damage)': np_res[0]
                        ,'1-3-(Moderately damaged)': np_res[1]
                        ,'4-7-(Damaged)': np_res[2]
                        ,'8-10-(Severely damaged)': np_res[3]}
            
        return final_res


    def inference_file(self, filename, model_name):

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
    
    
    def batch_processing(self, folder, model_name, output_filename, format="tiff"):

        processor = BatchProcessor()

        def processFunction(filepath, output_files):

            self.inference_file(filepath, model_name)

            #self.results.append({"filepath":filepath, "class":"class1"})


        processor.batch_process(input_dir=folder, output_dir=folder, processing_fc=processFunction, pattern = '**/*.' + format )

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
                parent_dir = os.path.dirname(file)  # Fix to use the correct relative path for the file
                output_sub_dir = os.path.join(output_dir, parent_dir)  # Ensure output directory keeps the same structure

                # Create output filepath list
                output_filepaths = []
                for suffix in output_suffixes:
                    output_filepaths.append(os.path.join(output_sub_dir, basename.replace("." + format, "_" + suffix + "." + output_format)))

                if not os.path.exists(output_filepaths[0]):  # Process only if first output file does not exist

                    if not os.path.exists(output_sub_dir):  # Create subfolders if necessary
                        pathlib.Path(output_sub_dir).mkdir(parents=True, exist_ok=True)

                    processing_fc(filepath, output_filepaths) # Process and save file

                    print(file)
                    print(output_filepaths[0])
                    print("****")