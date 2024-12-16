import gradio as gr
from bgremover import BackgroundRemover
from bgremover import DamageClassifier
from bgremover import clear
from bgremover import ColorCheckerDetector
from bgremover import Segmentor
import rasterio
import os
from PIL import Image
from gradio_client import Client

PRELOAD_MODELS = False

if PRELOAD_MODELS:
    backgroundRemover = BackgroundRemover()
    damage_classifier =  DamageClassifier()
    segmentor = Segmentor()

def process(input_img):

    if PRELOAD_MODELS:
        global backgroundRemover
    else:
        backgroundRemover = BackgroundRemover()

    output_mask, output_img = backgroundRemover.remove_background_gradio(input_img)


    return [output_img, output_mask]

def process_classification(input_img, model_name):

    if PRELOAD_MODELS:
        global damage_classifier
    else:
        damage_classifier =  DamageClassifier()

    res = damage_classifier.inference(input_img, model_name)

    #return {'No damage': 0.1, 'Moderately damaged': 0.1,'Damaged': 0.7, 'Severy damaged': 0.1}
    return res


def segment_plant(threshold, input_im, im_mask):

    if PRELOAD_MODELS:
        global backgroundRemover
    else:
        backgroundRemover = BackgroundRemover()

    print("segment plant", threshold)

    res, mask = backgroundRemover.apply_mask(input_im, im_mask, threshold)

    return res, mask

def rectangle(im, im_mask):

    colorCheckerDetector = ColorCheckerDetector()


    return colorCheckerDetector.process(im_mask, im)

def get_file_content(file):
	with rasterio.open(file) as src:
		# Read the image data
		image_data = src.read()
		image = Image.fromarray((image_data[0] * 255).astype(np.uint8))
	return (gr.Image(value=image, type="pil"))

def on_img_color_load(input):
    print("on_img_color_load")
    print(input)

def run_anything_task(input_image):

    text_prompt = "color-checker"
    task_type = "inpainting"

    #text_prompt = "rocket"

    if PRELOAD_MODELS:
        global segmentor
    else:
        segmentor = Segmentor()

    return segmentor.process(input_image, text_prompt)

with gr.Blocks(title="Phenotyping pipeline") as demo:

    # gr.Markdown(
    # """
    # # Phenotyping pipeline
    # Modular phenotyping pipeline.
    # """)

    big_block = gr.HTML("""

    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: white
            margin: 0;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px;
            color: #fff;
        }

        hr {
            border: 1px solid #ddd;
            margin: 5px;
        }

    </style>

    <header>
        <div style="display: flex; align-items: center;">
            <div style="text-align: left;">
            <h1>Phenotyping pipeline</h1>
            <p>Modular phenotyping pipeline.</p>
            <h3>Tropical Forages Program</h3>
            <p><b>Authors: </b>Andres Felipe Ruiz-Hurtado, Juan Andrés Cardoso Arango</p>
            <p></p>            
        </div>
        </div>
        <div style="background-color: white; padding: 5px; border-radius: 15px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
                        <img src="https://alliancebioversityciat.org/sites/default/files/styles/1920_scale/public/images/Alliance%20Logo%20Refresh-color.jpg" alt="Logo" width="200" height="100">
                    </div>
    </header>   
    
    """)

    input_im = gr.Image(render=False)
    im_result = gr.Image(render=False)
    im_mask = gr.Image(render=False)
    im_masked = gr.Image(render=False)

    im_color = gr.Image(render=False)
    im_color_orginal = gr.Image(render=False)
    im_color.change(on_img_color_load, im_color)

    im_color_checker_mask = gr.Image(render=False)

    
    
    with gr.Tab("Damage Classification"):

        model_option = gr.Dropdown(
            ["Regnet", "Resnet18", "Resnet152", "Googlenet"]
            , label="Classification model"
            , info="The classification model to use for inference"
            , value="Regnet"
        )

        gr.Interface(fn=process_classification
                    , inputs= [input_im, model_option]
                    , outputs="label" 
                    , examples = [
                        ["183_Week_1_(28th_Aug_-_1st_Sept.)_2023_nd.jpg"]                        
                        ,["20_WEEK_5_(_FIELD_A)_md.jpg"]
                        ,["30_WEEK_5_(_FIELD_A)_damaged.jpg"]
                        ,["25_WEEK_4_(_Field_A)_sd.jpg"]
                        #,["30_WEEK_4_(_Field_A)_sd.jpg"]
                    ]
                    )
        #gr.Button("Classify")

    with gr.Tab("Color Checker detection"):

        #gr.Interface(fn=process_classification, inputs= input_im, outputs="label" )
        #gr.Button("Classify")
        gr.Interface(fn=run_anything_task, inputs= input_im, outputs=gr.Gallery() )

    with gr.Tab("Color Calibration"):

        #gr.Interface(fn=process_classification, inputs= input_im, outputs="label" )
        #gr.Button("Classify")
        gr.Interface(fn=rectangle
                    , inputs= [input_im, im_color_checker_mask]
                    , outputs=gr.Gallery()
                    , examples = [["264_WEEK_5_(_FIELD_A).jpg","264_mask.jpg"]]
                    )
        gr.Button("Calibrate")
    
    with gr.Tab("Plant segmentation"):

        with gr.Column(scale=1):
            #gr.Interface(fn=process, inputs= gr.Image(), outputs=[im_result, "image"] )
            gr.Interface(fn=process, inputs= input_im, outputs=[im_result, im_mask] )

            slider_thresh = gr.Slider(minimum=0, maximum=255, value=100, step=1, label="Threshold"
                    , info="Segmentation threshold", interactive=True)
            slider_thresh.release(fn=segment_plant, inputs = [slider_thresh, input_im, im_mask], outputs = [gr.Image(), gr.Image()])

            #button = gr.Button("Clip")
            #button.click()
            #gr.Image(value=im_masked)

    # with gr.Tab("Damage segmentation"):

    #     gr.Button("Damage")

    # with gr.Tab("Batch processing"):

    #     gr.Button("Run")

    # with gr.Tab("Batch processing"):

    #     gr.Interface(fn=run_anything_task, inputs= input_im, outputs= gr.Gallery())

    #with gr.Tab("Tests"):

        # gr.Markdown("# Preview Images:")
        # with gr.Group(visible=True):
        #     with gr.Row(visible=True):
        #         preview = gr.FileExplorer( scale      = 1,
        #                     glob        = "*.tif",
        #                     value       = ["./"],
        #                     file_count  = "single",
        #                     root_dir    = "./",
        #                     elem_id     = "file",
        #                     every= 1,
        #                     interactive=True
        #                     )

        #         #image = gr.Image(type="pil")
        #         image = gr.Image()
        # preview.change(get_file_content, preview, image)



    

if __name__ == "__main__":
    #demo.launch(show_api=False)
    #client = Client(demo)
    #demo.launch(show_api=True, server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7861)))
    demo.launch(server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)), share=False)

    
