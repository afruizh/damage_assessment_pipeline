from app.interface.processor import Processor

params = {
    'task': 'single_segmentation',
    'input_file': r"D:\temporal\input\New folder\20_WEEK_5_(_FIELD_A)_md.jpg",
    'output_folder': r"D:\temporal\input\New folder"
}

proc = Processor(params)
proc.run()