from bgremover import DamageClassifier
from bgremover import BackgroundRemover


class Processor():

    def __init__(self, params, progress_callback, interruption_check):
        self.params = params
        self.progress_callback = progress_callback
        self.interruption_check = interruption_check

    def run(self):

        results = self.params


        task = self.params.get("task")

        #***************
        if task == "batch_damage_classification":
            # Perform batch damage classification
            input_folder = self.params.get("input_folder")
            model = self.params.get("model")
            output_file = self.params.get("output_file")

            # Initialize the DamageClassifier and perform batch processing
            self.damage_classifier = DamageClassifier()
            self.damage_classifier.batch_processing(input_folder
                                                    , model
                                                    , output_file
                                                    , progress_callback = self.progress_callback
                                                    , interruption_check = self.interruption_check)

        elif task == "batch_segmentation":

            # Perform batch segmentation
            input_folder = self.params.get("input_folder")
            output_folder = self.params.get("output_folder")

            # Initialize the BackgroundRemover and perform batch processing
            self.background_remover = BackgroundRemover()
            self.background_remover.batch_processing(input_folder
                                                    , output_folder
                                                    , progress_callback = self.progress_callback
                                                    , interruption_check = self.interruption_check)

        #****************

        return results

    
