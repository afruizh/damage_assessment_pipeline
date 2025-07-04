from bgremover import DamageClassifier
from bgremover import BackgroundRemover
from bgremover import DamageSegmentor



class Processor():

    def __init__(self, params, progress_callback = None, interruption_check = None):
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
        
        elif task == "single_segmentation":
            # Perform batch segmentation
            input_file = self.params.get("input_file")
            output_folder = self.params.get("output_folder")

            # Initialize the BackgroundRemover and perform batch processing
            self.background_remover = BackgroundRemover()
            self.background_remover.inference_file_save(input_file
                                                    , output_folder
                                                    , progress_callback = self.progress_callback
                                                    , interruption_check = self.interruption_check)
            
        elif task == "single_damage_classification":
            # Perform batch segmentation
            input_file = self.params.get("input_file")
            model = self.params.get("model")
            self.damage_classifier = DamageClassifier()
            res = self.damage_classifier.inference_file(input_file
                                                    , model
                                                    , progress_callback = self.progress_callback
                                                    , interruption_check = self.interruption_check)
            results.update({"results":res})

        elif task == "batch_phenobox_damage_segmentation":

            input_folder = self.params.get("input_folder")
            output_folder = self.params.get("output_folder")

            damage_segmentor = DamageSegmentor()
            res = damage_segmentor.batch_processing(input_folder
                                              , output_folder
                                              , progress_callback = self.progress_callback
                                              , interruption_check = self.interruption_check
                                            )
            results.update({"results":res})

        else:
            results.update({"status": "error", "message": "Invalid task specified."})



        #****************

        return results

    
