from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_florence_2_detection.infer_florence_2_detection_process import InferFlorence2DetectionFactory
        return InferFlorence2DetectionFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_florence_2_detection.infer_florence_2_detection_widget import InferFlorence2DetectionWidgetFactory
        return InferFlorence2DetectionWidgetFactory()
