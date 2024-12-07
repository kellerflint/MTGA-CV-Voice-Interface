from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO
from PIL import Image
import uuid

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        path_to_model = "../../yolo11n_mtga_train_40.pt"
        self.set("model_version", "0.0.1")
        self.yolo_model = YOLO(path_to_model)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]

        ### My Implementation ###

        predictions = []

        for task in tasks:
            # Download image to a local path
            image_path = self.get_local_path(task['data']['image'], task_id=task['id'])
            image = Image.open(image_path)

            # Perform inference with YOLO
            results = self.yolo_model.predict(image, conf=0.25)

            # save result to file
            results[0].save('test_out.png')

            # Convert YOLO predictions to Label Studio format
            task_predictions = []
            for result in results[0].boxes:
                x_min, y_min, x_max, y_max = result.xyxy[0].tolist()
                class_id = int(result.cls[0])
                confidence = float(result.conf[0])
                
                # Convert YOLO coordinates to percentages (Label Studio format)
                width = x_max - x_min
                height = y_max - y_min
                
                task_predictions.append({
                    "id": str(uuid.uuid4()),
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": (x_min / image.width) * 100,
                        "y": (y_min / image.height) * 100,
                        "width": (width / image.width) * 100,
                        "height": (height / image.height) * 100,
                        "rotation": 0,
                        "rectanglelabels": [self.yolo_model.names[class_id]]
                    },
                    "score": confidence
                })

            predictions.append({
                "result": task_predictions,
                "score": max([pred["score"] for pred in task_predictions], default=0),
                "model_version": self.get("model_version")
            })
        print(f'Predictions: {ModelResponse(predictions=predictions)}')
        return ModelResponse(predictions=predictions)
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

