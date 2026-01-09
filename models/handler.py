from typing import Literal

from models.handler_whisper import HandlerWhisper
from models.handler_yolo import HandlerYolo

import time

class HandlerModel:
    def __init__(self, type:Literal["whisper", "yolo"]):
        self.type = type
        self.handler_model = None

        match self.type:
            case "whisper":
                self.handler_model = HandlerWhisper()

            case "yolo":
                self.handler_model = HandlerYolo()

            case _:
                raise ValueError('ModelPicker parameter "type" must be an str: "whisper" or "yolo"')

    def load_next_model(self) -> bool:
        return self.handler_model.load_next_model()

    def before_inference(self) -> None:
        start_time = time.time()
        self.handler_model.before_inference()
        print(f" time: {(time.time()-start_time):.2f}")

    def inference(self) -> None:
        start_time = time.time()
        self.handler_model.inference()
        print(f" time: {(time.time()-start_time):.2f}")
    
    def after_inference(self) -> None:
        start_time = time.time()
        self.handler_model.after_inference()
        print(f" time: {(time.time()-start_time):.2f}\n")
