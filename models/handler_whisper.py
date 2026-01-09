import os, re
import pandas as pd, numpy as np
import whisper, soundfile, kagglehub

class HandlerWhisper:
    def __init__(self):
        self.model_names = ("tiny", "base", "small", "medium", "turbo")
        self.model_index = -1
        self.model_current = None

        self.dataset_path = kagglehub.dataset_download("neehakurelli/google-speech-commands") + "/"
        self.dataset_paths = pd.read_csv(self.dataset_path + "testing_list.txt", header=None, sep='/')
        self.dataset_paths.columns = ["word", "path"]
        self.dataset_paths["path"] = self.dataset_paths["word"] + "/" + self.dataset_paths["path"]

        self.sample = None
        self.sample_file = None


        os.makedirs("inference_output", exist_ok=True)
        self.dir_output = "inference_output/whisper"
        os.makedirs(self.dir_output, exist_ok=True)

        self.result_output_file = None
        self.result = None

    def __del__(self):
        self.output_close()

    def output_close(self):
        if self.result_output_file is not None:
            self.result_output_file.close()
            self.result_output_file = None

    def load_next_model(self) -> bool:
        self.model_index += 1
        self.output_close()

        if self.model_index < len(self.model_names):
            model_name = self.model_names[self.model_index]
            del(self.model_current)
            self.model_current = whisper.load_model(model_name)
            print(f"\nCurrent Model: {model_name}")

            output_path = f"{self.dir_output}/{model_name}.csv"
            self.result_output_file = open(output_path, "w")
            self.result_output_file.write("word,file,transcription\n")
            print(f"Output file: {output_path}")
            
            return True
        
        else:
            self.model_current = None
            return False


    def before_inference(self) -> None:
        self.sample_file = None
        while self.sample_file is None:
            self.sample = self.dataset_paths.sample(1)
            sample_path = self.sample["path"].to_list()[0]
            self.sample_file = soundfile.read(self.dataset_path + sample_path, dtype="float32")[0]
        print(f"\tLoaded audio: {sample_path}", end="")
        
    def inference(self) -> None:
        print(f"\tTranscribing audio...", end="")
        self.result = self.model_current.transcribe(
            self.sample_file,
            language="en",
            fp16=False
        )

    def after_inference(self) -> None:
        self.result = self.result["text"]
        self.result = HandlerWhisper.transcription_format(self.result)
        if self.result == "": self.result = "None"

        sample = (self.sample["word"].to_list()[0], self.sample["path"].to_list()[0])
        self.result_output_file.write(
            f"{sample[0]},{sample[1]},{self.result}\n"
        )
        print(f"\tResult written {self.result}", end="")

    def transcription_format(transcription:str) -> str:
        transcription = transcription.replace(' ', '').lower()
        transcription = re.sub(r'[^a-zA-Z0-9]', '', transcription)
        transcription = converter_numeros_por_extenso(transcription)
        return transcription

def converter_numeros_por_extenso(texto:str) -> str:
    numeros = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    }
    for num, extenso in numeros.items():
        texto = re.sub(r'\b' + num + r'\b', extenso, texto)
    return texto
