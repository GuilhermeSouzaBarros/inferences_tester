from sys import argv
import time

from models.handler import HandlerModel
from handler_raspberry import HandlerRaspberry

class INFERENCE:
    AMOUNT = 100
    TIME_ACTIVE = 600
    TIME_INTERVAL = 120

if __name__ == "__main__":
    if len(argv) < 2:
        print(
            "Model type not specified, expected whisper or yolo\n" \
            "e.g.\tpython3 main.py whisper"
        )
        exit()
    
    models = HandlerModel(argv[1])
    signal = HandlerRaspberry()

    interval_last = time.time()
    while models.load_next_model():
        for x in range(INFERENCE.AMOUNT):
            if time.time() - interval_last > INFERENCE.TIME_ACTIVE:
                print(f"Interval started ({INFERENCE.TIME_INTERVAL}s)")
                time.sleep(INFERENCE.TIME_INTERVAL)
                interval_last = time.time()
                print("Interval ended\n")

            signal.send_state_swap()
            models.before_inference()
            
            signal.send_state_swap()
            models.inference()

            signal.send_state_swap()
            models.after_inference()

            signal.send_state_swap()
