import time

try:
    import gpiozero
except Exception as e:
    print(f"Erro ao importar GPIO: {e.args[0]}")

class HandlerRaspberry:
    def __init__(self, pin_signal:int=17):
        self.pin = None
        try:
            self.pin = gpiozero.DigitalOutputDevice(pin_signal)
            self.pin.off()
        except Exception as e:
            print("Erro ao iniciar o pino: ", e.args[0])

    def send_state_swap(self):
        if not self.pin: return
        self.pin.on()
        time.sleep(0.050)
        self.pin.off()
        time.sleep(0.050)
    