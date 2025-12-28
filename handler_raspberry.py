import time

try:
    import RPi.GPIO as GPIO
except Exception as e:
    print(f"Erro ao configurar GPIO: {e}")

class HandlerRaspberry:
    def __init__(self, pin_signal:int=17):
        self.pin_signal = None
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin_signal, GPIO.OUT, initial=GPIO.LOW)
            print(f"GPIO {pin_signal} configurado com sucesso")
            self.pin_signal = pin_signal
            
        except NameError:
            pass
        except Exception as e:
            print(f"Erro ao configurar GPIO: {e}")

    def __del__(self):
        try:
            GPIO.cleanup()
        except:
            pass

    def send_state_swap(self):
        if not self.pin_signal: return
        GPIO.output(self.pin_signal, GPIO.HIGH)
        time.sleep(0.15)
        GPIO.output(self.pin_signal, GPIO.LOW)
    

        