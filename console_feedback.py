import threading
import sys
import time

class ActivitySpinner:
    def __init__(self, message="Verarbeite..."):
        self.message = message
        self.stop_event = threading.Event()
        self.thread = None
        self.symbols = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" # Ein schickes Punkte-Muster

    def _spin(self):
        while not self.stop_event.is_set():
            for symbol in self.symbols:
                if self.stop_event.is_set():
                    break
                # \r springt zum Anfang der Zeile zurück
                sys.stdout.write(f"\r{self.message} {symbol} ")
                sys.stdout.flush()
                time.sleep(0.1)

    def start(self, new_message=None):
        if new_message:
            self.message = new_message
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def stop(self):
        if self.thread:
            self.stop_event.set()
            self.thread.join()
            # Zeile löschen
            sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
            sys.stdout.flush()