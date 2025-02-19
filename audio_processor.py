import sounddevice as sd
import numpy as np
from queue import Queue

class AudioProcessor:
    def __init__(self, samplerate=16000, blocksize=4000):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.audio_queue = Queue()
        self.stream = None

    def start_stream(self):
        """Start audio stream capture"""
        def callback(indata, frames, time, status):
            self.audio_queue.put(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            callback=callback,
            dtype=np.float32
        )
        self.stream.start()

    def get_chunk(self):
        """Get latest audio chunk"""
        try:
            return self.audio_queue.get_nowait()
        except:
            return None

    def stop_stream(self):
        """Stop audio stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
   
