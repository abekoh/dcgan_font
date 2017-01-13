# -*- coding: utf-8 -*-
import time

def format_sec(elapsed_time):
    h = int(elapsed_time / (60 * 60))
    m = int((elapsed_time - (h * 60 * 60)) / 60)
    s = int(elapsed_time - (h * 60 * 60) - (m * 60))
    formatted_elapsed_time = 'elapsed time: {0:02d}:{1:02d}:{2:02d}'.format(h, m, s)
    return formatted_elapsed_time

class StopWatch():
    def start(self):
        self.start_time = time.time()

    def view(self):
        now_time = int(time.time() - self.start_time)
        formatted_now_time = format_sec(now_time)
        print (formatted_now_time)
        return formatted_now_time

    def stop(self):
        self.end_time = time.time()
        self.elapsed_time = int(self.end_time - self.start_time)
        return self.elapsed_time

    def format_print(self):
        formatted_elapsed_time = format_sec(self.elapsed_time)
        print (formatted_elapsed_time)
        return formatted_elapsed_time
