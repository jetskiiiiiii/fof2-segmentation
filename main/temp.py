import numpy as np
import pandas as pd

start_time = "00:00"
end_time = "23:45"
time_intervals = pd.date_range(start=start_time, end=end_time, freq="15min").time

float_indices = np.arange(0, 24, 0.25)

print(float_indices)

print(len(float_indices))
