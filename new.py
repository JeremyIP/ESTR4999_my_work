import pandas as pd
import numpy as np
import os

hist_len_list_01 = [1, 1, 1]
decimal_value = sum(bit << i for i, bit in enumerate(reversed(hist_len_list_01)))
print(decimal_value)