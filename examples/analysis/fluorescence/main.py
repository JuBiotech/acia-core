import os

from fluorescence_analysis import main as fa_main
from datapoints_analysis import main as da_main
from videoRender import main as vr_main
from config import basepath

if __name__ == '__main__':
    os.makedirs(basepath)

    # execute the three analysis steps to produce fluorescence results
    fa_main()
    da_main()
    vr_main()