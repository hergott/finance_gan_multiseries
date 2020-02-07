from enum import Enum
from shutil import rmtree
import os


### CONSTANTS, HYPERPARAMETERS AND SHARED FUNCTIONS


# Enumeration to select transformation of raw data.
class DataTransform(Enum):
    PERCENTAGE_RETURN = 1
    LOG_RETURN = 2
    NONE = 3


X_SAVE_PATH = '../data/data_saves/'
RESULTS_USED_DIR = f'../results_saves/dir_name/'
LOSSES_PATH = f"../images/losses.txt"

# data info
USE_SAVED_X = True  # if False, won't load data from 'weekly_returns' .py file

# data info
TRADING_DAYS_YEAR = 252
DATA_TRANSFORM = DataTransform.PERCENTAGE_RETURN  # using percentage returns because
# chaining returns for charts
WEEKLY = True  # can change daily data to weekly
SAMPLE_LEN = 52  # here, should be: (a) =52 for weekly data or,
# (b) =TRADING_DAYS_YEAR for daily
N_SERIES = 3
N_CONDITIONALS = 1
N_INPUTS = N_SERIES + N_CONDITIONALS

# settings for random generations
RAND_MEAN = 0.
RAND_STD = 1.

# optimization settings
UNCONDITIONAL = False
EPOCHS = 2000
MIN_EPOCHS = 1900
GRAPH_STEP = 20
L2_REGULARIZATION = 0. #0.01
DROPOUT =  0. #0.5
KEEP_PROB = 1.0 - DROPOUT
BATCH_SIZE = 79
GENERATOR_SCOPE = "GAN/Generator"
DISCRIMINATOR_SCOPE = "GAN/Discriminator"
GENERATOR_TRAINS_PER_BATCH = 1
DISCRIMINATOR_TRAINS_PER_BATCH = 5
GENERATOR_DROPOUT_ALWAYS_ON = True


def delete_files_in_folder(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)  # remove dir and all contains
        except Exception as exc:
            print(exc)

def create_moments_gif():
    from imageio import imread, mimsave
    # dir = 'C:/Users/matt_/Documents/Github/generator_gradient_post/images/moments/for_gif/small/'
    dir = 'C:/Users/matt_/Documents/Github/generator_gradient_post/images/simulations/raw/tonemapped/tm_for_gif/small/'
    gs = ['000', '025', '050', '075', '100', '150', '200', '300']

    images_dist_plot = list()
    for f in gs:
        img = imread(f'{dir}gs_{f}_sm.png')
        images_dist_plot.append(img)
    mimsave(f'{dir}simulations_gif_2secs.gif', images_dist_plot, duration=2.)