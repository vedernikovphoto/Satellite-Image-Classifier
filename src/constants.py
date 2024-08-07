import os

current_file_dir = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = os.path.abspath(os.path.join(current_file_dir, '..'))
EXPERIMENTS_PATH = os.path.join(PROJECT_PATH, 'experiments')
