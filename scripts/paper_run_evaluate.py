import evaluate
from tqdm import tqdm
import glob
import os

paper_run_base_dir = '/home/shush/profile/QuantPred/paper_runs/*/*'
projects_to_evaluate = [dir for dir in glob.glob(paper_run_base_dir) if ('finetune' not in dir) and ('fintune' not in dir)
                    and ('binary' not in dir) and ('coverage' not in dir)]

output_dir = '../paper_run_evaluations'
data_dir = '/home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/'
idr_data_dir_pattern = '/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/shush/15_IDR_test_sets_6K/cell_line_*/i_6144_w_1/'
for project_dir in tqdm(projects_to_evaluate):
    if 'new_models' in project_dir:
        batch_size = 2
    else:
        batch_size = 32
    if os.path.isfile(os.path.join(output_dir, os.path.basename(project_dir)+'.csv')):
        print('Skipping '+project_dir)
    else:
        print('Processing ', os.path.join(output_dir, project_dir+'.csv'))
        evaluate.evaluate_project(data_dir=data_dir, idr_data_dir_pattern=idr_data_dir_pattern,
                                  run_dir_list=None, project_dir=project_dir, wandb_project_name=None, wandb_dir=None,
                                  output_dir='../paper_run_evaluations', output_prefix=None, batch_size=batch_size)
