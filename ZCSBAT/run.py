import yaml
import argparse
from utils import DataDivision
from utils.Track_Demo import Muti_vedio_output
from utils.track import track_output
from utils import train
from utils.data_aug import data_aug


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def data_augmentation_operation(data_division_params, aug_params):
    augment_module = data_aug(data_division_params['Dataset_root'])
    for i in range(len(augment_module.Fish)):
        augment_module.VisualizeAndGenerator_result(i, num_sample=aug_params['methodA'][0], method=aug_params['methodA'][1])
        augment_module.trajectory_Generator_result(i, num_sample=aug_params['methodB'][1])


def data_division_operation(data_division_params):
    DataDivision.divide_data(data_division_params['Dataset_root'], data_division_params['original'],
                             data_division_params['format'], data_division_params['test_frac'])


def model_training_operation(training_params, data_division_params):
    train.train_model(training_params, data_division_params['Dataset_root'], pretrain=True)


def tracking_demo_operation(tracking_params, projectname):
    tracking_demo = Muti_vedio_output(tracking_params['pose_model'], tracking_params['tracking_method'],
                                      tracking_params['reid_model'], tracking_params['Anormaly_dection'],
                                      tracking_params['vediodir'], projectname,
                                      save_vedio=tracking_params['save_vedio'],
                                      save_json=tracking_params['save_json'], show=tracking_params['show'])
    tracking_demo.muti_inference()


def tracking_operation(tracking_params, projectname):
    tracker = track_output(tracking_params['joints'], tracking_params['pose_model'],
                           tracking_params['tracking_method'], tracking_params['reid_model'],
                           tracking_params['Anormaly_dection'], tracking_params['vediodir'], projectname,
                           save_vedio=tracking_params['save_vedio'],
                           save_json=tracking_params['save_json'], show=tracking_params['show'])
    tracker.muti_inference()


import argparse
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', choices=['Dataaug', 'DataDivision', 'training', 'tracking_demo', 'tracking_operation'])
    args = parser.parse_args()

    config = load_config('config.yaml')
    projectname = config['Project']
    aug_params = config['data_aug']
    data_division_params = config['data_division']
    tracking_params = config['tracking']
    training_params = config['training']

    if args.command == 'Dataaug':
        data_augmentation_operation(data_division_params, aug_params)
    elif args.command == 'DataDivision':
        data_division_operation(data_division_params)
    elif args.command == 'training':
        model_training_operation(training_params, data_division_params)
    elif args.command == 'tracking_demo':
        tracking_demo_operation(tracking_params, projectname)
    elif args.command == 'tracking_operation':
        tracking_operation(tracking_params, projectname)
    else:
        print("No valid command selected.")




if __name__ == "__main__":
    main()