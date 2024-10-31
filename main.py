import yaml
from utils import DataDivision
from utils.Track_Demo import Muti_vedio_output
from utils.track import track_output
from utils import train
from utils.data_aug import data_aug

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config('config.yaml')
    
    projectname = config['Project']
    aug_params = config['data_aug']
    data_division_params = config['data_division']
    tracking_params = config['tracking']
    training_params = config['training']
    print(config)

    #  #  Dataaug
    # augment_module = data_aug(data_division_params['Dataset_root'])
    # #Visualize and write image result
    # for i in range(len(augment_module.Fish)):
    #     augment_module.VisualizeAndGenerator_result(i,num_sample=aug_params['methodA'][0],method=aug_params['methodA'][1]) # num_sample: generate num_sample*num_sample images by one image    method:intersection normal
    #     # augment_module.VisualizeAndGenerator_result(i,num_sample=2,method='norm')
    #     augment_module.trajectory_Generator_result(i,num_sample=aug_params['methodB'][1])   # num_sample: numbers of trajectory per image
        
    
    #  DataDivision
    # DataDivision.divide_data(data_division_params['Dataset_root'], data_division_params['original'],data_division_params['format'],data_division_params['test_frac'])


    # #  model training
    # train.train_model(training_params,data_division_params['Dataset_root'],pretrain=True)
    

    # run traking demo directly
    tracking_demo = Muti_vedio_output(tracking_params['pose_model'],tracking_params['tracking_method'],tracking_params['reid_model'],tracking_params['Anormaly_dection'],tracking_params['vediodir'],projectname,
                                      save_vedio=tracking_params['save_vedio'],save_json=tracking_params['save_json'],show=tracking_params['show'])
    tracking_demo.muti_inference()
 
    # traking 
    # we will add reid model in the future 
    # tracker = track_output(tracking_params['joints'],tracking_params['pose_model'],tracking_params['tracking_method'],tracking_params['reid_model'],tracking_params['Anormaly_dection'],tracking_params['vediodir'],projectname,
    #                                   save_vedio=tracking_params['save_vedio'],save_json=tracking_params['save_json'],show=tracking_params['show'])
    # tracker.muti_inference()

if __name__ == "__main__":
    main()