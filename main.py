import argparse
import yaml
import warnings
import torch

from model.graphTransformer import ExpGraphTransformer

warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')

import random
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajGAT")
    parser.add_argument("-C", "--config", type=str)
    parser.add_argument("-G", "--gpu", type=str, default="0")
    parser.add_argument("-L", "--load-model", type=str, default=None)
    parser.add_argument("-J", "--just_embedding", action="store_true")


    parser.add_argument("-dataset_name", "--dataset_name", type=str)
    parser.add_argument("-dist_type", "--dist_type", type=str)
    parser.add_argument("-random_seed", "--random_seed")

    
    args = parser.parse_args()


    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    config["data"]             = args.dataset_name
    config["dis_type"]         = args.dist_type
    config["seed"]             = int(args.random_seed)
    

    if "0_porto_10000" in args.dataset_name:
        config["data_features"] = [-8.60743806083366, 0.0746714476194787, 41.17521080179066, 0.08419833953546144]
        config["x_range"] = [-13.172526, -6.724476]
        config["y_range"] = [39.256344, 42.124779]
        config["train_traj_path"]  = '/mnt/data_hdd1/czh/Neutraj/Test_porto/traj_list'
        config["dis_matrix_path"] = '/mnt/data_hdd1/czh/Neutraj/Test_porto/' + args.dist_type + '_train_distance_matrix_result'

    elif "0_porto_all" in args.dataset_name:
        config["data_features"] = [-8.619871526273565, 0.027390680370463525, 41.16061848396889,  0.017590713460087164]
        config["x_range"] = [-8.73, -8.5]
        config["y_range"] = [41.10, 41.24]
        # config["x_range"] = [-15.630759, -3.930948]
        # config["y_range"] = [36.886104, 45.657225]
        config["train_traj_path"]  = '/mnt/data_hdd1/czh/Neutraj/' + args.dataset_name + '/traj_list'
        config["dis_matrix_path"]  = '/mnt/data_hdd1/czh/Neutraj/' + args.dataset_name + '/' + args.dist_type + '_train_distance_matrix_result'
        
        config["val_traj_path"]    = '/mnt/data_hdd1/czh/Neutraj/' + args.dataset_name + '/traj_list'
        config["test_matrix_path"] = '/mnt/data_hdd1/czh/Neutraj/' + args.dataset_name + '/' + args.dist_type + '_train_distance_matrix_result'
    elif "0_geolife" in args.dataset_name:
        config["data_features"] = [116.35919110593036, 0.05556733096599747, 39.970607956390225, 0.03599086900045147]
        config["x_range"] = [116.200047, 116.499288]
        config["y_range"] = [39.851057, 40.0699999]
        config["train_traj_path"]  = '/mnt/data_hdd1/czh/TrajGAT/' + args.dataset_name + '/train_traj_list'
        config["dis_matrix_path"]  = '/mnt/data_hdd1/czh/TrajGAT/' + args.dataset_name + '/' + args.dist_type + '_train_distance_matrix_result'
        config["val_traj_path"]    = '/mnt/data_hdd1/czh/TrajGAT/' + args.dataset_name + '/traj_list'
        config["test_matrix_path"] = '/mnt/data_hdd1/czh/TrajGAT/' + args.dataset_name + '/' + args.dist_type + '_test_distance_matrix_result'
    
    else:
        print("args.dataset_name ERROR")
        exit()
    
    
    

    print("Args in experiment:")
    print(config)
    print("GPU:", args.gpu)
    print("Load model:", args.load_model)
    print("Store embeddings:", args.just_embedding, "\n")

    torch.set_num_threads(7) 

    if args.just_embedding:
        ExpGraphTransformer(config=config, gpu_id=args.gpu, load_model=args.load_model, just_embeddings=args.just_embedding).embedding()
    else:
        ExpGraphTransformer(config=config, gpu_id=args.gpu, load_model=args.load_model, just_embeddings=args.just_embedding).train()

    torch.cuda.empty_cache()
