# export OMP_NUM_THREADS=1
export OMP_NUM_THREADS=5


dataset_list=(0_porto_all)
dist_type_list=(haus)
seed_list=(666)
for dataset in ${dataset_list[@]}; do
    for seed in ${seed_list[@]}; do
        for dist_type in ${dist_type_list[@]}; do
            train_flag=${dataset}_${dist_type}_${seed}
            echo ${train_flag}
            export OMP_NUM_THREADS=4
            nohup python main.py --config=config.yaml --gpu=0 --dataset_name=${dataset} --dist_type=${dist_type} --random_seed=${seed} > train_log/${train_flag} &
            PID0=$!
            wait $PID0
        done
    done
done