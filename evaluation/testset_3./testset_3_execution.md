## To execute use following command

python create_submission_moco_sun397.py \      
    --checkpoint best_model_mocov3_nabirds_lr75e-4_e150.pth \
    --data_dir ./data \
    --output submission_sun397.csv \
    --batch_size 256 \
    --num_workers 4
