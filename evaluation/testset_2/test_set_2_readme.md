

To execute use following command
'''
python create_submission_moco_miniimagenet.py \
    --checkpoint best_model_mocov3_e300_lr15e-4.pth \
    --data_dir ./data \
    --output submission_miniimagenet.csv \
    --batch_size 256 \
    --num_workers 4
'''
