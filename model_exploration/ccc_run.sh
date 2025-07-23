export OMP_NUM_THREADS=5


nvidia-smi

torchrun --nproc_per_node=auto --master-port=29507 st_trainer_ddp_hf.py \
    --model_name "nasa-impact/indus-sde-st-v0.1" \
    --num_train_epochs 1 \
    --batch_size 64 \
    --gradient_accumulation_steps 8 \
    --lr 5e-5 \
    --eval_and_save_steps 2000 \
    --output_base "../training_output" \
    --max_datapoints_per_src_for_eval 0 \
    --warmup_ratio 0.02 \
    --cosine_cycle_steps_custom 8000 \
    
    
    #--wb_mode "offline"

    
    
    #--resume_checkpoint_path  "/dccstor/knowledge-hub/project/st-training-workflow/training_output/nrows_None__nsrc_None/timestamp_20250710_11-30-47/indus-sde-st-v0.1/checkpoints/checkpoint-268500" \
    #--resume_run_id 1ogtkl75 

    #--wb_mode "offline"
    
    
    #--resume_checkpoint_path  "/dccstor/knowledge-hub/project/st-training-workflow/training_output/nrows_None__nsrc_None/timestamp_20250710_11-30-47/indus-sde-st-v0.1/checkpoints/checkpoint-268500" \
    #--resume_run_id wbmtxedc \
    
    #--wb_mode "offline"
    #--lr 2e-5 \ #first run
