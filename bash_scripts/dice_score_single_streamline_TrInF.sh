#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
# Path to your config file
config_file="cfgs/dataset_configs/single_streamline_TrInF.yaml"

# Define the subject_ids blocks (pairs of subject_ids lines to toggle between)
subject_blocks=(
    "subject_ids: ['sub-1235']"
    "subject_ids: ['sub-1281']"
    "subject_ids: ['sub-1019']"
    "subject_ids: ['sub-1047']"
    "subject_ids: ['sub-1046']"
    "subject_ids: ['sub-1050']"
    "subject_ids: ['sub-1080']"
    "subject_ids: ['sub-1135']"
    "subject_ids: ['sub-1156']"
    "subject_ids: ['sub-1178']"
    "subject_ids: ['sub-1217']"
    "subject_ids: ['sub-1061']"
    "subject_ids: ['sub-1115']"
    "subject_ids: ['sub-1146']"
    "subject_ids: ['sub-1164']"
    "subject_ids: ['sub-1191']"
    "subject_ids: ['sub-1222']"
    "subject_ids: ['sub-1245']"
    "subject_ids: ['sub-1024']"  
    "subject_ids: ['sub-1049']"
    "subject_ids: ['sub-1066']"
    "subject_ids: ['sub-1124']"
    "subject_ids: ['sub-1149']"  
    "subject_ids: ['sub-1175']"
    "subject_ids: ['sub-1209']"
    "subject_ids: ['sub-1232']"
    "subject_ids: ['sub-1263']"
    "subject_ids: ['sub-1006']"
)

base_model_command="CUDA_VISIBLE_DEVICES=1 python main.py --num_workers 40 --config cfgs/PointGPT-S/finetune_single_streamline_TrInF.yaml --test --exp_name error_trial --ckpts experiments/TractoInferno/finetune_single_streamline_TrInF/PointGPT-S/single_streamline_TrInF_500/pps256/ckpt-best.pth"
dice_score_command="python dice_score_scripts/dice_score_single_streamline_TrInF.py"
# Loop through each subject_ids block
for i in "${!subject_blocks[@]}"
do
    echo "Adding subject_ids block $((i + 1)) to the config file..."
    echo "${subject_blocks[$i]}" >> "$config_file"
    echo "Running iteration $((i + 1)) with subject_ids block:"
    echo "${subject_blocks[$i]}"

    conda activate TractoBERT
    eval $base_model_command
    conda deactivate
    conda activate scilpy_env
    eval $dice_score_command
    conda deactivate


    echo "Removing subject_ids block $((i + 1)) from the config file..."
    sed -i "/^subject_ids: \[.*\]/d" "$config_file"
    echo "Iteration $((i + 1)) completed."
done