#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
# Path to your config file
config_file="cfgs/dataset_configs/single_streamline_105HCP.yaml"

# Define the subject_ids blocks (pairs of subject_ids lines to toggle between)
subject_blocks=(
    "subject_ids: ['622236']"
    "subject_ids: ['679568']"
    "subject_ids: ['673455']"
    "subject_ids: ['715647']"
    "subject_ids: ['761957']"
    "subject_ids: ['786569']"
    "subject_ids: ['802844']"
    "subject_ids: ['889579']"
    "subject_ids: ['930449']"
    "subject_ids: ['599469']"
)

base_model_command="CUDA_VISIBLE_DEVICES=0 python main.py --num_workers 40 --config cfgs/PointGPT-S/finetune_single_streamline_105HCP.yaml --test --exp_name error_trial --ckpts experiments/105HCP/finetune_single_streamline/PointGPT-S/single_streamline_105HCP_500/pps256/ckpt-best.pth"
dice_score_command="python dice_score_scripts/dice_score_single_streamline_105HCP.py"
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