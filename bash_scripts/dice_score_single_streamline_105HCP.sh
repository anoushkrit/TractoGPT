#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
# Path to your config file
config_file="cfgs/dataset_configs/single_streamline_105HCP_23.yaml"

# Define the subject_ids blocks (pairs of subject_ids lines to toggle between)
subject_blocks=(
    "subject_ids: ['622236', '679568']"
    "subject_ids: ['673455', '715647']"
    "subject_ids: ['761957', '786569']"
    "subject_ids: ['802844', '889579']"
    "subject_ids: ['930449', '599469']"
)

base_model_command="CUDA_VISIBLE_DEVICES=0 python main.py --num_workers 40 --config cfgs/PointGPT-M/finetune_single_streamline_TrInF_23.yaml --test --exp_name error_trial --ckpts experiments/TractoInferno/finetune_single_streamline_TrInF_23/PointGPT-M/single_streamline_23_1000/pps256/ckpt-best.pth"
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