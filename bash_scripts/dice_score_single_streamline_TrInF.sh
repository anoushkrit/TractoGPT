#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
# Path to your config file
config_file="cfgs/dataset_configs/single_streamline_TrInF_23.yaml"

# Define the subject_ids blocks (pairs of subject_ids lines to toggle between)
subject_blocks=(
    # "subject_ids: ['sub-1235', 'sub-1281', 'sub-1019', 'sub-1047']"
    "subject_ids: ['sub-1046', 'sub-1050', 'sub-1080', 'sub-1135']"
    "subject_ids: ['sub-1156', 'sub-1178', 'sub-1217', 'sub-1061']"
    "subject_ids: ['sub-1115', 'sub-1146', 'sub-1164', 'sub-1191']"
    "subject_ids: ['sub-1222', 'sub-1245', 'sub-1024', 'sub-1049']"
    "subject_ids: ['sub-1066', 'sub-1124', 'sub-1149', 'sub-1175']"
    "subject_ids: ['sub-1209', 'sub-1232', 'sub-1263', 'sub-1006']"
)

base_model_command="CUDA_VISIBLE_DEVICES=1 python main.py --num_workers 40 --config cfgs/PointGPT-M/finetune_single_streamline_105HCP_23.yaml --test --exp_name error_trial --ckpts experiments/105HCP/finetune_single_streamline_105HCP_23/PointGPT-M/single_streamline_23_500/ckpt-best.pth"
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