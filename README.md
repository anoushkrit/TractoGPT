# TractoGPT
White matter bundle segmentation is crucial for studying brain structural connectivity, neurosurgical planning, and neurological disorders. White Matter Segmentation remains challenging due to structural similarity in streamlines, subject variability, symmetry in 2 hemispheres, etc. 

To address these challenges, we propose **TractoGPT**, a GPT-based architecture trained on streamline, cluster, and fusion data representations separately. TractoGPT is a fully-automatic, registration-free method that generalizes across datasets and retains shape information of the white matter bundles.

Experiments also show that **TractoGPT**outperforms state-of-the-art methods on average DICE, Overlap and Overreach scores. We use TractoInferno and 105HCP datasets and validate generalization across datasets.

## Prepare Data

Before running this code, you need to create proper python environment.
For preparing data for a specific use case, run the provided scripts. The dataloaders will automatically handle data preparation.

### Examples:

1. **Pretraining**  
   Use the following script for pretraining. This will create data for the `train` subset only:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/TractoGPT-S/pretrain_TrInF.yaml --exp_name demo_pretrain --exp_subset TractoInferno
   ```

2. **Finetuning**
    Use the following script for finetuning. This will create data for train and val subset, but not for test subset:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/TractoGPT-S/finetune_single_streamline_TrInF.yaml --exp_name demo_finetune --exp_subset TractoInferno --ckpts experiments/TractoInferno/pretrain_TrInF/TractoGPT-S/single_streamline_TrInF/ckpt-last.pth
    ```

3. **Inference/Testing/Dice Score Calculation**
    Use the following script for inference. This will not save the processed data for test subset:

    ```bash
    sh bash_scripts/dice_score_universal.sh --exp_name demo_inference --ckpts experiments/TractoInferno/finetune_single_streamline_TrInF/TractoGPT-S/single_streamline_TrInF/ckpt-best.pth --config cfgs/TractoGPT-S/finetune_single_streamline_TrInF.yaml
    ```
    
    The dice score will be save in ./predictions folder.

Modifying the --config flag will allow you to use different configurations for the model. The configurations are stored in the `cfgs` folder.

## Train Model 

## Infer Model 

## Metrics 

