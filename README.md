# GeometryAwareField2FieldTransformations

The repository contains code for 3D segmentation via Geometry Aware Field2Field Transformations. It is build upon the Nerfstudio Framework and utilizes PointNet++ and the Stratified Point Transformer for PointCloud Segmentation.

## Requirements
It is required to have the dependencies of [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and [Stratified Point Transformer](https://github.com/dvlab-research/Stratified-Transformer) installed. 

We had complications to install these dependencies ourself and provide forks of the repositories. Yet installation might prove complicated with your system. We provide our conda environment for easier installation, as this was a working configuration. However, the Stratified Point Transformer requires some manual installation. So first install it's dependencies.

In case of manual installation we recommend to install the depencies of Nerfstudio first. Then continue by installing the not yet installed dependencies of PointNet and the Stratified Point transformer.

### Conda environment

We provide our conda environment with a environment.yml. You can install it with:
```
conda env create --name nerfstudio -f environment.yml
```

### Nerfstudio

The instructions for installing Nerfstudio are listed here: https://github.com/nerfstudio-project/nerfstudio

### Pointnet

We recommend to install pointnet as pip module directly as to not need any adaptations of our code. Install all requirements of PointNet and then install the packages as a pip module as listed here: https://github.com/DominikVincent/PointNet

### Stratified Point Transformer

This dependencies of the Stratified Point Transformer were incompatible with our pytorch and nerfstudio version. Therefore, we modifed the Stratified Point Transformer repo. If the installation from our conda succeeded it is still required to compile pointops. We modified the cuda code to work with our pytorch version in [here](https://github.com/DominikVincent/Stratified-Transformer).

If the conda install didnt work. First install the requirements as stated in our [fork](https://github.com/DominikVincent/Stratified-Transformer), despite the install instructions we struggle to find a working combination of dependencies. Try checking our conda environment for a working combinations of requirements. We modified the lib/pointops cuda code to work with our pytorch version as well. Install it as stated in the [fork](https://github.com/DominikVincent/Stratified-Transformer).

After all the dependencies are installed install the repo as a local pip module as stated in the fork.


## Run experiments
First it is necessary to prepare the data. Secondly, one can train the NeRFs. Finally, we can create a train config and train the Field2Field Transformation. 

### Data
We require the data to be in the nerfstudio format. If you want to use the NeSF data download it from [here](https://console.cloud.google.com/storage/browser/kubric-public/data/NeSFDatasets) and extract it. You can convert it's metadata information via the jupyter notebook provided in playground/metadata_transforms.ipynb. Adapt the paths in the notebook to you folder of the dataset.

### Nerf Training
Now that the data is extracted you can train the collection of NeRFs with train_scripts/train_all.bash. The train_scripts/train.bash determines the config used for the NeRF. Adapt the data and output paths within the train_all.bash script to the ones on you system. Each NeRF trains for roughly 10-15 minutes.

### Transformation Model Training
You can create the final train configuration for the transformation model with the notebook playground/create_nesf_data_config.ipynb for it. Again adapt the paths in here. Ensure that the (NeRF) model config in there matches with the one used in train_scripts/train.bash. 

Now that you have obtained a data config for the Transformation Model Training. You can train it using the provided scripts in train_scritps:
- Stratified Point Transformer: train_scripts/train_nesf_stratified_toybox.bash
- Custom transformer: train_scripts/train_nesf_big_toybox.bash
- Pointnet: train_scripts/train_nesf_pointnet.bash

Our code relies on logging data to weights and biases. We also use weights and biases to finetune pretrained models. You can start pretraining with any of the train_scripts/train_nesf_pretrain_[...] scripts. You can fine-tune a run called pretrained_run1 on W&B with: 
```
./train_scripts/train_pretrained.py --data /path/to/transformation/model/data.config runs pretrained_run1
```

In case you use slurm you try to directly dispatch the runs to slurm via:
```
./train_scripts/train_pretrained.py --slurm --data /path/to/transformation/model/data.config runs pretrained_run1 pretrained_run2 ...
```

To evaluate a run first create the eval data config via playground/create_nesf_data_config.ipynb. Then one can evaluate the run from weights and biases run_name_1:
```
./train_scripts/eval_sweep.py --eval_config /path/to/eval/data.config --proj_name "toybox-5-nesf" run run_name_1
```
It should directly create a new run on W&B containing the results. 

Otherwise, you could manually use the ns-eval command line tool to evaluate runs. It might be needed to either specify the model configuration manually or adapt the model config which is supposed to be loaded.
