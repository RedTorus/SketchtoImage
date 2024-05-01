dataDir="/home/aidan/ComputerScience/school/18786-DeepLearning/project/photosketch/PhotoSketch-master/data"

python test_pretrained.py \
    --name pretrained \
    --dataset_mode test_dir \
    --dataroot examples/ \
    --results_dir ${sketchHomeDir} \
    --checkpoints_dir ${dataDir} \
    --model pix2pix \
    --which_direction AtoB \
    --norm batch \
    --input_nc 3 \
    --output_nc 1 \
    --which_model_netG resnet_9blocks \
    --no_dropout \
