# SketchToImage
### A project for 18786: Introduction for Deep Learning @ CMU.

Team members: Kaustabh Paul, Aidan Erickson, Denis Alpay

You can download the sketchy dataset from google drive using this link: https://drive.google.com/file/d/0B7ISyeE8QtDdTjE1MG9Gcy1kSkE/view?usp=sharing&resourcekey=0-r6nB4crmdU-LK7H38xnOUw

We created numerous pipelines, many illustrating our failed attempts and iterative progress.
Important ones are described below.

`pipelineV4.py`: A pipeline using reconstruction loss, scrapped for taking too long in training.

The results of this pipeline can be seen in folders imagesV4_28-04_23:58 (results after training for 20 epochs) and imagesV4_29-04_19:20 (results after training for 100 epochs) (Note that the factor behind Los term 2 changed as well)

`pipelineV5.py`: A baseline pipeline with only sketch conditioning.

The results can be seen in folder imagesV5_29-04_02:06 (trained for 1000 epochs)

`pipelineV6.py`: Our final pipeline, using class conditioning as well as sketch conditioning.

The result of this pipeline can be seen inside the folder imagesV6_30-04_03:32 (trained for 100 epochs)

Below are some of our model implementations, and dataloaders. They include the unet we used, 
as well as other components such as auto-encoders.


`simpleAE.py`: Our implementation for variational auto encoders and deterministic auto encoders.

`sketch_condition_unet.py`: Our augmentation of the unet class from unet.py taken from [minimal_diffusion](https://github.com/VSehwag/minimal-diffusion) to support sketch and class conditioning.

`datal.py`: Our dataloader class. Used on the sketchy dataset with the capability to provide class labeling.

`augmentor.py` Our augmentor script, used in tandem with the [PhotoSketch](https://github.com/mtli/PhotoSketch) code, to create realistic sketch-like edgemaps.

`photoketch_dataloder.py` A script for generating and loading edge maps efficiently.

To see performance of our models V4, V5 or V6 with random noise conditioned on sketches run the ipynb file: test_diffusion_for_ V4 V5 or V6
