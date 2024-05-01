# SketchToImage
### A project for 18786: Introduction for Deep Learning @ CMU.

You can download the sketchy dataset from google drive using this link: https://drive.google.com/file/d/0B7ISyeE8QtDdTjE1MG9Gcy1kSkE/view?usp=sharing&resourcekey=0-r6nB4crmdU-LK7H38xnOUw

We created numerous pipelines, many illustrating our failed attempts and iterative progress.
Important ones are described below.


`pipelineV6.py`: Our final pipeline, using class conditioning as well as sketch conditioning.
`pipelineV5.py`: A baseline pipeline with only sketch conditioning.
`pipelineV4.py`: A pipeline using reconstruction loss, scrapped for taking too long in training.


Below are some of our model implementations, and dataloaders. They include the unet we used, 
as well as other components such as auto-encoders.


`simpleAE.py`: Our implementation for variational auto encoders and deterministic auto encoders.
`sketch_condition_unet.py`: Our augmentation of [minimal_diffusion](https://github.com/VSehwag/minimal-diffusion) to support sketch and class conditioning.

`datal.py`: Our dataloader class. Used on the sketchy dataset with the capability to provide class labeling.

`augmentor.py` Our augmentor script, used in tandeom with the [PhotoSketch](https://github.com/mtli/PhotoSketch) code, to create realistic sketch-like edgemaps.

`photoketch_dataloder.py` A script for generating and loading edge maps efficiently.
