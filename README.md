# SketchToImage
### A project for 18786: Introduction for Deep Learning @ CMU.

We created numerous pipelines, many illustrating our failed attempts and iterative progress.
Important ones are described below.

`pipelineV6.py`: Our final pipeline, using class conditioning as well as sketch conditioning.
`pipelineV5.py`: A baseline pipeline with only sketch conditioning.
`pipelineV4.py`: A pipeline using reconstruction loss, scrapped for taking too long in training.

Below are some of our model implementations, and dataloaders. They include the unet we used, 
as well as other components such as auto-encoders.

`simpleAE.py`: Our implementation for variational auto encoders and deterministic auto encoders.
`sketch_condition_unet.py`': Our augmentation of [minimal_diffusion](https://github.com/VSehwag/minimal-diffusion) to support sketch and class conditioning.
`datal.py`: Our dataloader class. Used on the sketchy dataset with the capability to provide class labeling.
