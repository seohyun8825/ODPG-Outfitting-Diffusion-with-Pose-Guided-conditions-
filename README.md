This is a virtual try-on model that adapts to both pose and garment features. When provided with a source image, a pose map, and a garment image, the model outputs an image of a person with altered pose and changed clothing.

Here is the pipeline for our model:

#OUR MODEL PIPELINE
![pipeline_ODPG](https://github.com/seohyun8825/ODPG_1/assets/153355118/a4e2c20e-5a0c-4ab8-b9ea-c5de18c64d9e)

Below are the results after a training process of 10 hours (20 epochs) using a single A100 GPU:

#Qaulitative Results
![qualitative result](https://github.com/seohyun8825/ODPG_1/assets/153355118/da31df0c-4179-4a6e-a280-2500d0d003c9)




This code is modified based on CFLD official code (https://github.com/YanzuoLu/CFLD)
