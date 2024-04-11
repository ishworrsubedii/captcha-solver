# Captcha Solver

This code base is for captcha solver using tensorflow and keras on custom architecture of cnn based approaches.

## Setup

It is always best practice to have the anaconda env or we can also use venv for the same.

First create the conda env using the following command.

```angular2html
conda create -n captcha_solver python==3.10
```

Activate the conda env using the following command.

```angular2html
conda activate captcha_solver
```

Install the required packages using the following command.

```angular2html
pip install -r requirements.txt
```

## Training

For training the model, we need to do the following steps.

1. First put you captcha images in the `images/dataset/images` folder.(there should be images with the captcha label as
   the image name)
2. After that we need to split the images into train and test set by running the `examples/image_split.py`
   [image_split.py](examples/image_split.py).

3. After doing that we need to run the `examples/model_train.py` file to train the model.
   [model_train.py](examples/model_train.py)
   We can change the epoch and other parameters from the [config.py](src/entity/configuration.py) file.

## Inference

For inference, we need to run the `examples/inference.py` file [inference.py](examples/inference.py).

```python
if __name__ == "__main__":
    draw = True
    model_information_config = "resources/models/202403281555/configs.yaml"  # Add your trained model config file
    output_result_file = "output/result.txt"
    image_test_dir = "resources/dataset/dataset_20240328/test"  # Add your test image directory

    main(draw=draw, model_information_config=model_information_config, output_result_file=output_result_file,
         image_test_dir=image_test_dir)
```

Here update the `model_information_config` with the trained model config file and `image_test_dir` with the test image

## Accuracy

To Find the accuracy we can run `examples/accuracy_cal.py` [accuracy_cal.py](examples/accuracy_cal.py).

There you ned to change the config file and image information that need to be in .csv format

