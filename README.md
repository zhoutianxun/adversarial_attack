# adversarial_attack

### Requirements
```
conda env create -f environment.yml
conda activate adversarial
```

### Directory setup
```
├── main.py # main script 
├── OnePixelAttack.py
├── OP_attack_fast.py 
├── imagenet # directory to store image files for testing
      ├── test-images
      ├── imagenet_class_index.json
      ├── imagenet_mask.pkl
```

### Run experiments
Runs the following experiments
* Network without attack
* Network with fast OP attack (our method)
  * option to initialize optimization with image structured attack mask if available
* Network with original OP attack

```
usage: main.py [-h] [-p PIXELS] [-n N_TEST] [-m MASK]

optional arguments:
  -h, --help            show this help message and exit
  -p PIXELS, --pixels PIXELS
                        number of pixels for attack, default: 20, type:int
  -n N_TEST, --n_test N_TEST
                        number of test images, default: 100, type:int
  -m MASK, --mask MASK  initialize attack with points sampled from image mask, default: False
```
Eg.
```
# 20 pixels attack without initialization from structured attack mask
python main.py --pixels 20 --n_test 50

# 20 pixels attack, with initialization from structured attack mask
python main.py --pixels 20 --n_test 50 --mask True
```
Results will be saved in .csv files while running, so there will be results saved even if program is terminated early
