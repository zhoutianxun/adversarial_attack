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
```

### Run experiments
Runs the following experiments
* Network without attack
* Network with fast OP attack (our method)
* Network with original OP attack

```
usage: main.py [-h] [-p PIXELS] [-n N_TEST]

optional arguments:
  -h, --help            show this help message and exit
  -p PIXELS, --pixels PIXELS
                        number of pixels for attack, type:int
  -n N_TEST, --n_test N_TEST
                        number of test images, type:int
```
Eg.
```
# 20 pixels attack
python main.py --pixels 20 -n_test 50
```
Results will be saved in .csv files while running, so there will be results saved even if program is terminated early
