- Prerequisite: cuda 10.0, cudnn 7, python 3.6 installed
- Install dependencies: run "pip3 install -r requirement.txt" 
- Put the data under "images" and "./data" folder for training
- Download trained model from https://drive.google.com/open?id=1NaxOs3pI97PLOFRc3NAM6sZyMxRCSYXs and put to "./weights" folder
- For training from scratch, run: "python3 train.py", model will be saved to "./weights" folder with timestamp as subfolder. Make sure "checkpoint_dir" variable in config is set to None.
- For training from saved checkpoint, change "checkpoint_dir" variable in config file and run: "python3 train.py" to resume from provided checkpoint. 
Or run "python3 train.py --checkpoint [/path/to/your/saved/checkpoint]" to resume from your own saved checkpoint. Newly trained model will be saved at the same directory of provided checkpoint
- Explanations for choosing model/parameter can be found in Report file.
- To view metrics, run "tensorboard --logdir=[log directory] --port [port] --host [host]"
- For inference, simple run "python3 predict.py" to predict fasion mnist dataset. 
