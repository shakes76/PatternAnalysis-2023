## Project Title

Brief description of the project, its objectives, and its significance.




### Table of Contents
- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Todo](#todo)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Usage](#usage)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

### Todo
- [x] Create my own training script that works on top of yolov7/train.py
- [x] Make a downsampled version of the dataset.
- [x] Manage a way of logging the runs to wandb
- [x] Zip and upload dataset to gdrive, and make functions for fetching the dataset from the cluster.
- [x] Make training scripts for use on the rangpur cluster.
- [ ] Manage to train on the downsampled dataset using the cluster. Need gpu version of CUDA for this to work.


### Installation

- Prerequisites: List any pre-required libraries or software. 
- Installation steps: 
  ```bash
  pip install -r requirements.txt
  ```

### Dataset

- Description of the dataset.
- Source of the dataset (if it's publicly available).
- Preprocessing steps (if any).

### Usage

- How to use the code/model.
- Example commands:
  ```bash
  python your_script_name.py --arg1 value1 --arg2 value2
  ```

### Model Architecture
The model and architecture used in this project is the open source yolov7 model:

https://github.com/WongKinYiu/yolov7


### Training

- Details about the training process.
- Hyperparameters used.
- Training time, hardware details (if relevant).

### Results

- Summarize the results obtained.
- Any metrics used (e.g., accuracy, F1-score).
- Visual results (e.g., plots, graphs) if applicable.

### Contributing

- Details on how others can contribute to this project.
- Any specific guidelines for contributing.

### License

- Licensing information. For example:
  ```
  This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
  ```

### Acknowledgements

- Any credits or acknowledgments for third-party resources or researchers.

---

Remember, the above is just a basic skeleton. Depending on the project's complexity and breadth, you might want to add more sections or elaborate on the existing ones.