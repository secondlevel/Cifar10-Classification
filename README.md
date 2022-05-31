# Cifar10-Classification(Pattern Recognition Homework5)

This assignment is to train a model to classification the images of cifar10. All the models in this project were built by pytorch.

In addition, please refer to the following report link for detailed report and description of the experimental results.
https://github.com/secondlevel/Cifar10-Classification/

![image](https://user-images.githubusercontent.com/44439517/170820943-9cb1c3ae-74d1-429b-a631-60637d67d013.png)

## Hardware
```bash=
Operating System: Ubuntu 20.04.3 LTS  

CPU: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  

GPU: NVIDIA GeForce GTX TITAN X  
```

## Requirement

In this part, I use anaconda and pip to build the execution environment.

In addition, the following two **options** can be used to build an execution environment
  
- ### First Option
```bash=
conda env create -f environment.yml
```

- ### Second Option 
```bash=
conda create --name cifar python=3.8
conda activate cifar
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install matplotlib pandas scikit-learn -y
pip install tqdm
```


## Directory Tree

In this homework, you can put the folder on the specified path according to the pattern in the following directory tree for training and testing.

The model weight can be download in the following link, please put it under the checkpoint directory.  
The data can be download in the following link, please put it in the under the repository according to the following description.
https://drive.google.com/drive/folders/1Boe0EZT1cyV6MxqqTFk1mufsGbThs4BG?usp=sharing

```bash=
├─ 310551031_HW5.py
├─ environment.yml
├─ history_csv
│  └─ BEST_VIT_CIFAR.csv
├─ checkpoint
│  └─ BEST_VIT_CIFAR.rar
├─ x_train.npy
├─ x_test.npy
├─ y_train.npy
├─ y_test.npy
└─ README.md
```

## Flow Chart

<p float="left">
  <img src="https://user-images.githubusercontent.com/44439517/170855893-8102bee2-3cc1-4159-bc91-70724ed00a37.png" title="normalization" width="70%" height="70%" hspace="200"/>
</p>


## Hyperparameter Setting
```python=
image_size = 224
number_worker = 4
batch_size = 64
epochs = 10
lr = 2e-5
optimizer = AdamW
loss function = CrossEntropy
```

## Data preprocess
The Data Preprocess include two parts. The first part is the standardization of pixel value([0, 255] to [0, 1]). The second part is to adjust the image to 224 x 224.

#### 1. Pixel Value Normalization
<p float="left">
  <img src="https://user-images.githubusercontent.com/44439517/170822922-be60ba86-3468-45ce-912a-6dea30300e3c.png" title="normalization" width="80%" height="80%"/>
</p>

#### 2. Resize Image to 224x224
<p float="left">
  <img src="https://user-images.githubusercontent.com/44439517/170822937-4018b58f-5fdb-4369-8452-64b325c56e73.png" title="resize image" width="50%" height="50%"/>
</p>

## Data Loader
In order to avoid the problem of the cuda out of memory, I create the data loader to process the data.

- Input: Image Array, Label Array, Data Augmentation method.  
- Ouput: DataLoader

```python=
class CIFARLoader(data.Dataset):
    def __init__(self, image, label, transform=None):

        self.img_name, self.labels = image, label
        self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        self.img = self.img_name[index]
        self.label = self.labels[index]

        if self.transform:
            self.img = self.transform(self.img)

        return self.img, self.label
```

## Model Architecture
In this homework, I used the [**Vision Transformer**](https://arxiv.org/pdf/2010.11929.pdf) pretrained model to classify images. 

In addition, I added the linear layer to the Vision Transformer (VIT) [1], all the weight of the VIT is **unfreeze**.

The Architecture of the classification model is as follows.

```python=
class VIT(nn.Module):
    def __init__(self, pretrained=True):
        super(VIT, self).__init__()
        self.model = models.vit_b_32(pretrained=pretrained)
        self.classify = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.classify(x)
        return x

model = VIT()
for name, child in model.named_children():
    for param in child.parameters():
        param.requires_grad = True
```


## Training
You can switch to the training mode with the following instruction, and then you can start training the classification model.
```bash=
python 310551031_HW5.py --mode train
```

The best model weight during training will be stored at **checkpoint directory**, and the training history will in the **history_csv directory**.

The training accuracy history is as following.
<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/170851167-f2885d45-c32d-4ceb-be79-117699e38442.png" title="accuracy curve" width="80%" height="80%"/>
</p>

The training Loss history is as following.
<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/170851186-f468e5d4-70be-474d-8df2-73b06cfdcf4f.png" title="loss curve" width="80%" height="80%"/>
</p>


## Testing
You can switch to the testing mode with the following instruction, and then you can evaluate the classification result.  
**Best Model Weight name: BEST_VIT_CIFAR.rar** (Which is in the checkpoint directory) 
```bash=
python 310551031_HW5.py --mode test
```

<p float="left">
  <img src="https://user-images.githubusercontent.com/44439517/170824794-c6c9a71e-96b5-4739-9d37-c1b707b07364.png" title="testing result" width="80%" height="80%"/>
</p>

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/170837774-216a6784-cb07-4bed-9f1d-9aa6f0b50add.png" title="confusion matrix" width="40%" height="40%" hspace="250"/>
</p>


## Reference
[1] A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” arXiv, arXiv:2010.11929, Jun. 2021. doi: 10.48550/arXiv.2010.11929.
