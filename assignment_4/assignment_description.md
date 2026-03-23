# assignment 4: Dog-cat object detection

## Overall goal

In this assignment, you will build and train an object detection network in the spirit of [YOLOv1](https://arxiv.org/abs/1506.02640) for dog/cat head detection. Core learning objectives are (1) understanding the architecture of a one-stage object detector and being able to make informed changes, (2) setting up an object detection experiment, and (3) reporting the performance of an object detection experiment and reflecting on the results. 

## Software and installation

We will program in Python. From Assignment 3, you should be familiar with PyTorch's functionality to build, train and evaluate CNNs. We will build on this expertise to build an object detector.

## Data

The data for this assignment comes from the dog-cat-detection (kaggle) dataset. It contains 3686 color images of different sizes, split into a train+val and a test set. There are only two labels (cat and dog). Each image is annotated with (1) a class label, and (2) a bounding box around the head. We provide a python script to process and visualize the image and bounding box (you can build your Pytorch Dataset and Dataloader similarly).

## Running experiments

If your computer has a GPU with CUDA enabled, you can benefit from significant speed-up, especially when training your models. You can make use of computation services such as [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). Google Cloud will also give you some free allowance on sign-in. Microsoft offers free [Azure](https://azure.microsoft.com/en-us/developer/students/) access for students. There are also free cloud GPU services from Kaggle as well. **Important**: Don't forget to shut down your notebooks when you have finished running experiments.

## Tasks
Your tasks are:

1. **Load and prepare data:** Load the dog-cat-detection datasets and split the trainval set into a training and validation set. Use a stratified 80/20 split. Resize the images in such a way that they have a fixed 112x112x3 format. Explain in the report how you do this. Make sure the bounding boxes are transformed in a similar manner. Check [these functions](https://pytorch.org/vision/0.9/transforms.html). It helps to visualize for some random images whether the resize is good by drawing the transformed head bounding box over the resized image. This functionality is included in the provided script.

2. **Develop the architecture:** We will implement a small version of an object detector architecture. Your network contains the layers below. The output is similar to the YOLOv1 output, with a 7x7 spatial grid but with B=1 detections per cell, and only C=2 classes. You can decide the type of pooling, activation functions, and whether or not to use dropout and batch normalization. Pay attention to the activation function of the output layer. At this point, we're not using non-maximum suppression. Your network should have approximately 1M parameters.
```
Input layer (output: 112x112x3)

Convolution layer: 3x3, stride 1, zero padding 1, 16 kernels
Batchnormalization
Pooling layer: 2x2, stride 2, zero padding 0

Convolution layer: 3x3, stride 1, zero padding 1, 32 kernels
Batchnormalization
Pooling layer: 2x2, stride 2, zero padding 0

Convolution layer: 3x3, stride 1, zero padding 1, 64 kernels
Batchnormalization
Pooling layer: 2x2, stride 2, zero padding 0

Convolution layer: 3x3, stride 1, zero padding 1, 64 kernels
Batchnormalization
Pooling layer: 2x2, stride 2, zero padding 0

Convolution layer: 3x3, stride 1, zero padding 1, 32 kernels
Batchnormalization

Flatten layer

Dropout

Fully connected layer: 512 neurons

Output layer: 343 neurons. Use a sigmoid activation function for your output layer because we are not dealing (exclusively) with probabilities.
```
3. **Train your object detector model:** Use the dog-cat-detection train and val sets to train your network from scratch. In particular, use early stopping on the loss of your val set. Use the same five losses as in the original YOLOv1 paper, with the same weights. You are free to choose your hyperparameters, including batch size, optimizer and learning rate.
4. **Evaluate the performance of your trained model:** Report the mAP when changing the objectness threshold from 0 to 1. Present a confusion matrix for the threshold setting for which mAP is maximal.
5. **Report your results:** See the rubric below.

## Report
Reporting is important in this assignment, so pay attention to the motivation of your choices, and the discussion of your results. You can use infomcv_assignment_4_report_template.docx, but feel free to deviate from the suggested lengths. Your report should contain:
1. A description of the **architecture** and parameters (summary).
2. A description of the way you **processed your images** and bounding boxes.
3. Overview of your **training hyperparameters**.
4. **Graphs** for your loss components, over the training epochs.
5. The training, validation and test **mAP** of your model.
6. **Confusion matrix** for the best objectness threshold. Make sure you can identify misclassifications.
7. Link to your **models weights**.
8. A list of the **choice tasks** you implemented. For each task, look at what additional reporting we expect.
9. Any **genAI prompts** you used, and how the answers influenced your work.

## Submission
Through Brightspace, hand in the following two deliverables:
1. A zip of your code (**no binaries, no libraries, no data/model weight files, no images**). Your scripts should be .py files, or notebooks.
2. A report (**2-5 pages**), see above.

## Grading
The maximum score for this assignment is 100 (grade 10). The assignment counts for 15% of the total grade. You can get 70 regular points and max 30 points for chosen tasks. Fixed tasks:
1. Load and process your image and bounding boxes: 5 (explain in your report how you do this)
2. Implement your architecture: 10
3. Implement the loss functions: 15
4. Train your model: 15
5. Reporting of model architecture and hyperparameters: 10 (model summary, hyperparameters, and architectures motivated)
6. Reporting and discussion of results: 15 (result overview, confusion matrix, graphs, explain difference in performance between training, validation and test, discuss results)

Choice tasks also include a reporting aspect. When applicable, choose one of your top two architectures and report the performance both with and without the novel functionality.
1. CHOICE 1: Improve the architecture to provide better results: 10. Motivate your changes. More impactful changes will be awarded more points. A simple change in the number of kernels, for example, will not grant many points. Report the results in terms of mAP.
2. CHOICE 2: Replace the backbone with a pretrained ResNet: 10. Convolution layers before the first FC layer can be treated as the backbone. Replace all layers before the first FC layer with ResNet-18 (without FC layers), then initialize ResNet-18 with the checkpoint pretrained on ImageNet (checkout models.resnet18). Finally finetune the whole model on the cat-dog detection dataset. Refer to stackoverflow for using the backbone of ResNet. You need to change the input size from 112 to 224 to adapt to ResNet-18. Report the mAP and loss of this network.
3. CHOICE 3: Pretrain the backbone: 15. Instead of training the detection network from scratch, pretrain the backbone on the [cat-dog-classification](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset/data) dataset (you need to change the prediction head),  then initialize your original detection network using the pretrained checkpoint. Finally finetune your network for the detection task using the dog-cat-detection dataset. Report the mAP and loss of your pretrained YOLOv1.
4. CHOICE 4: Instead of having a fixed validation set, implement k-fold cross-validation: 5. Note that this significantly increases the running time. Use k=5. Discuss the differences between folds.
5. CHOICE 5: Evaluate your network on a video of a cat and a dog: 10. You are free in choosing the video, and no ground truth locations have to be annotated. Include a link to the result (video) of the object detection, with clear distinction in color between a cat detection and a dog detection. At least 50 frames need to be processed but you can skip frames to cover a longer period.
6. CHOICE 6: Perform data augmentation techniques (at least 3): 10. Only techniques that do not affect the bounding box: max 5. Report mAP and explain how data augmentation affects your performance. Select meaningful techniques.
7. CHOICE 7: Show examples of misdetections: 10. Provide and discuss samples of cat->dog, dog->cat misclassifications, false positives and false negatives.
8. CHOICE 8: Implement non-maximum suppression: 10. Implement it as a step after prediction, but not during training. Report the mAP and confusion matrix with the same objectness threshold with and without non-maximum suppression.
9. CHOICE 9: Use your creativity. Check with [Ronald](r.w.poppe@uu.nl) for eligibility.

## Contact
Should you have any questions about the assignment, post them in the Assignment channel of the INFOMCV 2026 Teams. If you need help with your code, do not post your code completely. Rather post the chunks to which the question should refer.
## Frequently Asked Questions

*Q: Are we allowed to use tutorials?*

A: You can use tutorials as a guideline but we expect you to be able to implement all tasks yourself eventually. Make sure you understand what happens. Do NOT copy existing repositories.

---

*Q: Are we allowed to use high level packages such as Keras, TIMM or HuggingFace)?*

A: No.

---

*Q: Am I going to be marked down for not having great results?*

A: No, this is not meant to be a benchmarking assignment, but you should try your best and be able to explain your results. For the main model, your map@50 should be around 55-60%. With the ResNet-18 backbone, you can get mAP@50 above 80%.

---

*Q: How large should my batch sizes be?*

A: Typically, larger is better but this is up to the hardware that is available to you. A batch size of 8 should work nicely.
