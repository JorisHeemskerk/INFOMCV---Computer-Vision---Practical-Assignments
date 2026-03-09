# Assignment 3: CNN-based image classification

## Instructions
### Overall goal
In this assignment, you will get hands-on experience in training and validating CNN models, pre-training and setting up and reporting an experimental study.

### Software and installation
We will use [Pytorch](https://pytorch.org/), which is straightforward to [install](https://pytorch.org/get-started/locally/) once you have Python installed. Looking at [this](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) tutorial will help you to get familiar with Pytorch' basics.

### Data
The data for this assignment come from the [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. These dataset are available as part of Pytorch, read [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) about datasets and dataset loaders. It contains 50k training and 10k test images. Each image in CIFAR-10 has one out of ten possible class labels. Images are in color (three channels) of size 32x32.

### Running experiments
If your computer has a GPU with CUDA enabled, you can benefit from significant speed-up, especially when training your models. You can make use of computation services such as [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). Google Cloud will also give you some free allowance on sign-in. Microsoft offers free [Azure](https://azure.microsoft.com/en-us/developer/students/) access for students. There are also free cloud GPU services from Kaggle as well. **Important:** Don't forget to shut down your notebooks when you have finished running experiments.

### Pytorch vs. OpenCV
We mainly use Pytorch but OpenCV might still be your go-to option to address some of the choice tasks, such as preprocessing the data. In this case, you can either integrate both packages, or create a separate program/script that does the job and saves the output (images). While the integration of OpenCV in Python is straightforward, you don't need to do this necessarily.

### Tasks
Your tasks are to develop the scripts to:

1. Load the CIFAR-10 dataset including the data labels. This would import two sets (training set and test set). Create a third set (validation set) by splitting the training set into two (training set and validation set). Decide what a good ratio of training/validation is, and motivate your choice. You should use the validation set to evaluate the different choices you make when building your CNNs. Keep in mind that the test set will only be used for final benchmarking and will not be included in the validation step.
2. Recreate the LeNet-5 architecture (but then for color images) as your **baseline**. The model takes as input a color image of size 32x32x3 and has 10 outputs, one for each class. Make sure all parameters (number of neurons, number and size of kernels) are as similar as the original architecture. You may assume that no zero-padding was applied. You may also assume that averagePool or maxPool was used (choose one). The activation function can be set to ReLU everywhere, and Softmax for the output layer. The model is trained using cross-entropy loss, and Adam optimizer with a learning rate of 0.001. Use `torch.nn.init.kaiming_uniform` to initialize your weights. Use a batch size of 32, unless your hardware doesn't allow you to. Then reduce the size accordingly. Now create **two model variants**. Each model may differ from the previous model by **only one aspect**, such that we can compare each pair of subsequent models pair-wise. An aspect should be a meaningful property, e.g., change the type of one layer (convolution --> pooling, etc.), add one layer, use dropout, change your activation function, change the number or size of your kernels. No change of hyperparameters. No use of any merging, attention, recurrent or locally-connected layers. **Your variants should be aimed at getting a better performance.**
3. Train your three models on the **training set** and validate them on the **validation set** of CIFAR-10. For each model and each epoch, store your training and validation loss and accuracy. You now have three trained models: CIFAR10_lenet, CIFAR10_model1, CIFAR10_model2. Report their performance in a table and explain differences in terms of the architectural choices.
4. From the three trained networks, choose the best architecture based on the validation set. Then load the CIFAR-100 dataset in a similar way as before. Again, split the training set in a training and validation set. Change your CIFAR-10 architecture of the best model to be able to deal with 20 class outputs. Then train the model from scratch until convergence. Keep the hyperparameters the same as before. You now have additional model CIFAR100_model.
5. Use your CIFAR100_model and change the last layer back to 10 outputs. Keep all other weights in the model the same as the CIFAR100_model. Now fine-tune this model on CIFAR-10 with the learning rate half of your initial learning rate. This model is CIFAR10_pretrained.
6. Compare the best CIFAR-10 model and your CIFAR10_pretrained model on your test set. Report accuracy and confusion matrices. Compare the two models. Explain differences between the respective training, validation and test sets.
7. Report your findings (see under submission).

### Report
Reporting is important in this assignment, so pay attention to the motivation of your choices, and the discussion of your results. You can use infomcv_assignment_3_report_template.docx, but feel free to deviate from the suggested lengths. Your report should contain:

1. For the three CIFAR-10 models in task 2, give (1) a **description** of the architecture and parameters (`summary`), (2) a **graph** with the training/validation loss on the y-axis and epochs on the x-axis, and (3) a **link** to your model's weights (publicly accessible).
2. For the three model variants, add a description of **which property differs** from the previous model, and motivate your choice (so why do you expect a better performance). 
3. A **table** with the train/validation top-1 accuracy for all models.
4. A discussion of your results in terms of your models (e.g., complexity, type of layers, etc.). Make **pair-wise comparisons** between the two variants and the baseline model (so CIFAR10_lenet vs CIFAR10_model1 and CIFAR10_model1 vs CIFAR10_model2).
5. A **discussion** of the differences between the results of your best model trained from scratch and pre-trained on CIFAR-100, evaluated on test set. How can differences be explained? Add a confusion matrix for both tested models. Explain differences of each model in terms of performance on training, validation and test sets.
6. A list of the **choice tasks** you implemented. For each task, look at what additional reporting we expect.
7. An overview of all **genAI prompts** and how you used the answers in your work.

### Submission
Through Blackboard, hand in the following two deliverables:

1. A zip of your code (**no binaries, no libraries, no data/model weight files, no images**). Your scripts should be .py files or notebooks.
2. A report (**2-5 pages**), see above.

### Grading

The maximum score for this assignment is 100 (grade 10). The assignment counts for 15% of the total grade. You can get 60 regular points and max 40 points for chosen tasks. Fixed tasks:

1. Create the baseline and two CNNs: 10 (suitable choice of parameter)
2. Train and validate three models on CIFAR-10: 10 (correct use of sets)
3. Training on CIFAR-100: 5
4. Fine-tuning CIFAR-10 model on CIFAR-100: 5
5. Test best CIFAR-10 models (from scratch and pre-trained) on CIFAR-10 test set: 5
6. Reporting: motivation: 5 (model summary, and architectures motivated)
7. Reporting: presentation of results: 15 (performance, loss graphs, confusion matrix presented)
8. Reporting: discussion of results: 5 (explain which choices were most beneficial, explain role of training, explain difference in performance between training, validation and test sets)

Choice tasks also include a reporting aspect. When applicable, choose one of your top two architectures and report the performance both with and without the novel functionality.

1. CHOICE 1: Create and apply a function to decrease the learning rate at a 1/2 of the value every 5 epochs: 5. Add a graph of the learning rate over time. Compare against baseline without learning rate schedule.
2. CHOICE 2: Instead of having a fixed validation set, implement k-fold cross-validation: 10. Note that this significantly increases the running time. Use k=5. Compare results against fixed train/val split.
3. CHOICE 3: Do a hyperparameter search for optimizer (evaluate 3 options), learning rate (evaluate 3 options), weight decay (evaluate 2 options). If you do a grid search: max 5 points. If you use evolutionary search, max 10 points. In this case, you also need to evaluate two different batch sizes. Report accuracy for each tested combination.
4. CHOICE 4: Create output layers at different parts of the network for additional feedback. Show and explain some outputs of a fully trained network: 15. The output layers should connect to the convolution or pooling layers.
5. CHOICE 5: Perform data augmentation techniques (at least 3): 5. Report and explain how they affect your performance by comparing to a baseline without data augmentation. Only select meaningful techniques.
6. CHOICE 6: Provide t-SNE visualization of the fully connected layer before your output layer: 10. Show a graph with the embeddings of the test set. Discuss the graph in terms of the observed and expected confusions.
7. CHOICE 7: Evaluate cross-dataset performance on the Tiny ImageNet dataset (download from [Huggingface](https://huggingface.co/datasets/zh-plus/tiny-imagenet) or [Kaggle](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)). Pre-process the images so they fit your network, choose the classes that overlap with CIFAR-10 and report the test performance on one of your models with accuracy and confusion matrix. Discuss the differences with CIFAR-10: 15. 
8. CHOICE 8: Fine-tune your best CIFAR-10 model on the Tiny ImageNet overlapping classes. Report accuracy and confusion matrices. Compare model trained from scratch (this is CHOICE 7) with this fine-tuned model: 5 points.
9. CHOICE 9: Use your creativity. Check with [Ronald](r.w.poppe@uu.nl) for eligibility.

### Contact
Should you have any questions about the assignment, post them in the Assignment channel of the INFOMCV 2026 Teams. If you need help with your code, do not post your code completely. Rather post the chunks to which the question should refer.

### Frequently Asked Questions
*Q: Are we allowed to use tutorials?*\
A: You can use tutorials as a guideline but we expect you to be able to implement all tasks yourself eventually. Make sure you understand what happens. Do NOT copy existing Github repositories and report all genAI prompts.

*Q: Are we allowed to use high level packages such as Keras, TIMM or HuggingFace?*\
A: No, only for importing the data.

*Q: Am I going to be marked down for not having great results?*\
A: No. We mostly want to see that you understand the basic principles of how CNNs work and how you implemented cross-validation and testing. This is not meant to be a benchmarking assignment, but you should try your best to get decent results (around 55-60%). You will need to motivate which changes from LeNet have the most (positive) impact.

*Q: How large should my batch sizes be?*\
A: Standard batch sizes in literature have been: 32, 64, 128, 256 – larger than these are mostly for significantly more complex networks. Keep in mind large batch size => large learning rates, small batch size => small learning rates. Keep the batch size fixed across all experiments. Ideally, you use 32.