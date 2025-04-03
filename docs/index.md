<!-- <head>
    <style>
        .photos img{
          display: inline;
          vertical-align: top;
          float: none;
        }
    </style>
</head>
 -->
# Trash Sorting Algorithm - Computer Vision

*Björn Bucksch, Samuel Goldie, Per Skullerud*

*Technical University of Delft*

---

## ABSTRACT

This project investigates the fine-tuning of object detection models, in particular YOLO v8, for the task of identifying and classifying different types of trash in images. Due to the lack of a suitable labeled dataset, a custom dataset is created by editing individual trash objects - such as plastic bottles - and their respective bounding boxes into a background image. Data augmentation techniques, including rotation, scaling, and color manipulation, are applied to make the dataset more realistic and improve the model's robustness and generalizability. In order to test whether the model is able to adapt to real-life scenarios, the test set consists of real (un-edited) annotated images containing trash. Furthermore, two different types of backgrounds are tested to determine what is most effective: white and natural backgrounds. The fine-tuned model achieved a mean average precision (mAP) of **X%** on the test set, proving its effectiveness in detecting and categorizing different trash types. This work shows that by simply fine-tuning existing object detection models, these can perform accurately in complex tasks even if trained on non-real images.

<!-- CHANGE PRECISION -->

## INTRODUCTION

Being able to accurately sort trash and recycle is very important for sustainable living in today’s world. In the Netherlands, trash-sorting plays an especially important role. However, this can represent a challenge for newcomers who, due to their backgrounds, may not be familiar with these practices. Apart from disrupting recycling efforts and increasing costs, incorrect trash sorting can lead to unexpected fines for those who are unaware. This project aims to provide a step in the direction of making the recycling system more newcomer-friendly by developing a robust computer vision-based trash-sorting algorithm. Using an existing object detection model and a custom dataset built for this purpose, various types of trash are identified and classified. The system's robustness is then tested with real-life images to ensure its accuracy. This innovation not only hopes to help newcomers adapt to Dutch recycling norms, but also to improve overall waste management efficiency. 

## BACKGROUND





## METHODOLOGY

The most efficient method to make a model that classifies different types of waste is to use an existing classifier model, and train only the last (few) layers to adapt to the new objects that should be classified, while keeping the rest of the weights frozen. A robust initial model has to be used for this purpose so that it can generalize well.

In order to explore the extent to which the modified model generalizes

### Image background

### Model choice
<!-- Why YOLO v8 -->



## RESULTS

## ANALYSIS

## CONCLUSION & DISCUSSION

