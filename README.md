# GraFix
GraFix is a Graph Neural Network based project started at AECTech Hackathon 2022. It is a collaborative effort by Omid, Yankun, Mostapha, Yuan-Tung, Mingbo, Sila, Taro, Alireza. The target is to fix 2d floor plan mis-alignment assuming the floor plan as a graph data structure, in which rooms are nodes and adjacencies are edges.
![image](https://github.com/simpleSketche/GraFix/blob/main/Results/images/predict%20data.gif)

## Probelm
The architecture 2d/3d models from architects are often "broken" and things are not precisely aligned. The time put in to manually fix the errors in the models usually taken about hours and sometimes even a week depending on the size of the model, and there is not a relatively good solution exsiting on market to optimize the fixing process. Therefore, we take on this challenge ourselves and use the power of graph neural network model to tackle this problem. If you are new to Graph or Graph neural network world, here are some good beginner friendly articles that make it easy to understand -> [Introduction to Graph Neural Network](https://distill.pub/2021/gnn-intro/) | [Understand Convolutions on Graph](https://distill.pub/2021/understanding-gnns/).
![image](https://user-images.githubusercontent.com/71196100/200374021-3603563d-1031-4b5b-8175-b3792dca0a80.png)

## Data Preparation
We used the modular building generative model Yankun worked on in the past to generate the synthetic floor plan dataset. Each of the floor plan data could contain the number of boxes from 4 to 8, and we destroy the good "floor plan" to a broken state to train the machine to learn:
1. What is bad floor plan?
2. Which corner of the box to move in order to fix the "floor plan"?
<br>
Graph Neural Network is able to learn the buttom logic of the fixing logic from small batch (2000 data) of small floor plan training samples, and we can apply the training result on relatively larger "floor plan" that contains more boxes. This is greatly advantageous compared to other neural network models.
![image](https://github.com/simpleSketche/GraFix/blob/main/Results/images/data%20generator.gif)

## Prediction
![image](https://user-images.githubusercontent.com/71196100/200372202-b45c4124-59f5-4d7d-b7a9-c71464247467.png)

## Team
Click the images to visit their github pages!
<br>
[![image](https://user-images.githubusercontent.com/71196100/200363326-11f51cab-0df9-449f-a33d-57a6cbe175ee.png)](https://github.com/simpleSketche)
[![image](https://user-images.githubusercontent.com/71196100/200363532-1a264ae7-0207-4ae9-a202-cc29fb18e8ca.png)](https://github.com/OmidSaj)
[![image](https://user-images.githubusercontent.com/71196100/200363710-a56c2c3f-c4ba-4f32-9677-870368cbebc5.png)](https://github.com/ngulgec)
[![image](https://user-images.githubusercontent.com/71196100/200364102-a190c599-6652-4fa3-93d3-2c52832a3021.png)](https://github.com/ton731)
[![image](https://user-images.githubusercontent.com/71196100/200364698-8e3ebd56-0e7d-452b-b5b6-55ba217fc334.png)](https://github.com/mostaphaRoudsari)
[![image](https://user-images.githubusercontent.com/71196100/200364997-2739efcb-7f11-4d38-a3a2-6b15dd6ba670.png)](https://github.com/MingboPeng)
[![image](https://user-images.githubusercontent.com/71196100/200366117-041be235-8ba4-4378-9ce5-8738bff5903b.png)](https://github.com/tnarah)
[![image](https://user-images.githubusercontent.com/71196100/200366292-2880c5af-66ee-49b3-be91-9a08f94b6b5b.png)](https://github.com/Memortal)

## Synthetic Data Generation
Please generate the synthetic graph dataset, please refer to the provided Grasshopper script using Rhino 7

## Dependencies
To run this project, you would need:
- python 3.7+
- pytorch 1.12.2
- pytorch.geometric 0.8.0
- networkX


