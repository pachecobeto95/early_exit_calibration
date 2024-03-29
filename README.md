# On the impact of deep neural network calibration on adaptive edge offloading

This article provides an extensive calibration study on different datasets and early-exit deep neural network (DNN) models. Moreover, this article evaluates the calibration impact on early-exit DNN models considering an adaptive offloading scenario. The next section described the concepts of early-exit DNNs, DNN partitioning, and adaptive offloading scenario.


## Adaptive Offloading Via Early-exit DNNs

DNNs can be described as a sequence of neural layers that can extract features from inputs. In general, shallow DNNs (i.e., DNNs with few neural layers)
can extract simple features, while deeper DNN models can extract more complex features and obtain more accurate predictions. Early-exit DNNs classify
some inputs based on feature representation obtained by shallow neural layers, while other inputs rely on features provided by deeper layers to be classified. The intuition behind this approach is that distinct samples may not require
features of equal complexity to be classified. Therefore, early-exit DNNs leverage the fact that not all inputs are equally difficult
to classify. 

Figure 1 illustrates an early-exit DNN with multiple side branches inserted into its intermediary layers. The vertices v1, · · · , vN represent the DNN backbone’s layers. The vertices b1, · · · , bk are the side branches, each of which contains a fully-connected layer capable of classifying inputs based on the features
extracted in the previous layers v1, · · · , vi. In the experimental results, we insert five side branches. Thus, we have six exits, including the DNN backbone’s output layer. 

![Figure 1 - Illustration of adaptive model partitioning between the edge device and cloud via
early-exit DNNs.](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/motivation/adaptive_offloading_via_ee_comnet2.png)

As illustrated in Figure 1, the early-exit DNN is split into two parts: the first part being implemented on the edge device and the second remotely at the
cloud server. Side branches are included only in the first part. At inference time, given an input **x**, the device estimates the prediction confidence for the
i-th side branch as the probability of the most likely class. If the confidence value is greater or equal to a predefined target threshold ptar,
the device concludes that the i-th side branch can classify the input and inference terminates by classifying the input as the class with the largest probability. Otherwise, the input is processed by the subsequent layers until it reaches the next side branch, following the same procedure described above. If no side branches reach the desired accuracy level ptar, the edge device offloads data to the cloud, which processes the remaining DNN backbone’s layers until the output layer. The cloud server then sends the prediction and its confidence level back to the edge device. If the confidence level does not reach
the target ptar, the classification uses the most confident predicted class among all exits.

## Objective

## Application-Level Metrics
In order to evaluate the performance at the application level, we consider the average accuracy of the decisions made at the edge and at the cloud as well as two novel metrics.  The first metric, referred to as edge inference outage probability, measures the model’s ability to guarantee an accuracy requirement via inference at the edge. The second metric, referred to as missed deadline probability, accounts for the probability in meeting both accuracy and inference latency requirements when allowing for inference at the edge or at the cloud.

### Application-Level Metrics

### Average Accuracy Metrics
We define the average edge accuracy Accedge(θ) as the fraction of test examples classified at any of the edge side branches that are correctly classified. The average total accuracy Acctotal(θ) is similarly defined as the fraction of test
examples that are correctly classified, irrespective of whether detection is done
at the edge or at the cloud.

### Edge Inference Outage Probability
This proposed metric measures the model's abality to achieve a predefined threshold ptar. Therefore, a more suitable measure of performance in
terms of accuracy is the fraction of inputs that are classified with an accuracy level larger than ptar. To this end, we divide test inputs into batches and evaluate
the average accuracy Accedge(Bb|θ) for each batch Bb. Note that only inputs classified at the edge are included in the average. The edge inference outage
probability is then defined as the fraction of batches that the event edge inference outage occurs.

### Missed Deadline Probability
A missed deadline is defined as the event that occurs when either the average inference time is larger than an application-defined latency deadline ttar or when
the average total accuracy is smaller than an accuracy requirement ptar. The missed deadline probability is thus a measure of the fraction of inputs that are reliably classified within an acceptable maximal latency. Unlike the edge inference outage, which is edge-focused, the missed deadline probability is an
end-to-end application performance metric. To evaluate the missed deadline probability, we again divide the test inputs
x into image batches B1, · · · , BNB . For each batch, we measure the average
inference time and the average total accuracy Acctotal(Bb|θ). The missed deadline probability is defined as the fraction of batches that the event missed deadline occurs.

## Numerical Results on Calibration
This section presents numerical results with the aim of quantifying the calibration performance in terms of reliability diagrams, ECE (Expected Calibration Error),
and probability of offloading with and without calibration. The next section will focus on application-level performance metrics.

### Reliability Diagrams
This subsection evaluates the benefits of TS in calibrating early-exit DNNs based on reliability diagrams and the ECE metric. For conciseness, we only
present results for Caltech-256 since Cifar-100.

The following figures present the reliability diagram for MobileNetV2 on Caltech256.

Conventional             |  Global TS          |  Per-branch TS 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_2_conventional_alt-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_2_overall_alt-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_2_early_alt-1.png) |
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_3_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_3_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_3_early_alt-1.png)
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_4_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_4_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_4_early_alt-1.png)
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_5_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_5_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/reliability_diagram/reliability_diagram_branch_5_early_alt-1.png)


The following figures present the reliability diagram for ResNet18 on Caltech256.

Conventional             |  Global TS          |  Per-branch TS 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_2_conventional_alt-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_2_overall_alt-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_2_early_alt-1.png) |
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_3_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_3_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_3_early_alt-1.png)
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_4_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_4_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_4_early_alt-1.png)
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_5_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_5_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/reliability_diagram/reliability_diagram_branch_5_early_alt-1.png)

The following figures present the reliability diagram for VGG16 on Caltech256.

Conventional             |  Global TS          |  Per-branch TS 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_2_conventional_alt-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_2_overall_alt-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_2_early_alt-1.png) |
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_3_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_3_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_3_early_alt-1.png)
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_4_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_4_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_4_early_alt-1.png)
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_5_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_5_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/reliability_diagram/reliability_diagram_branch_5_early_alt-1.png)



The following figures present the reliability diagram for ResNet152 on Caltech256.

Conventional             |  Global TS          |  Per-branch TS 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_2_conventional_alt-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_2_overall_alt-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_2_early_alt-1.png) |
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_3_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_3_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_3_early_alt-1.png)
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_4_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_4_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_4_early_alt-1.png)
| ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_5_conventional_alt-1.png)   |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_5_overall_alt-1.png)|  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/reliability_diagram/reliability_diagram_branch_5_early_alt-1.png)




### Offloading Probability
We evaluate the impact of calibrating an early-exit DNN on the offloading probability. The next figures show the probability of classifying at the
edge considering a given number k of side branches at the edge device.

The following figures show the edge probability considering a early-exit MobileNetV2 evaluated on Caltech-256 dataset.
           k=2             |   k=3                     |      k=4                  |         k=5 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![k=2](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/edge_probability/edge_probability_2_branches-1.png)  |  ![k=3](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/edge_probability/edge_probability_3_branches-1.png) |  ![k=4](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/edge_probability/edge_probability_4_branches-1.png)|  ![k=5](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/edge_probability/edge_probability_5_branches-1.png)

The following figures show the edge probability considering a early-exit ResNet18 evaluated on Caltech-256 dataset.
           k=2             |   k=3                     |      k=4                  |         k=5 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![k=2](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/edge_probability/edge_probability_2_branches-1.png)  |  ![k=3](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/edge_probability/edge_probability_3_branches-1.png) |  ![k=4](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/edge_probability/edge_probability_4_branches-1.png)|  ![k=5](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/edge_probability/edge_probability_5_branches-1.png)

The following figures show the edge probability considering a early-exit VGG16 evaluated on Caltech-256 dataset.
           k=2             |   k=3                     |      k=4                  |         k=5 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![k=2](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/edge_probability/edge_probability_2_branches-1.png)  |  ![k=3](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/edge_probability/edge_probability_3_branches-1.png) |  ![k=4](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/edge_probability/edge_probability_4_branches-1.png)|  ![k=5](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/edge_probability/edge_probability_5_branches-1.png)

The following figures show the edge probability considering a early-exit ResNet152 evaluated on Caltech-256 dataset.
           k=2             |   k=3                     |      k=4                  |         k=5 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![k=2](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/edge_probability/edge_probability_2_branches-1.png)  |  ![k=3](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/edge_probability/edge_probability_3_branches-1.png) |  ![k=4](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/edge_probability/edge_probability_4_branches-1.png)|  ![k=5](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/edge_probability/edge_probability_5_branches-1.png)



### Average Accuracy Before and After Calibration
This section evaluates the impact of calibration on the average edge accuracy in the same numerical setting considered in the previous section. The following figures show the average edge accuracy considering k side branches at the edge device.

The following figures show the average edge accuracy considering a early-exit MobileNetV2 evaluated on Caltech-256 dataset.
           k=2             |   k=3                     |      k=4                  |         k=5 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![k=2](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/accuracy/accuracy_edge_threshold_2_branches-1.png)  |  ![k=3](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/accuracy/accuracy_edge_threshold_3_branches-1.png) |  ![k=4](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/accuracy/accuracy_edge_threshold_4_branches-1.png)|  ![k=5](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/accuracy/accuracy_edge_threshold_5_branches-1.png)


The following figures show the average edge accuracy considering a early-exit ResNet18 evaluated on Caltech-256 dataset.
           k=2             |   k=3                     |      k=4                  |         k=5 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![k=2](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/accuracy/accuracy_edge_x_threshold_2_branches-1.png)  |  ![k=3](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/accuracy/accuracy_edge_x_threshold_3_branches-1.png) |  ![k=4](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/accuracy/accuracy_edge_x_threshold_4_branches-1.png)|  ![k=5](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/accuracy/accuracy_edge_x_threshold_5_branches-1.png)

The following figures show the average edge accuracy considering a early-exit VGG16 evaluated on Caltech-256 dataset.
           k=2             |   k=3                     |      k=4                  |         k=5 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![k=2](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/accuracy/accuracy_edge_x_threshold_2_branches-1.png)  |  ![k=3](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/accuracy/accuracy_edge_x_threshold_3_branch-1.png) |  ![k=4](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/accuracy/accuracy_edge_x_threshold_4_branch-1.png)|  ![k=5](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/accuracy/accuracy_edge_x_threshold_5_branch-1.png)

The following figures show the average edge accuracy considering a early-exit ResNet152 evaluated on Caltech-256 dataset.
           k=2             |   k=3                     |      k=4                  |         k=5 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![k=2](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/accuracy/accuracy_edge_x_threshold_2_branches-1.png)  |  ![k=3](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/accuracy/accuracy_edge_x_threshold_3_branches-1.png) |  ![k=4](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/accuracy/accuracy_edge_x_threshold_4_branches-1.png)|  ![k=5](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/accuracy/accuracy_edge_x_threshold_5_branches-1.png)



### Edge Inference Outage Probability
The next figures evaluated the edge inference outage probability versus the desired reliability level ptar for several early-exit DNN models on different datasets.
To obtain these results, we allocate a number k of side branches at the edge device and compute the edge inference outage probability as explained before. 


The following figures show the edge inference probability considering a early-exit MobileNetV2 evaluated on Caltech-256 dataset.
k = 3             |  k = 4          |  k = 5 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/inference_outage/inference_outage_x_threshold_3_branch-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/inference_outage/inference_outage_x_threshold_4_branch-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/inference_outage/inference_outage_x_threshold_5_branch-1.png) 

The following figures show the edge inference probability considering a early-exit ResNet18 evaluated on Caltech-256 dataset.
k = 3             |  k = 4          |  k = 5 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/inference_outage/inference_outage_x_threshold_3_branches-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/inference_outage/inference_outage_x_threshold_4_branches-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/inference_outage/inference_outage_x_threshold_5_branches-1.png) 

The following figures show the edge inference probability considering a early-exit VGG16 evaluated on Caltech-256 dataset.
k = 3             |  k = 4          |  k = 5 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/inference_outage/inference_outage_x_threshold_3_branches-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/inference_outage/inference_outage_x_threshold_4_branches-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/vgg16/caltech256/inference_outage/inference_outage_x_threshold_5_branches-1.png) 

The following figures show the edge inference probability considering a early-exit ResNet152 evaluated on Caltech-256 dataset.
k = 3             |  k = 4          |  k = 5 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/inference_outage/inference_outage_x_threshold_3_branch-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/inference_outage/inference_outage_x_threshold_4_branch-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet152/caltech256/inference_outage/inference_outage_x_threshold_5_branch-1.png) 


### Missed Deadline Probability
This section presents results of missed deadline probability as a function of the latency deadline for several early-exit DNN models on different datasets. As explained before, the event missed deadline occurs when the early-exit DNN model cannot meet the application-defined accuracy and latency requirements. Therefore, we compute missed deadline probability as a fraction of batches that the event missed deadline occurs. First, we focus on showing the results on Caltech-256 dataset.


The next figures show the results of missed deadline probability evaluating an early-exit MobileNetV2 considering the cloud location on Ohio, USA.

k = 3             |  k = 4          |  k = 5 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/missed_deadline/missed_deadline_3_branches_ptar_0.8_ohio_more_points-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/missed_deadline/missed_deadline_4_branches_ptar_0.8_ohio_more_points-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/missed_deadline/missed_deadline_5_branches_ptar_0.8_ohio_more_points-1.png) 

The next figures show the results of missed deadline probability evaluating an early-exit MobileNetV2 considering the cloud location on São Paulo, Brazil.

k = 3             |  k = 4          |  k = 5 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/missed_deadline/missed_deadline_3_branches_ptar_0.8_sp_more_points-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/missed_deadline/missed_deadline_4_branches_ptar_0.8_sp_more_points-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/mobilenet/caltech256/missed_deadline/missed_deadline_5_branches_ptar_0.8_sp_more_points-1.png) 

The next figures show the results of missed deadline probability evaluating an early-exit ResNet18 considering the cloud location on Ohio, USA.

k = 3             |  k = 4          |  k = 5 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/missed_deadline/missed_deadline_3_branches_ptar_0.8_ohio_more_points-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/missed_deadline/missed_deadline_4_branches_ptar_0.8_ohio_more_points-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/missed_deadline/missed_deadline_5_branches_ptar_0.8_ohio_more_points-1.png)

The next figures show the results of missed deadline probability evaluating an early-exit ResNet18 considering the cloud location on São Paulo, Brazil.

k = 3             |  k = 4          |  k = 5 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/missed_deadline/missed_deadline_3_branches_ptar_0.8_sp_more_point-1.png)  |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/missed_deadline/missed_deadline_4_branches_ptar_0.8_sp_more_point-1.png) |  ![](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/resnet18/caltech256/missed_deadline/missed_deadline_5_branches_ptar_0.8_ohio_more_points-1.png)
