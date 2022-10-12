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
early-exit DNNs.](https://github.com/pachecobeto95/early_exit_calibration/blob/main/plots/motivation/adaptive_offloading_via_ee_comnet2.pdf)


As illustrated in Figure 1, the early-exit DNN is split into two parts: the first part being implemented on the edge device and the second remotely at the
cloud server. Side branches are included only in the first part. At inference time, given an input **x**, the device estimates the prediction confidence for the
i-th side branch as the probability of the most likely class. If the confidence value is greater or equal to a predefined target threshold ptar,
the device concludes that the i-th side branch can classify the input and inference terminates by classifying the input as the class with the largest probability. Otherwise, the input is processed by the subsequent layers until it reaches the next side branch, following the same procedure described above. If no side branches reach the desired accuracy level ptar, the edge device offloads data to the cloud, which processes the remaining DNN backbone’s layers until the output layer. The cloud server then sends the prediction and its confidence level back to the edge device. If the confidence level does not reach
the target ptar, the classification uses the most confident predicted class among all exits.

