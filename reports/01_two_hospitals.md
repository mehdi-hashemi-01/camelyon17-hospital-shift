# Two hospitals, eight predictions

Each row in the toy CSV represents a different patch. Both classifiers choose the same class on every row. Model A assigns its chosen class a probability of 0.75 and model B assigns it 0.95.

For U1, model B assigns class 1 a probability of 0.05. Its prediction is class 0 with confidence 0.95. The label is 0, so this prediction is correct. For U2 the probabilities are identical, but the label is 1 and the prediction is wrong.

In the source hospital, both models get three of four patches right. In the unseen hospital, they get two of four right. Model B has a larger confidence gap in each hospital. The signed gap is mean confidence minus accuracy. A positive value indicates average overconfidence. Each model uses a single confidence level here. With varying confidence levels, errors in opposite directions can cancel in a pooled mean.

The hospitals share the same two labels. To check a 90% confidence claim, gather representative labeled predictions near 90% confidence from the domain of interest and compare their mean confidence with their observed accuracy. Four patches per hospital illustrate the calculation. A research estimate needs larger groups and uncertainty estimates that account for patches from the same slide.
