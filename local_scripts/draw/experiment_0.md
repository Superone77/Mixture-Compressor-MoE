Experiment 1.1: Expert Activation Heatmaps (for Figure 1)

Goal: Prove that calibration data fails to activate deep-layer experts.

Method:

Take a pre-trained MoE model (e.g., Mixtral8x7B).

Run inference on the WikiText2 calibration set.

Run inference on GSM8K (and perhaps C4 or Code) validation sets.

Plot: A heatmap or bar chart showing "Expert Utilization Rate" per layer.

Expected Result: WikiText2 shows 0% activation for certain experts in deep layers, whereas GSM8K activates them.

Visualization:

Experiment 1.2: Cross-Domain Performance Drop (for Figure 2)

Goal: Prove that data-driven calibration overfits to the calibration domain.

Method:

Quantize the model using calibration data from WikiText2. Call this Model_Wiki.

Quantize the model using calibration data from GSM8K. Call this Model_GSM.

Evaluate Model_Wiki on WikiText2 Test and GSM8K Test.

Evaluate Model_GSM on WikiText2 Test and GSM8K Test.

Plot: A grouped bar chart showing Perplexity (PPL) or Accuracy.

Expected Result: Model_Wiki beats Model_GSM on WikiText2 but loses significantly on GSM8K.