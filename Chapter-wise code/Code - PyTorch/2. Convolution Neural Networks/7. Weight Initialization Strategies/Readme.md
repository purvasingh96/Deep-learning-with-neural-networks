# Weight Initialization Strategies

This project talks about how you can initialize weights for your neural network for better accuracy. The below table summarizes results of using various weights and their compares them according to training and validation loss.

PyTorch implementation : [weight_initializaion_strategies](https://github.com/purvasingh96/Deep-learning-with-neural-networks/blob/master/Chapter-wise%20code/Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/7.%20Weight%20Initialization%20Strategies/Weight_initialization.ipynb)

## Results and Conclusion
| Weight Initialization Strategy               | Comments                                                                                                                                    | Results                                               |
|----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| Uniform weight initialization with 0s and 1s | The neural network has a hard time determining which<br> weights need to be changed, <br> since the neurons have the same output<br> for each layer | <img src="./images/zeros_vs_ones.png"></img>          |
| Uniform distribution between 0.0 and 1.0     | Better than case-1. Neural network<br> starts to learn.                                                                                          | <img src="./images/uniform_weights.png"></img>        |
| General Rule                                 | Model learns perfectly and training loss decreases gradually                                                                                | <img src="./images/general_rule.png"></img>           |
| Normal distribution v/s general rule         | Performs similar to general rule. Model<br> learns effectively.                                                                                 | <img src="./images/normal_vs_general.png"></img>      |
| No weight initialization                     | Unexpected behaviour. PyTorch has its own default weight initialization strategy                                                            | <img src="./images/default_initialization.png"></img> |
