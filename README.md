Vehicle speed prediction from dashboard camera.
The code implements a Convolutional LSTM model on a stacked tensor of Google Inception features and optical flow features computed using FlownetS, as demonstrated in https://github.com/ClementPinard/FlowNetPytorch.
The model achieves <20mph MSE on Comma AI's test set.
