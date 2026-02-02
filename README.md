# Python-LSTM Stock Price Prediction

This repository contains a Python implementation of a Long Short-Term Memory (LSTM) model for stock price prediction. The model is built using PyTorch and trained on historical stock price data fetched using the `yfinance` library.

## Features

- Downloads historical stock price data using `yfinance`.
- Preprocesses data with `StandardScaler` for normalization.
- Implements an LSTM model for time-series prediction.
- Trains the model on GPU (if available) for faster computation.
- Evaluates the model using Root Mean Squared Error (RMSE).
- Visualizes actual vs predicted stock prices and prediction errors.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- yfinance

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Python-LTSM.git
   cd Python-LTSM
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the `Main.ipynb` notebook in Jupyter Notebook or VS Code.
2. Run the cells step-by-step to:
   - Download stock price data.
   - Preprocess the data.
   - Train the LSTM model.
   - Evaluate and visualize the results.

## Model Architecture

The LSTM model is implemented using PyTorch and consists of:

- Input layer with 1 feature.
- 3 LSTM layers with 64 hidden units each.
- Fully connected output layer.
- Dropout for regularization.

## Results

- The model predicts stock prices based on historical data.
- RMSE is calculated for both training and testing datasets.
- Visualization includes actual vs predicted prices and prediction errors.

## Example Output

![Stock Price Prediction](example_output.png)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [scikit-learn](https://scikit-learn.org/)
