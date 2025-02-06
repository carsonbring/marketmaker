# MarketMaker

MarketMaker is a Python project that implements a Gaussian Hidden Markov Model (HMM) for time series analysis using PyTorch. It also includes utilities for retrieving and processing financial data from the Alpha Vantage API. The project uses Poetry for dependency management and packaging, and Pydantic for configuration.

## Features

- **Gaussian Hidden Markov Model**:  
  Implements forward, backward, Viterbi, and Baum–Welch (EM) algorithms in log–domain for continuous data.

- **Data Retrieval and Caching**:  
  Retrieves time series data (e.g., stock data) from the Alpha Vantage API and caches it locally using pickle.

- **Data Conversion**:  
  Converts Pandas DataFrames of financial data into PyTorch tensors for further analysis.

- **Configuration via Environment Variables**:  
  Uses Pydantic’s `BaseSettings` to load API keys and configuration options from a `.env` file.

## Project Structure

marketmaker/
├── data.pkl # Cached data file (generated after first run)
├── data.py # Module for data retrieval and processing
├── discrete_hmm.py # (Optional) Discrete HMM implementation
├── gaussian_hmm.py # Gaussian HMM implementation using PyTorch
└── main.py # Entry point to run the project
pyproject.toml # Poetry project configuration
poetry.lock # Lock file for Poetry dependencies
README.md # This README file

## Installation

### Prerequisites

- Python 3.12^ (or compatible version)
- [Poetry](https://python-poetry.org/) (for dependency management)
