# Individual Assignment: Coin Toss Game and Rolling Window Regression Analysis

## Overview

This project consists of two independent analyses performed using Python:

- **Coin Toss Game:** Investigation of how splitting bets over multiple coin tosses affects the expected payoff and its risk (standard deviation).
- **Rolling Window Regression Analysis:** Calculation of rolling alpha and beta parameters for Apple Inc. stock relative to the market index, assessing how these financial indicators evolve over time.

## Project Structure

### 1. Coin Toss Game

- **Objective:** Explore the relationship between the number of coin tosses and the variability (risk) of the payoff.
- **Tasks:**
  - Calculate expected payoffs and standard deviations for different numbers of coin tosses.
  - Implement simulations using:
    - Basic Python arithmetic
    - For loops and itertools product
    - NumPy and Pandas libraries
  - Develop and use a generalized function (`gameN`) to iteratively compute and visualize risk reduction as the number of tosses increases.

### Key Findings:

- Risk decreases rapidly as the number of tosses increases.
- Standard deviation scales approximately as \( \frac{1}{\sqrt{n}} \), consistent with the Law of Large Numbers.

### 2. Rolling Window Regression Analysis

- **Objective:** Assess the dynamic relationship between Apple Inc.'s stock returns and the market (S&P 500) by calculating rolling alpha (intercept) and beta (slope) coefficients.
- **Tasks:**
  - Data acquisition from Yahoo Finance (AAPL stock prices, S&P 500, and risk-free rates).
  - Calculation of daily excess returns.
  - Implementation of a rolling window regression analysis (3-month window or 63 observations).
  - Visualization of the evolution of alpha and beta parameters over time.

### Key Observations:

- Alpha indicates periods when Apple Inc. outperformed or underperformed the market, adjusted for risk.
- Beta variations reveal shifts in the stock's volatility relative to the market over time.

## Technologies and Libraries

- **Python 3**
- **Libraries Used:**
  - `numpy` (numerical computations)
  - `pandas` (data management and manipulation)
  - `matplotlib` (data visualization)
  - `sklearn` (linear regression modeling)
  - `yfinance` (financial data retrieval)

## Files Included

- `IA_AM02_Nov2024_IagoPuenteEsteban.py`: Python script containing code implementation and analyses.
- `IA_AM02_Nov2024_IagoPuenteEsteban.pdf`: Project documentation and assignment guidelines.

## How to Run the Project

- Ensure Python and required libraries are installed.
- Run the Python script to execute simulations, perform rolling window regressions, and generate visualizations of results.

## Author

- **Iago Puente Esteban**
- Contact: iago.puente@gmail.com
