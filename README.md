# Numerical_Project
Simulation of a biological neural network with a random graph, compute the transfert entropy with respect to a time lack and nodes indexes in order to detect a phase transition in the graph. 

# ⚙️ Installation and Environment Setup

This project, **bnn-package**, requires specific system libraries (GSL, FFTW) and a local compilation step for the `entropy` package to function correctly. Follow these steps to set up your environment on macOS.

## Prerequisites (macOS)

Before proceeding, ensure you have the following installed:

1.  **Xcode Command Line Tools:** Necessary for the `gcc`/`g++` compilers and `make`.
    ```bash
    xcode-select --install
    ```

2.  **Homebrew:** The package manager for macOS.
    * Installation instructions can be found on the [Homebrew website](https://brew.sh/).

3.  **Python 3:** Ensure Homebrew's Python is installed to provide necessary utilities like `python3-config`.
    ```bash
    brew install python3
    ```

---

## Step 1: Install System Dependencies (GSL & FFTW)

These are the libraries written in C/C++ that the `entropy` package depends on.

```bash
brew install gsl fftw

## Step 2: 

./configure CFLAGS=-I/opt/homebrew/include LDFLAGS=-L/opt/homebrew/lib
