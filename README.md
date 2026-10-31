# Tokenising the Order-Event Stream for Generative Models of Order Flow

Code for my Master's thesis, *Generative Models in Market Microstructure: Tokenising the Order-Event Stream*. The thesis was written for the M2 MMMEF at Université Paris 1 Panthéon-Sorbonne, under the supervision of Fabián Aguirre-Lopez, and the defence is upcoming.

## The question
Generative models of order flow are built like language models, with one token per order event. That forces a choice that text never poses: where to cut continuous delays, sizes and price distances into discrete bins. This project rebuilds the TradeFM pipeline (Kawawa-Beaudan et al., 2026) end to end and changes **only the binning rule**. The question is what that rule changes for a decoder-only transformer trained on the resulting tokens.

## What the notebook does
- **Data**: the Hyperliquid BTC Level-3 order-lifecycle stream of Albers et al. (2026), CC BY 4.0, for December 2025. The model trains on 1, 2 and 8 December (31.1M events) and is tested on 22 and 29 December, which it never sees in training.
- **Features**:
  - the block-level inter-arrival time;
  - the log order size;
  - the depth relative to an exponentially weighted VWAP mid-price, which uses passive fills only.
- **Tokenisers**, each with 16 bins per feature and a shared vocabulary of 40,960 tokens:
  - **Fractile(s)**: a frozen rank map composed with a symmetric power warp `w(u) = u^(1/s) / (u^(1/s) + (1-u)^(1/s))`, with a single parameter;
  - **TradeFM's hybrid rule**, used as the baseline.
- **Round-trip test** (no model involved): measures how much each code keeps. At s = 2 the fractile code recovers 92% of the aggregate order size, against 66% for TradeFM's rule. The choice of s is linked to companding-quantiser theory (Bennett, Panter–Dite).
- **Model**: a 25.19M-parameter decoder-only transformer in PyTorch. It uses RoPE, grouped-query attention and a Δp conditioning embedding. It is trained with AdamW, cosine schedule and mixed precision on a single T4 GPU.
- **Generation and evaluation**:
  - 512-event real prompts followed by 256-event rollouts;
  - KS and Wasserstein distances against the real flow, with 95% bootstrap intervals;
  - a noise floor, autocorrelations and robustness checks (1,024-event window, unseen Saturday).

## Main result
Trained on the fractile code, the model generates order sizes about **twice as close** to the real distribution, in every run on the test days. The KS distance is 0.081, against 0.159–0.180 for TradeFM's rule. The limits are documented in the notebook and the thesis:
- no code improves step-by-step prediction;
- the advantage falls within the noise floor out of regime.

## Running it
The notebook was written for Google Colab with a GPU. Download the Hyperliquid data from Albers et al. (2026), then set `path_Data` and `path_Models`. The comments in the code are in French.

## Author
Abdoulaye Hottmann Bamba · [LinkedIn](https://www.linkedin.com/in/hottmann)
