# Predictive Funding Challenge 🌱

A machine learning challenge to predict relative funding between open source projects, aiming to create scalable and fair public goods funding allocation. This repository contains the code and resources for two challenges:

- [HuggingFace Competition](https://huggingface.co/spaces/DeepFunding/PredictiveFundingChallengeforOpenSourceDependencies)
- [CryptoPond Competition](https://cryptopond.xyz/modelfactory/detail/306250)

## 📦 Data

Each contest uses a similar dataset, it contains a pair of projects and the relative funding amounts in historical funding rounds.

## 🎯 Goal

The goal is predicting the relative funding received between any pair of projects. For that, we need to compare each of these repos with one another and give a relative value between them, such that the total in each case adds up to 1.

## 📊 Evaluation

*Winners* are decided based on *novelty and approach taken to predict answers for 1023 comparisons*. They are determined by their marginal contribution: how much better the final outcome is compared to if their submission (code or dataset) had never existed? That means that even if someone doesn't make a submission but provides a valuable dataset that all other contestants end up using, that would be rewarded.

The evaluation metric is **Mean Squared Error (MSE)**. The lower, the better.

Submission weights must be self-consistent, ie. for any triple _a_,_b_,_c_, `c/a = c/b * b/a`. Ensure mathematical consistency in outputs given to reflect logical relationships rather than reflecting biases from the training data.

## 🚀 Quickstart

Make sure you have [`uv` installed](https://docs.astral.sh/uv/). Then run the following command to install the dependencies.

```bash
uv sync
```

Once all the dependencies are installed, you can run any Notebook in the `notebooks` folder!

### 🔐 Environment

Create a `.env` file in the root directory with the following variables:

- `GITHUB_TOKEN`: A GitHub personal access token with rate limiting. You can create one on [GitHub Developer Settings](https://github.com/settings/tokens?type=beta).

## 📚 Resources

- [Official Website](https://deepfunding.org) - [FAQ](https://deepfunding.org/faq)
- [GitHub Repository](https://github.com/deepfunding/dependency-graph)
- [Demo of the Voting UI](https://pairwise-df-demo.vercel.app/allocation)
- [Deep Funding Podcast](https://www.youtube.com/watch?v=ygaEBHYllPU)
