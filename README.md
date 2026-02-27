# 🤖 ML Learning Journey

A structured 12-week hands-on program to go from ML-aware software engineer to a confident ML engineer capable of building and deploying end-to-end AI systems.

This repository documents every step of the journey — from foundational concepts to production-ready projects — and serves as a living portfolio of applied machine learning work.

---

## 👤 About

**Background:** ML Software Engineer transitioning into ML/AI Engineering II 
**Goal:** Build real, deployable AI projects across computer vision, NLP, and recommender systems  
**Timeline:** 12 weeks | 3–5 hours/week  
**Stack:** Python · PyTorch · scikit-learn · HuggingFace · LangChain · FastAPI · Docker · MLflow

---

## 🗺️ Program Structure

The program is divided into 3 phases, each building on the previous:

| Phase | Focus | Weeks |
|---|---|---|
| [Phase 1 — Foundations](./foundations/) | ML mechanics, neural nets, Transformers | 1–3 |
| [Phase 2 — Projects](./projects/) | Computer vision, NLP, recommender system | 4–8 |
| [Phase 3 — MLOps](./mlops/) | Deployment, tracking, monitoring, capstone | 9–12 |

---

## 📦 Phase 1 — Foundations

> *Filling the gaps that hurt you later: backpropagation, CNN/RNN/Transformer architecture intuition, and hands-on implementation from scratch.*

### [01 — Backpropagation from Scratch](./foundations/backprop_from_scratch)
Implementing a fully connected neural network using only NumPy — forward pass, loss computation, backpropagation, and gradient descent — to deeply understand what happens under the hood before using any framework.

**Concepts covered:** Chain rule · Vanishing gradients · Binary cross-entropy · Weight updates

---

### [02 — CNN Intuition & Image Classifier](./foundations/cnn_intuition/)
*Coming in Week 2*

**Concepts covered:** Convolutions · Pooling · Feature maps · Transfer learning

---

### [03 — Transformers & the Attention Mechanism](./foundations/transformers_intro/)
*Coming in Week 3*

**Concepts covered:** Self-attention · Encoder vs decoder · BERT vs GPT · HuggingFace intro

---

## 🔨 Phase 2 — Projects

> *One real project per focus area, each introducing new tools and pushing toward production quality.*

### [Image Classifier](./projects/image_classifier/)
*Coming in Weeks 4–5*

A computer vision model trained on a custom image dataset using transfer learning with ResNet. Includes data augmentation, evaluation metrics, and a clean inference interface.

**Stack:** PyTorch · torchvision · ResNet

---

### [NLP Text Summarizer / Chatbot](./projects/nlp_text_summarizer/)
*Coming in Weeks 6–7*

A text summarization or conversational app built with HuggingFace Transformers and LangChain, exploring tokenization, embeddings, and RAG basics.

**Stack:** HuggingFace Transformers · LangChain · OpenAI API

---

### [Recommendation System](./projects/recommendation_system/)
*Coming in Week 8*

A content or collaborative filtering recommender system, exploring how embeddings power modern recommendation engines.

**Stack:** scikit-learn · PyTorch · Surprise

---

## 🚀 Phase 3 — MLOps

> *Turning projects into real deployable systems: tracked, containerized, served, and monitored.*

### [Experiment Tracking](./mlops/experiment_tracking/)
*Coming in Week 9*

Wrapping a Phase 2 project with proper experiment tracking using MLflow — logging metrics, parameters, and model artifacts across runs.

**Stack:** MLflow · Weights & Biases

---

### [Model Deployment API](./mlops/model_deployment_api/)
*Coming in Week 10*

Serving a trained model as a REST API with FastAPI, containerized with Docker and ready for cloud deployment.

**Stack:** FastAPI · Docker

---

### [Monitoring & Data Drift](./mlops/monitoring/)
*Coming in Week 11*

Detecting data drift and performance degradation in production using Evidently AI, with a simulated drift scenario.

**Stack:** Evidently AI

---

### [Capstone Project](./mlops/capstone/)
*Coming in Week 12*

A production-ready end-to-end ML system combining the best of Phase 2 and Phase 3: trained, tracked, deployed as an API, containerized, and monitored.

---

## 🧰 Tech Stack

| Purpose | Tool |
|---|---|
| Deep learning | PyTorch |
| Classical ML | scikit-learn |
| NLP / LLMs | HuggingFace Transformers |
| LLM apps | LangChain |
| Experiment tracking | MLflow · Weights & Biases |
| Deployment | FastAPI · Docker |
| Monitoring | Evidently AI |
| Environment | Python 3.10+ · conda/venv |

---

## 📈 Progress

- [x] Assessment & learning plan
- [x] Phase 1 — Week 1: Backpropagation from scratch
- [ ] Phase 1 — Week 2: CNN intuition & image classifier
- [ ] Phase 1 — Week 3: Transformers & attention
- [ ] Phase 2 — Weeks 4–5: Image classifier project
- [ ] Phase 2 — Weeks 6–7: NLP app
- [ ] Phase 2 — Week 8: Recommendation system
- [ ] Phase 3 — Week 9: Experiment tracking
- [ ] Phase 3 — Week 10: Model deployment
- [ ] Phase 3 — Week 11: Monitoring
- [ ] Phase 3 — Week 12: Capstone

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/danielasicaru/ml-learning-journey.git
cd ml-learning-journey

# Create a virtual environment
python -m venv ml_learning_env
source ml_learning_env/bin/activate 

# Install base dependencies
pip install -r requirements.txt
```

Each sub-project has its own `README.md` with specific setup instructions and how to run it.

---

## 📬 Contact

Feel free to connect or follow the journey:  
[LinkedIn](https://www.linkedin.com/in/sicarudaniela/) · [GitHub](https://github.com/danielasicaru)