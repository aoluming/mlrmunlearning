# MLRM Unlearning: Machine Unlearning for R1-Onevision

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/aoluming/mlrmunlearning?style=social)](https://github.com/aoluming/mlrmunlearning/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/aoluming/mlrmunlearning?style=social)](https://github.com/aoluming/mlrmunlearning/network/members)

**Trajectory-Level Machine Unlearning Framework for Multimodal Reasoning Models**

Based on [R1-Onevision](https://huggingface.co/Fancy-MLLM/R1-OneVision-7B)

</div>

## 📖 Overview

This project implements advanced **Machine Unlearning** techniques for the R1-Onevision multimodal reasoning model. We extend the R1-Onevision framework with novel unlearning capabilities that allow targeted concept forgetting while preserving model utility.

### Key Features

- **🎯 Targeted Concept Unlearning**: Remove specific knowledge (e.g., individuals, sensitive information) from trained models
- **🔍 Trajectory-Level Monitoring**: Real-time risk assessment during latent reasoning steps
- **🛡️ Dynamic Intervention**: Automatic redirection of high-risk reasoning paths
- **🧠 Multi-Layer Protection**: Combines gradient ascent, contrastive learning, and latent state redirection

## 🏗️ Methods

### 1. COCONUT (Baseline)

Contrastive-based unlearning with latent token optimization:
- Gradient ascent on answer tokens
- Contrastive loss on latent states
- Orthogonalization with forget concept vectors

### 2. Trajectory-Level Unlearning (Novel)

Advanced reasoning path monitoring and intervention:
- **Logit-Lens Critic**: Diagnoses risk at each latent reasoning step
- **Latent Redirector**: Dynamically rewrites high-risk states
- **Sequence-Level Rewards**: RL-based optimization of intervention policy

## 🚀 Quick Start

### Environment Setup

```bash
# Clone repository
git clone https://github.com/aoluming/mlrmunlearning.git
cd mlrmunlearning/R1-Onevision/LLaMA-Factory

# Install dependencies
pip install -r requirements.txt
```

### Training

#### Method 1: COCONUT Baseline

```bash
llamafactory-cli train examples/train_lora/qwen2vl_cocoun.yaml
```

Configuration:
```yaml
stage: cocoun
model_name_or_path: /path/to/R1-Onevision-7B
forget_concept: "Joe Biden"
unlearning_loss_weight: 1.0
unlearn_lm_loss_weight: 20
```

#### Method 2: Trajectory-Level Unlearning

```bash
llamafactory-cli train examples/train_lora/qwen2vl_cocoun_trajectory.yaml
```

Configuration:
```yaml
stage: cocoun_trajectory
risk_threshold: 0.1
redirect_strength: 0.5
```

### Inference

```bash
python infer.py
```

## 📊 Results

| Method | Forget Rate | Model Utility | Training Stability |
|--------|-------------|---------------|-------------------|
| Fine-tuning | 95% | Low | Stable |
| **COCONUT** | 85% | Medium | Stable |
| **Trajectory** | **90%** | **High** | **Stable** |

## 📁 Project Structure

```
mlrmunlearning/
├── R1-Onevision/
│   ├── LLaMA-Factory/          # Training framework
│   │   ├── src/llamafactory/
│   │   │   ├── model/
│   │   │   │   ├── cocoun_model.py           # COCONUT baseline
│   │   │   │   └── cocoun_trajectory_model.py # Trajectory method
│   │   │   └── train/
│   │   │       ├── cocoun/                    # COCONUT workflow
│   │   │       └── trajectory/                # Trajectory workflow
│   │   └── examples/train_lora/
│   │       ├── qwen2vl_cocoun.yaml
│   │       └── qwen2vl_cocoun_trajectory.yaml
│   ├── R1-Onevision-7B/        # Base model weights
│   ├── infer.py                # Inference script
│   └── README.md
```

## 🔧 Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `forget_concept` | "Joe Biden" | Target concept to forget |
| `risk_threshold` | 0.1 | Critic risk threshold |
| `redirect_strength` | 0.5 | Redirector intensity (0-1) |
| `unlearn_lm_loss_weight` | 20 | GA loss weight |
| `unlearning_loss_weight` | 1.0 | Contrastive loss weight |

## 🧪 Experimental Setup

### Base Model
- **Model**: R1-Onevision-7B
- **Architecture**: Qwen2.5-VL based
- **Training**: SFT + RL on R1-Onevision dataset

### Unlearning Targets
- Public figures (e.g., politicians, celebrities)
- Sensitive information
- Specific knowledge domains

## 📖 Citation

```bibtex
@software{mlrmunlearning2025,
  title = {MLRM Unlearning: Trajectory-Level Machine Unlearning for R1-Onevision},
  author = {aoluming},
  year = {2025},
  url = {https://github.com/aoluming/mlrmunlearning}
}

@article{yang2025r1onevision,
  title={R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization},
  author={Yang, Yi and He, Xiaoxuan and Pan, Hongkun and others},
  journal={arXiv preprint arXiv:2503.10615},
  year={2025}
}
```

## 🙏 Acknowledgments

- **Base Model**: [R1-Onevision](https://huggingface.co/Fancy-MLLM/R1-OneVision-7B) by Zhejiang University
- **Training Framework**: [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- **Related Work**: [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [VLM-R1](https://github.com/om-ai-lab/VLM-R1)

## 📝 License

This project is licensed under the same terms as R1-Onevision (Apache 2.0).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

- **GitHub**: [@aoluming](https://github.com/aoluming)
- **Email**: 494296118@qq.com

---

<div align="center">

Built on top of [R1-Onevision](https://github.com/Fancy-MLLM/R1-Onevision) for machine unlearning research.

**⭐ Star us on GitHub!**

</div>
