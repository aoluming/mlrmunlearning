# MLRM Unlearning: Machine Unlearning for Vision-Language Models

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/aoluming/mlrmunlearning?style=social)](https://github.com/aoluming/mlrmunlearning/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/aoluming/mlrmunlearning?style=social)](https://github.com/aoluming/mlrmunlearning/network/members)
[![GitHub issues](https://img.shields.io/github/issues/aoluming/mlrmunlearning)](https://github.com/aoluming/mlrmunlearning/issues)
[![License](https://img.shields.io/github/license/aoluming/mlrmunlearning)](https://github.com/aoluming/mlrmunlearning/blob/main/LICENSE)

**Trajectory-Level Unlearning Framework for Multi-modal Large Language Models**

</div>

## 📖 Overview

This project implements advanced **Machine Unlearning** techniques for Vision-Language Models (VLMs), specifically targeting the **R1-Onevision** model. We introduce two novel unlearning approaches:

1. **COCONUT** - Contrastive-based unlearning with latent token optimization
2. **Trajectory-Level Unlearning** - Dynamic monitoring and intervention during reasoning

## ✨ Features

### 🎯 Core Capabilities

- **Answer-Level Unlearning**: Gradient ascent on target answer tokens
- **Trajectory-Level Monitoring**: Real-time risk assessment of latent reasoning steps
- **Dynamic Intervention**: Automatic redirection of high-risk latent states
- **Multi-Layer Protection**: Combines GA, Contrastive Loss, Critic, and Redirector

### 🔬 Novel Components

- **Logit-Lens Critic**: Diagnoses risk at each latent reasoning step
- **Latent Redirector**: Dynamically rewrites high-risk latent states to safe trajectories
- **COCOUN Tokens**: Special latent tokens for controlled reasoning

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Image + Question                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Prefix Encoding (Vision + Language)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Latent Reasoning Steps (with Monitoring)        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Step 1: Logit-Lens Critic → Risk Assessment         │  │
│  │  Step 2: Redirector → High-risk State Intervention    │  │
│  │  Step 3: GA + Contrastive Loss → Unlearning           │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Answer Generation                       │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/aoluming/mlrmunlearning.git
cd mlrmunlearning
pip install -r requirements.txt
```

### Training

#### Method 1: COCONUT (Baseline)

```yaml
# config.yaml
stage: cocoun
model_name_or_path: /path/to/R1-Onevision-7B
forget_concept: "Joe Biden"
unlearning_loss_weight: 1.0
unlearn_lm_loss_weight: 20
```

```bash
llamafactory-cli train config.yaml
```

#### Method 2: Trajectory-Level Unlearning (Recommended)

```yaml
# config_trajectory.yaml
stage: cocoun_trajectory
model_name_or_path: /path/to/R1-Onevision-7B
forget_concept: "Joe Biden"
risk_threshold: 0.1          # Critic risk threshold
redirect_strength: 0.5       # Redirector intensity
unlearning_loss_weight: 1.0
unlearn_lm_loss_weight: 20
```

```bash
llamafactory-cli train config_trajectory.yaml
```

### Inference

```bash
python infer.py
```

## 📊 Performance

| Method | GA Loss | Contrastive Loss | Trajectory Risk | Forget Rate |
|--------|---------|-----------------|-----------------|-------------|
| COCONUT | ✓ | ✓ | ✗ | Baseline |
| **Trajectory** | ✓ | ✓ | ✓ | **Better** |

## 🔧 Configuration

### Key Parameters

- `risk_threshold` (default: 0.1): Threshold for triggering latent redirection
- `redirect_strength` (default: 0.5): Intensity of latent state redirection
- `unlearn_lm_loss_weight` (default: 20): Weight for gradient ascent
- `unlearning_loss_weight` (default: 1.0): Weight for contrastive loss

### Model Cards

- **Base Model**: R1-Onevision-7B (Qwen2.5-VL based)
- **Training Data**: Custom unlearning datasets
- **Quantization**: 4-bit (optional)

## 📁 Project Structure

```
mlrmunlearning/
├── src/llamafactory/
│   ├── model/
│   │   ├── cocoun_model.py              # COCONUT baseline
│   │   ├── cocoun_trajectory_model.py   # Trajectory model
│   │   ├── load_cocoun.py
│   │   └── load_cocoun_trajectory.py
│   ├── train/
│   │   ├── cocoun/                      # COCONUT training workflow
│   │   └── trajectory/                  # Trajectory training workflow
│   └── hparams/
│       └── finetuning_args.py           # Extended arguments
├── examples/train_lora/
│   ├── qwen2vl_cocoun.yaml             # COCONUT config
│   └── qwen2vl_cocoun_trajectory.yaml   # Trajectory config
└── infer.py                             # Inference script
```

## 🧪 Experimental Results

Our trajectory-level unlearning approach demonstrates:

- ✅ **Enhanced Unlearning**: Better forget rates on target concepts
- ✅ **Maintained Utility**: Preserves model capabilities on non-target tasks
- ✅ **Real-time Monitoring**: Detects and mitigates risky reasoning paths
- ✅ **Stable Training**: No significant degradation in model performance

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@software{mlrmunlearning,
  title = {MLRM Unlearning: Trajectory-Level Machine Unlearning for VLMs},
  author = {aoluming},
  year = {2025},
  url = {https://github.com/aoluming/mlrmunlearning}
}
```

## 🙏 Acknowledgments

- Built upon [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- Base model: [R1-Onevision](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Inspired by recent advances in machine unlearning

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- GitHub: [@aoluming](https://github.com/aoluming)
- Email: 494296118@qq.com

---

<div align="center">

**⭐ Star this repo if it helps you!**

Made with ❤️ by the MLRM Unlearning Team

</div>
