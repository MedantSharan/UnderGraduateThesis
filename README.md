# Four-Player Mancala Reinforcement Learning Framework

A comprehensive framework for exploring multi-agent reinforcement learning in a novel four-player Mancala variant using Q-learning with experience replay.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Research Contributions](#research-contributions)
- [Quick Start](#quick-start)
- [Game Rules](#game-rules)
- [Experimental Framework](#experimental-framework)
- [Algorithm Details](#algorithm-details)
- [Results Summary](#results-summary)
- [File Structure](#file-structure)
- [Usage Examples](#usage-examples)
- [Dependencies](#dependencies)
- [Performance Notes](#performance-notes)
- [Limitations and Future Work](#limitations-and-future-work)
- [Statistical Validation](#statistical-validation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Copyright Disclaimer](#copyright-disclaimer)

## Overview

This project implements and evaluates a generalized four-player Mancala game environment designed for reinforcement learning research. The framework demonstrates how Q-learning agents can master complex strategic board games, achieving win rates up to **50.4%** in single-agent scenarios and **62%** in dual-agent configurations—significantly exceeding theoretical baselines.

## Key Features

- 🎮 **Flexible Game Environment**: Customizable board configurations with variable pit counts and stone distributions
- 🤖 **Multi-Agent Support**: Independent Q-learning agents with experience replay
- 📊 **Comprehensive Analysis**: Parameter studies, positional analysis, and emergent behavior investigation
- 📈 **Statistical Validation**: Rigorous statistical testing with confidence intervals
- 🎨 **Visualization Pipeline**: Publication-quality plots and heatmaps for result analysis

## Research Contributions

### Strategic Sweet Spots

The research identifies optimal board configurations where reinforcement learning agents significantly outperform theoretical expectations:

| Configuration | Win Rate | Baseline |
|---------------|----------|----------|
| 6 pits, 2 stones | **50.4%** | 25% random |
| Adjacent positioning (Players 1&2) | **62%** combined | 50% theoretical |

### Emergent Behaviors

- 🤝 **Pseudo-cooperative dynamics**: Independently trained agents develop complementary strategies
- 🎯 **Positional advantages**: Early turn positions provide consistent strategic benefits
- 📉 **Non-linear complexity effects**: Performance varies non-monotonically with game parameters

## Quick Start

### 🚀 Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `SourceCode.ipynb` notebook
3. Run cells sequentially from top to bottom

> 💡 **For faster results**: Search for `#TRAININGPARAMETER` and reduce `train_episodes` from 3000 to 300-500.

### System Requirements

- **Google Colab**: Web browser and Google account
- **Local**: Python 3.7+, 8GB RAM (recommended), 4GB disk space

## Game Rules

### Four-Player Mancala Mechanics

```
Player 4    [R]  P3  P2  P1  [R]    Player 1
           P8  P7  P6  P5  P4  P3
           P1  P2  P3  P4  P5  P6
Player 3    [R]  P1  P2  P3  [R]    Player 2
```

**Setup**: Each player controls N pits initially containing M stones, plus one reservoir

**Turns**: Players move counterclockwise, distributing stones one per pit

**Special Rules**:
- ❌ Skip opponents' reservoirs when distributing
- ➕ Land in your reservoir → earn extra turn
- 🎯 Land in empty pit on your side → capture stones from opposite player's corresponding pit

**Victory**: Game ends when any player's pits are empty; most stones in reservoir wins

### Key Modifications from Traditional Mancala

| Feature | Traditional | Four-Player Variant |
|---------|-------------|-------------------|
| **Capture Rule** | All opponents | Only directly opposite player |
| **Game End** | One player empty | Any player empty |
| **Dynamics** | Two-player zero-sum | Four-player positional strategy |

## Experimental Framework

### Parameter Studies

Systematic exploration of board configurations:

- **Pit counts**: 3, 4, 6, 8 pits per player
- **Stone counts**: 2, 3, 4, 6 initial stones per pit
- **Total configurations**: 16 distinct environments tested

### Multi-Agent Analysis

| Scenario | Description | Purpose |
|----------|-------------|---------|
| Single agent vs 3 random | Performance baseline establishment | Baseline metrics |
| Dual agents vs 2 random | Cooperative/competitive dynamics | Multi-agent behavior |
| Positional studies | Adjacent vs opposite placement effects | Position analysis |

## Algorithm Details

### Q-Learning Implementation

```python
# Key hyperparameters
learning_rate = 0.1        # α
discount_factor = 0.95     # γ
exploration_decay = "ε-greedy"  # 1.0 → 0.01
experience_replay = {
    "buffer_size": 10000,
    "batch_size": 32
}
```

### Reward Structure

| Action | Reward |
|--------|--------|
| Stone collection | +0.5 per stone moved to reservoir |
| Captures | +2.0 per captured stone |
| Extra turns | +3.0 bonus |
| End-game bonuses | Based on final ranking |

## Results Summary

### 🏆 Performance Highlights

- **Best single-agent**: 50.4% win rate (6p-2s configuration)
- **Best dual-agent**: 62% combined win rate (adjacent positioning)  
- **Statistical significance**: All major findings validated with p < 0.05
- **Convergence**: Stable performance achieved within 3,000 episodes

### 💡 Key Insights

1. **Non-monotonic complexity**: Intermediate complexity creates optimal learning conditions
2. **Position matters**: Early turn order provides measurable advantages
3. **Emergent cooperation**: Independent agents develop mutually beneficial strategies
4. **Configuration sensitivity**: Small parameter changes dramatically affect learning outcomes

## File Structure

```
📁 four-player-mancala-rl/
├── 📓 SourceCode.ipynb          # Main Jupyter notebook (recommended)
├── 🐍 SourceCode.py            # Python script version (requires modifications)
├── 📄 README.md               # This file
└── 📁 generated_files/        # Output directory for results and plots
    ├── 📦 *.pkl              # Experiment results data
    ├── 🖼️ *.png              # Generated visualizations
    └── 📊 *.csv              # Statistical summaries
```

## Usage Examples

### Basic Parameter Study

```python
# Run comprehensive parameter analysis
param_results = run_mancala_experiments('param_study')
```

### Custom Configuration Testing

```python
# Test specific board configuration
custom_config = {'pits': 4, 'stones': 3, 'label': 'Custom Board'}
positioning_results = run_mancala_experiments('positioning', config=custom_config)
```

### Dual Agent Analysis

```python
# Analyze multi-agent dynamics
dual_results = run_mancala_experiments('dual_agent')
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and numerical computing |
| `matplotlib` | Plotting and visualization |
| `gymnasium` | RL environment framework |
| `pandas` | Data manipulation and analysis |
| `scipy` | Statistical testing |
| `tqdm` | Progress tracking |
| `seaborn` | Statistical visualization |

### Installation

```bash
pip install numpy matplotlib gymnasium pandas scipy tqdm seaborn
```

## Performance Notes

- ⏱️ **Training time**: 3,000 episodes per configuration (~5-10 minutes per configuration on standard hardware)
- 💾 **Memory usage**: Sparse Q-table representation manages memory efficiently
- 📈 **Scalability**: Framework tested up to 8 pits, 6 stones (state space ~10^16)

## Limitations and Future Work

### ⚠️ Current Limitations

- Fixed 3,000 episode training duration may be suboptimal for some configurations
- Evaluation primarily against random agents
- Independent training may not capture full multi-agent potential
- Potential first-player advantage requires further investigation

### 🔮 Future Directions

- **Advanced algorithms**: Deep Q-Networks, policy gradient methods
- **True multi-agent training**: Simultaneous agent development
- **Human player comparison**: Strategy analysis against human expertise
- **Transfer learning**: Strategy generalization across configurations

## Statistical Validation

All reported results include:

- ✅ **Confidence intervals**: 95% confidence bounds for all metrics
- 📊 **Significance testing**: Binomial tests for win rates, t-tests for score differences
- 📏 **Effect sizes**: Practical significance beyond statistical significance
- 🔧 **Multiple testing corrections**: Bonferroni adjustments where appropriate

## Citation

If you use this framework in your research, please cite:

```bibtex
@techreport{sharan2025mancala,
  title={Exploring Multi-Agent Reinforcement Learning in a Four-Player Mancala Variant},
  author={Sharan, M.},
  institution={King's College London},
  year={2025},
  type={Final Project Report}
}
```

## License

This project is released under the MIT License. See the source code for full license details.

## Acknowledgments

- 🎓 King's College London Department of Informatics
- 🔧 OpenAI Gymnasium framework
- 🎮 Gary MacLeod's four-player Mancala variant as foundational inspiration

## Contact

For questions about the implementation or research methodology, please refer to the detailed technical documentation in the accompanying report.

---

**Note**: This framework is designed for research and educational purposes. The implementation provides a foundation for exploring multi-agent reinforcement learning in traditional board games and can be extended for investigating other strategic multiplayer environments.

---

## Copyright Disclaimer

### ⚖️ IMPORTANT LEGAL NOTICE

This work and all associated materials are protected under **British and Indian copyright law**. The content, methodology, code implementation, research findings, and documentation contained within this repository are the intellectual property of the author(s).

### 🚨 Unauthorized Usage Warning

**Unauthorized copying, reproduction, distribution, or use of any portion of this work without explicit written permission constitutes copyright infringement and plagiarism.**

Any attempt to copy, reproduce, or derive works from this content may result in:

- ⚖️ **Legal action for copyright infringement under British law**
- ⚖️ **Legal action for copyright infringement under Indian law**  
- 🎓 **Academic misconduct proceedings for plagiarization**
- 💰 **Claims for damages and legal costs**

### 📝 Licensing Inquiries

For licensing inquiries or permission requests, please contact the author through the appropriate academic channels.

---

**© 2025 - All Rights Reserved**

---

*Protected under the Copyright, Designs and Patents Act 1988 (UK) and the Copyright Act, 1957 (India)*
