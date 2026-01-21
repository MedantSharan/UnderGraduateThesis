# Four-Player Mancala Reinforcement Learning Framework

A comprehensive framework for exploring multi-agent reinforcement learning in a novel four-player Mancala variant using Q-learning with experience replay.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Key Features](#key-features)
- [Research Contributions](#research-contributions)
- [Game Rules](#game-rules)
- [Experimental Framework](#experimental-framework)
- [Algorithm Details](#algorithm-details)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
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

The complete implementation is provided in a self-contained Jupyter notebook that can be executed independently. All methodological details, experimental design, and theoretical foundations are documented in the accompanying research report.

## Getting Started

### 📓 Primary Implementation: MancalaFinal.ipynb

The **MancalaFinal.ipynb** notebook contains the complete, self-contained implementation of this project. This is the recommended way to explore and run the framework.

**Features of MancalaFinal.ipynb:**
- ✅ Fully self-contained code requiring no external scripts
- ✅ Complete implementation of all experiments
- ✅ Integrated visualization and analysis tools
- ✅ Step-by-step execution with detailed comments
- ✅ Ready to run in Google Colab or locally

### 📄 Methodology Reference

**For detailed methodological information, experimental design rationale, and theoretical background, please refer to the accompanying PDF report.** The report provides:

- Comprehensive literature review and theoretical foundations
- Detailed explanation of algorithmic choices and hyperparameter selection
- Complete experimental methodology and validation procedures
- In-depth analysis of results and emergent behaviors
- Discussion of limitations and future research directions

### 🚀 Quick Start Options

#### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the **MancalaFinal.ipynb** notebook
3. Run cells sequentially from top to bottom
4. All dependencies will be installed automatically

> 💡 **For faster preliminary results**: Locate training parameters and reduce episode counts from 3000 to 300-500.

#### Option 2: Local Jupyter Environment

```bash
# Clone or download the repository
# Navigate to the project directory
jupyter notebook MancalaFinal.ipynb
```

### System Requirements

- **Google Colab**: Web browser and Google account (no local setup required)
- **Local Execution**: 
  - Python 3.7 or higher
  - Jupyter Notebook or JupyterLab
  - 8GB RAM (recommended for full experiments)
  - 4GB available disk space

## Key Features

- 🎮 **Flexible Game Environment**: Customizable board configurations with variable pit counts and stone distributions
- 🤖 **Multi-Agent Support**: Independent Q-learning agents with experience replay mechanisms
- 📊 **Comprehensive Analysis**: Parameter studies, positional analysis, and emergent behavior investigation
- 📈 **Statistical Validation**: Rigorous statistical testing with confidence intervals
- 🎨 **Visualization Pipeline**: Publication-quality plots and heatmaps for result analysis
- 📓 **Self-Contained Implementation**: Complete framework in a single executable notebook

## Research Contributions

### Strategic Sweet Spots

The research identifies optimal board configurations where reinforcement learning agents significantly outperform theoretical expectations:

| Configuration | Win Rate | Baseline | Improvement |
|---------------|----------|----------|-------------|
| 6 pits, 2 stones | **50.4%** | 25% random | +101.6% |
| Adjacent positioning (Players 1&2) | **62%** combined | 50% theoretical | +24% |

### Emergent Behaviors

- 🤝 **Pseudo-cooperative dynamics**: Independently trained agents develop complementary strategies without explicit coordination
- 🎯 **Positional advantages**: Early turn positions provide consistent strategic benefits
- 📉 **Non-linear complexity effects**: Performance varies non-monotonically with game parameters
- 🧩 **Strategic pattern recognition**: Agents learn to identify and exploit board states for optimal stone capture

## Game Rules

### Four-Player Mancala Mechanics

```
Player 4    [R]  P3  P2  P1  [R]    Player 1
           P8  P7  P6  P5  P4  P3
           P1  P2  P3  P4  P5  P6
Player 3    [R]  P1  P2  P3  [R]    Player 2
```

**Setup**: Each player controls N pits initially containing M stones, plus one reservoir (scoring pit)

**Turn Sequence**: Players take turns counterclockwise, distributing stones one per pit

**Special Rules**:
- ❌ Skip opponents' reservoirs when distributing stones
- ➕ Land in your reservoir → earn an extra turn
- 🎯 Land in an empty pit on your side → capture stones from the opposite player's corresponding pit

**Victory Condition**: Game ends when any player's pits become empty; player with most stones in reservoir wins

### Key Modifications from Traditional Mancala

| Feature | Traditional (2-Player) | Four-Player Variant |
|---------|----------------------|-------------------|
| **Capture Rule** | Capture from all opponents | Only from directly opposite player |
| **Game End** | One player's side empty | Any player's side empty |
| **Dynamics** | Two-player zero-sum | Four-player positional strategy |
| **Stone Distribution** | Continuous around board | Skip opposing reservoirs |

## Experimental Framework

### Parameter Studies

Systematic exploration of board configurations:

- **Pit counts**: 3, 4, 6, 8 pits per player
- **Stone counts**: 2, 3, 4, 6 initial stones per pit
- **Total configurations**: 16 distinct environments tested
- **Episodes per configuration**: 3,000 training episodes

### Multi-Agent Analysis Scenarios

| Scenario | Agent Configuration | Purpose |
|----------|-------------------|---------|
| Single agent | 1 RL agent vs 3 random players | Baseline performance establishment |
| Dual agents | 2 RL agents vs 2 random players | Cooperative/competitive dynamics analysis |
| Positional studies | Adjacent vs opposite agent placement | Strategic position advantage quantification |

### Evaluation Metrics

- Win rate percentage (primary metric)
- Average reward accumulation
- Stone capture efficiency
- Extra turn exploitation rate
- Convergence speed analysis

## Algorithm Details

### Q-Learning Implementation

The framework employs tabular Q-learning with experience replay for stable convergence:

```python
# Core hyperparameters
learning_rate = 0.1              # α - Learning rate
discount_factor = 0.95           # γ - Future reward discount
exploration_strategy = "ε-greedy"  # Exploration policy
epsilon_decay = {
    "initial": 1.0,
    "minimum": 0.01,
    "decay_rate": "exponential"
}

experience_replay = {
    "buffer_size": 10000,        # Memory capacity
    "batch_size": 32,            # Training batch size
    "sampling": "uniform_random"  # Replay sampling strategy
}
```

### State Representation

States are encoded as tuples capturing:
- Stone counts in all pits for all players
- Stone counts in all reservoirs
- Current player turn indicator

### Action Space

Valid actions correspond to selecting non-empty pits on the current player's side.

### Reward Structure

| Event | Reward Value | Rationale |
|-------|--------------|-----------|
| Stone to reservoir | +0.5 per stone | Encourages incremental progress |
| Stone capture | +2.0 per stone | Rewards strategic captures |
| Extra turn gained | +3.0 bonus | Incentivizes turn extension |
| Game win | Variable (rank-based) | Final outcome reinforcement |

**For complete algorithmic details and theoretical justification, refer to the methodology section in the accompanying PDF report.**

## Results Summary

### 🏆 Performance Highlights

- **Best single-agent win rate**: 50.4% (6 pits, 2 stones configuration)
- **Best dual-agent combined rate**: 62% (adjacent positioning, 6 pits, 2 stones)  
- **Statistical significance**: All major findings validated with p < 0.05
- **Training convergence**: Stable performance achieved within 3,000 episodes
- **Theoretical baseline**: 25% random agent win rate in four-player setting

### 💡 Key Research Insights

1. **Non-monotonic complexity relationship**: Intermediate complexity configurations (6 pits, 2 stones) create optimal learning conditions
2. **Positional advantages**: Early turn order provides measurable strategic benefits (~5-10% win rate improvement)
3. **Emergent cooperation**: Independently trained agents develop mutually beneficial strategies without explicit communication
4. **Configuration sensitivity**: Small parameter changes create dramatic differences in learning outcomes
5. **Sparse reward challenges**: Experience replay proves essential for stable convergence

### 📊 Statistical Validation

All results include:
- 95% confidence intervals
- Binomial tests for win rate significance
- T-tests for score difference validation
- Multiple testing corrections where appropriate

**For detailed statistical analysis and comprehensive results discussion, consult the research report PDF.**

## Project Structure

```
📁 UnderGraduateThesis/
├── 📓 MancalaFinal.ipynb       # PRIMARY: Complete self-contained implementation
├── 📄 Report.pdf              # Comprehensive methodology and analysis document
├── 📄 README.md               # This file

```

## Usage Guide

### Running the Complete Framework

1. **Open MancalaFinal.ipynb** in Jupyter or Google Colab
2. **Execute cells sequentially** from top to bottom
3. **Monitor training progress** through integrated progress bars
4. **View results** in automatically generated plots and tables

### Customizing Experiments

The notebook is structured with clearly marked sections:

```python
# Example: Modify training parameters
TRAINING_EPISODES = 3000  # Reduce for faster testing
LEARNING_RATE = 0.1       # Adjust learning dynamics
BUFFER_SIZE = 10000       # Experience replay capacity

# Example: Test custom board configuration
custom_config = {
    'pits': 5,
    'stones': 3,
    'label': 'Custom_5x3'
}
```

### Experimental Workflows

The notebook supports three primary experimental workflows:

1. **Parameter Study**: Systematic evaluation across all pit/stone combinations
2. **Positional Analysis**: Agent placement effect quantification
3. **Dual Agent Study**: Multi-agent interaction exploration

**Each workflow is documented within the notebook. For theoretical background and design rationale, refer to the methodology section of the PDF report.**

## Dependencies

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.19.0 | Numerical computing and array operations |
| `matplotlib` | ≥3.3.0 | Plotting and visualization |
| `gymnasium` | ≥0.26.0 | RL environment framework |
| `pandas` | ≥1.1.0 | Data manipulation and analysis |
| `scipy` | ≥1.5.0 | Statistical testing and analysis |
| `tqdm` | ≥4.50.0 | Progress bar visualization |
| `seaborn` | ≥0.11.0 | Statistical data visualization |

### Installation

The MancalaFinal.ipynb notebook includes automatic dependency installation. For manual setup:

```bash
pip install numpy matplotlib gymnasium pandas scipy tqdm seaborn
```

### Compatibility

- **Python**: 3.7, 3.8, 3.9, 3.10, 3.11
- **Operating Systems**: Windows, macOS, Linux
- **Jupyter**: Notebook 6.x or JupyterLab 3.x

## Performance Notes

### Computational Requirements

- ⏱️ **Training time**: ~5-10 minutes per configuration (3,000 episodes on standard hardware)
- 💾 **Memory usage**: Sparse Q-table representation manages memory efficiently (~500MB peak)
- 📈 **Scalability**: Framework tested up to 8 pits, 6 stones (state space ~10^16)
- 🖥️ **Hardware**: CPU-only implementation (no GPU required)

### Optimization Tips

- Reduce training episodes for preliminary testing (300-500 episodes)
- Use smaller board configurations (3-4 pits) for rapid prototyping
- Enable progress bars to monitor convergence
- Batch experiment execution for efficiency

## Limitations and Future Work

### ⚠️ Current Limitations

1. **Fixed training duration**: 3,000 episodes may be suboptimal for some configurations
2. **Random baseline evaluation**: Limited comparison against sophisticated opponents
3. **Independent training**: Agents trained separately, not simultaneously
4. **First-player advantage**: Potential positional bias requires further investigation
5. **Tabular approach**: State space explosion limits scalability to larger boards

### 🔮 Future Research Directions

#### Algorithmic Enhancements
- **Deep Q-Networks (DQN)**: Neural network function approximation for larger state spaces
- **Policy gradient methods**: Actor-critic architectures for continuous improvement
- **Monte Carlo Tree Search**: Hybrid planning-learning approaches

#### Multi-Agent Extensions
- **Simultaneous training**: Co-evolutionary agent development
- **Communication protocols**: Explicit agent coordination mechanisms
- **Competitive training**: Agents trained against evolving opponents

#### Comparative Studies
- **Human player analysis**: Strategy comparison with expert human play
- **Alternative algorithms**: Comparative evaluation (SARSA, PPO, A3C)
- **Transfer learning**: Strategy generalization across board configurations

#### Theoretical Analysis
- **Positional advantage quantification**: Mathematical formalization of turn order effects
- **Convergence guarantees**: Theoretical bounds on learning performance
- **Optimal strategy characterization**: Game-theoretic analysis

**For detailed discussion of limitations and future work, see the research report PDF.**

## Statistical Validation

### Methodology

All reported results undergo rigorous statistical validation:

- ✅ **Confidence intervals**: 95% confidence bounds for all performance metrics
- 📊 **Hypothesis testing**: Binomial tests for win rates, t-tests for score differences
- 📏 **Effect sizes**: Practical significance assessment beyond statistical significance
- 🔧 **Multiple testing corrections**: Bonferroni adjustments for family-wise error control
- 🎲 **Reproducibility**: Random seed control for experiment replication

### Significance Thresholds

- Statistical significance: p < 0.05
- Practical significance: Effect size > 0.3 (Cohen's d)
- Confidence level: 95% for all interval estimates

**Complete statistical procedures and validation protocols are documented in the PDF report.**

## Citation

If you use this framework in your research, please cite:

```bibtex
@techreport{sharan2025mancala,
  title={Exploring Multi-Agent Reinforcement Learning in a Four-Player Mancala Variant},
  author={Sharan, M.},
  institution={King's College London},
  year={2025},
  type={Final Project Report},
  note={Complete implementation available in MancalaFinal.ipynb}
}
```

## License

This project is released under the MIT License. See the LICENSE file for full legal details.

### Usage Terms

- ✅ Academic and educational use
- ✅ Research and experimentation
- ✅ Extension and modification (with attribution)
- ❌ Commercial use without permission
- ❌ Reproduction without citation

## Acknowledgments

- 🎓 **King's College London** Department of Informatics for academic support
- 🔧 **OpenAI Gymnasium** framework developers for the RL environment foundation
- 🎮 **Gary MacLeod** for the original four-player Mancala variant design
- 📚 **Reinforcement Learning Community** for algorithmic insights and best practices

## Contact

For questions about the implementation, methodology, or research findings:

- **Technical inquiries**: Refer to code documentation in MancalaFinal.ipynb
- **Methodological questions**: Consult the comprehensive PDF report
- **Academic correspondence**: Contact through appropriate institutional channels

---

## Copyright Disclaimer

### ⚖️ IMPORTANT LEGAL NOTICE

**This work and all associated materials are protected under British and Indian copyright law.** The content, methodology, code implementation, research findings, documentation, and the MancalaFinal.ipynb notebook are the intellectual property of the author(s).

### 🚨 Unauthorized Usage Warning

**Unauthorized copying, reproduction, distribution, or use of any portion of this work—including but not limited to the MancalaFinal.ipynb notebook, research report, code, methodology, or findings—without explicit written permission constitutes copyright infringement and plagiarism.**

Any attempt to copy, reproduce, or derive works from this content may result in:

- ⚖️ **Legal action for copyright infringement** under the Copyright, Designs and Patents Act 1988 (UK)
- ⚖️ **Legal action for copyright infringement** under the Copyright Act, 1957 (India)  
- 🎓 **Academic misconduct proceedings** for plagiarization at institutional and professional levels
- 💰 **Claims for damages**, legal costs, and injunctive relief
- 📋 **Disciplinary action** through academic integrity committees

### 📝 Protected Components

All of the following are protected intellectual property:

- MancalaFinal.ipynb notebook and all code therein
- Research methodology and experimental design
- Algorithm implementations and optimizations
- Documentation and written content
- Generated results, visualizations, and analysis
- Project structure and organization

### 📧 Licensing Inquiries

For licensing inquiries, permission requests, or collaboration proposals, please contact the author through official academic channels.

### ✅ Permitted Use

With proper citation:
- Educational reference and learning
- Academic study and research
- Critical analysis and review

**All other uses require explicit written permission.**

---

**© 2026 - All Rights Reserved**

*Protected under the Copyright, Designs and Patents Act 1988 (UK) and the Copyright Act, 1957 (India)*

---

## Final Notes

**This framework is designed for research and educational purposes.** The implementation provides a robust foundation for exploring multi-agent reinforcement learning in traditional board games and can be extended for investigating other strategic multiplayer environments.

**To begin your exploration:**
1. Open **MancalaFinal.ipynb** 
2. Review the **PDF report** for theoretical background
3. Execute the notebook cells sequentially
4. Experiment with custom configurations

**Happy researching! 🎮🤖**
