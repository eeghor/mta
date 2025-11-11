# Multi-Touch Attribution (MTA)

A comprehensive Python library for multi-touch attribution modeling in marketing analytics. This library implements various attribution models to help marketers understand the contribution of different touchpoints in the customer journey.

## 🎯 Features

### Attribution Models Implemented

- **First Touch**: 100% credit to the first interaction
- **Last Touch**: 100% credit to the last interaction before conversion
- **Linear**: Equal credit distribution across all touchpoints
- **Position-Based (U-Shaped)**: Customizable weights for first/last touch with remaining credit distributed to middle touches
- **Time Decay**: Higher credit to more recent touchpoints
- **Markov Chain**: Probabilistic model using transition matrices
- **Shapley Value**: Game-theoretic fair allocation based on marginal contributions
- **Shao's Model**: Probabilistic Shapley-equivalent approach
- **Logistic Regression**: Machine learning-based ensemble attribution
- **Additive Hazard**: Survival analysis-based attribution

## 📦 Installation

```bash
pip install mta
```

Or install from source:

```bash
git clone https://github.com/eeghor/mta.git
cd mta
pip install -e .
```

## 🚀 Quick Start

### Basic Usage

```python
from mta import MTA

# Initialize with your data
mta = MTA(data="your_data.csv", allow_loops=False, add_timepoints=True)

# Run a single attribution model
mta.linear(share="proportional", normalize=True)
mta.show()

# Chain multiple models
(mta.linear(share="proportional")
    .time_decay(count_direction="right")
    .markov(sim=False)
    .shapley()
    .show())
```

### Using Configuration

```python
from mta import MTA, MTAConfig

# Create custom configuration
config = MTAConfig(
    allow_loops=False,
    add_timepoints=True,
    sep=" > ",
    normalize_by_default=True
)

mta = MTA(data="data.csv", config=config)
```

### Working with DataFrames

```python
import pandas as pd
from mta import MTA

# Load your data
df = pd.read_csv("customer_journeys.csv")

# Initialize MTA with DataFrame
mta = MTA(data=df, allow_loops=False)

# Run attribution models
mta.first_touch().last_touch().linear().show()
```

## 📊 Data Format

Your input data should be a CSV file or pandas DataFrame with the following columns:

```
path,total_conversions,total_null,exposure_times
alpha > beta > gamma,10,5,2023-01-01 10:00:00 > 2023-01-01 11:00:00 > 2023-01-01 12:00:00
beta > gamma,5,3,2023-01-02 09:00:00 > 2023-01-02 10:00:00
```

**Required Columns:**

- `path`: Customer journey as channel names separated by `>` (or custom separator)
- `total_conversions`: Number of conversions for this path
- `total_null`: Number of non-conversions for this path
- `exposure_times`: Timestamps of channel exposures (optional, can be auto-generated)

## 🎨 Advanced Usage

### Position-Based Attribution with Custom Weights

```python
# Give 30% to first touch, 30% to last touch, 40% distributed to middle
mta.position_based(first_weight=30, last_weight=30, normalize=True)
```

### Time Decay with Direction Control

```python
# Count from left (earliest gets lowest credit)
mta.time_decay(count_direction="left")

# Count from right (latest gets highest credit - more common)
mta.time_decay(count_direction="right")
```

### Markov Chain Attribution

```python
# Analytical calculation (faster)
mta.markov(sim=False, normalize=True)

# Simulation-based (more flexible, handles complex scenarios)
mta.markov(sim=True, normalize=True)
```

### Shapley Value Attribution

```python
# With custom coalition size
mta.shapley(max_coalition_size=3, normalize=True)
```

### Logistic Regression Ensemble

```python
# Custom sampling and iteration parameters
mta.logistic_regression(
    test_size=0.25,
    sample_rows=0.5,
    sample_features=0.5,
    n_iterations=1000,
    normalize=True
)
```

### Export Results

```python
# Compare all models
results_df = mta.compare_models()

# Export to various formats
mta.export_results("attribution_results.csv", format="csv")
mta.export_results("attribution_results.json", format="json")
mta.export_results("attribution_results.xlsx", format="excel")
```

## 📈 Example: Complete Analysis Pipeline

```python
from mta import MTA
import pandas as pd

# Load data
mta = MTA(
    data="customer_journeys.csv",
    allow_loops=False,  # Remove consecutive duplicate channels
    add_timepoints=True  # Auto-generate timestamps if missing
)

# Run all heuristic models
(mta
    .first_touch()
    .last_touch()
    .linear(share="proportional")
    .position_based(first_weight=40, last_weight=40)
    .time_decay(count_direction="right"))

# Run algorithmic models
(mta
    .markov(sim=False)
    .shapley(max_coalition_size=2)
    .shao()
    .logistic_regression(n_iterations=2000)
    .additive_hazard(epochs=20))

# Display and export results
results = mta.compare_models()
mta.export_results("full_attribution_analysis.csv")

# Access specific model results
print(f"Markov Attribution: {mta.attribution['markov']}")
print(f"Shapley Attribution: {mta.attribution['shapley']}")
```

## 🔬 Model Comparison

| Model               | Type             | Strengths                | Use Case                     |
| ------------------- | ---------------- | ------------------------ | ---------------------------- |
| First/Last Touch    | Heuristic        | Simple, fast             | Quick baseline               |
| Linear              | Heuristic        | Fair, interpretable      | Equal value assumption       |
| Position-Based      | Heuristic        | Balances first/last      | Awareness + conversion focus |
| Time Decay          | Heuristic        | Recency-weighted         | When recent matters more     |
| Markov Chain        | Algorithmic      | Considers path structure | Sequential dependency        |
| Shapley Value       | Algorithmic      | Game-theoretic fairness  | Complex interactions         |
| Logistic Regression | Machine Learning | Data-driven              | Large datasets               |
| Additive Hazard     | Statistical      | Time-to-event modeling   | Survival analysis fans       |

## 🛠️ Requirements

- Python >= 3.8
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- arrow >= 1.0.0

## 📝 Citation

If you use this library in your research, please cite:

```bibtex
@software{mta2024,
  author = {Igor Korostil},
  title = {MTA: Multi-Touch Attribution Library},
  year = {2024},
  url = {https://github.com/eeghor/mta}
}
```

## 📚 References

This library implements models and techniques from the following research papers:

1. **Nisar, T. M., & Yeung, M. (2015)**  
   _Purchase Conversions and Attribution Modeling in Online Advertising: An Empirical Investigation_  
   [PDF](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2612997)

2. **Shao, X., & Li, L. (2011)**  
   _Data-driven Multi-touch Attribution Models_  
   Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining  
   [PDF](https://dl.acm.org/doi/10.1145/2020408.2020453)

3. **Dalessandro, B., Perlich, C., Stitelman, O., & Provost, F. (2012)**  
   _Causally Motivated Attribution for Online Advertising_  
   Proceedings of the Sixth International Workshop on Data Mining for Online Advertising  
   [PDF](https://dl.acm.org/doi/10.1145/2351356.2351363)

4. **Cano-Berlanga, S., Giménez-Gómez, J. M., & Vilella, C. (2017)**  
   _Attribution Models and the Cooperative Game Theory_  
   Expert Systems with Applications, 87, 277-286  
   [PDF](https://www.sciencedirect.com/science/article/abs/pii/S0957417417304505)

5. **Ren, K., Fang, Y., Zhang, W., Liu, S., Li, J., Zhang, Y., Yu, Y., & Wang, J. (2018)**  
   _Learning Multi-touch Conversion Attribution with Dual-attention Mechanisms for Online Advertising_  
   Proceedings of the 27th ACM International Conference on Information and Knowledge Management  
   [PDF](https://dl.acm.org/doi/10.1145/3269206.3271676)

6. **Zhang, Y., Wei, Y., & Ren, J. (2014)**  
   _Multi-Touch Attribution in Online Advertising with Survival Theory_  
   2014 IEEE International Conference on Data Mining  
   [PDF](https://ieeexplore.ieee.org/document/7023387)

7. **Geyik, S. C., Saxena, A., & Dasdan, A. (2014)**  
   _Multi-Touch Attribution Based Budget Allocation in Online Advertising_  
   Proceedings of the 8th International Workshop on Data Mining for Online Advertising  
   [PDF](https://dl.acm.org/doi/10.1145/2648584.2648586)

### Model-to-Paper Mapping

- **Linear & Position-Based**: Baseline models referenced across multiple papers
- **Time Decay**: Nisar & Yeung (2015), Zhang et al. (2014)
- **Markov Chain**: Shao & Li (2011), Dalessandro et al. (2012)
- **Shapley Value**: Cano-Berlanga et al. (2017)
- **Logistic Regression**: Dalessandro et al. (2012), Ren et al. (2018)
- **Additive Hazard**: Zhang et al. (2014)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by various academic papers on marketing attribution
- Built with pandas, numpy, and scikit-learn
- Special thanks to the open-source community

## 📧 Contact

Igor Korostil - eeghor@gmail.com

Project Link: [https://github.com/eeghor/mta](https://github.com/eeghor/mta)

## 🐛 Known Issues

- Shapley value computation can be slow for large numbers of channels
- Additive hazard model requires evenly-spaced time points for best results
