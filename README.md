# mta
Multi-Touch Attribution. Find out which channels contribute most to user conversion. 

### Models

This package contains implementations the following Multi-Touch Attribution models:

* Shapley 
* Markov
* So-called Simple Probabilistic Model by Shao and Li
* Bagged Logistic Regression by Shao and Li

In addition, some popular heuristic “models” are included, specifically

* First Touch
* Linear
* Last Touch
* Time Decay

### Included Data

The package comes with the same test data set as an R package called [ChannelAttribution](https://cran.r-project.org/web/packages/ChannelAttribution/ChannelAttribution.pdf)  - there are 10,000 rows containing customer journeys across 12 channels: alpha, beta, delta, epsilon, eta, gamma, iota, kappa, lambda, mi, theta and zeta.

![](README/data_snippet.png)

### References

* Nisar and Yeung (2015) - Purchase Conversions and Attribution Modeling in Online Advertising: An Empirical Investigation [pdf](https://eprints.soton.ac.uk/380534/1/GHLEFMG_FGMJHM_VJ1QM9QF.pdf)
* Shao and Li (2011)  - Data-driven Multi-touch Attribution Models [pdf](http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/data-conv-att.pdf)
* Dalessandro et al (2012) - Causally Motivated Attribution for online Advertising [pdf](https://dstillery.com/wp-content/uploads/2016/07/CAUSALLY-MOTIVATED-ATTRIBUTION.pdf)
* Cano-Berlanga et al  (2017) - Attribution models and the Cooperative Game Theory [pdf](https://www.recercat.cat/bitstream/handle/2072/290758/201702.pdf?sequence=1)
* Ren et al (2018) - Learning Multi-touch Conversion Attribution
with Dual-attention Mechanisms for Online Advertising [pdf](https://arxiv.org/pdf/1808.03737.pdf)
* Zhang et al (2014)  - Multi-Touch Attribution in Online Advertising with Survival Theory [pdf](http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/attr-survival.pdf)