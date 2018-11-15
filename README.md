# mca
Multi-Channel Attribution

* Nisar and Yeung (2015) - Purchase Conversions and Attribution Modeling in Online Advertising: An Empirical Investigation [pdf](https://eprints.soton.ac.uk/380534/1/GHLEFMG_FGMJHM_VJ1QM9QF.pdf)
* Shao and Li (2011)  - Data-driven Multi-touch Attribution Models [pdf](http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/data-conv-att.pdf)
* Dalessandro et al (2012) - Causally Motivated Attribution for online Advertising [pdf](https://dstillery.com/wp-content/uploads/2016/07/CAUSALLY-MOTIVATED-ATTRIBUTION.pdf)
* Cano-Berlanga et al  (2017) - Attribution models and the Cooperative Game Theory [pdf](https://www.recercat.cat/bitstream/handle/2072/290758/201702.pdf?sequence=1)
* Ren et al (2018) - Learning Multi-touch Conversion Attribution
with Dual-attention Mechanisms for Online Advertising [pdf](https://arxiv.org/pdf/1808.03737.pdf)

### Terminology

* **Positive User**  is a user who converted
* **Negative User** is a user who did not convert

#### Simple Probabilistic Model by Shao
Suppose there are _p_ channels _x1, x2, …, xp_ and _y_ denotes conversion,. Then we calculate the probability of conversion given exposure to channel _xi_  for all _I=1,2,.., p_. This is a ratio 
(number of users who converted following exposure to channel _xi_)_(number of all users who were exposure to channel /xi_). Next we calculate the probabilities of conversion given exposure to every possible pair of channels (the order of exposure doesn’t matter), i.e. for any two channels _i_ and _j_ calculate the ratio (number of users who converted following exposure to  both channels _xi_ and _xj_) _(number of all users who were exposed to both channels /xi_ and _xj_).

#### Data

We have 10,000 rows that contain paths across 12 channels, 
alpha, beta, delta, epsilon, eta, gamma, iota, kappa, lambda, mi, theta, zeta