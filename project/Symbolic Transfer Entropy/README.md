# Symbolic-Transfer-Entropy

## Overview

There are several techniques (e.g. mutual information, Granger casuality) that can be used to detect *directed exchange of information* across the time-series. Here, we characterize the information flow between different time-series using the **Symbolic Transfer Entropy** (STE) [1]. **STE quantifies the directional flow of information between pairs of time series, <img src="https://render.githubusercontent.com/render/math?math=X"> and <img src="https://render.githubusercontent.com/render/math?math=Y">, both of length <img src="https://render.githubusercontent.com/render/math?math=N">, by first categorizing the signals in a small set of symbols or alphabet according to the pattern trends.** In [2] the authors have shown that the transfer entropy is equivalent to Granger causality for Gaussian processes. An advantage to use STE despite to Granger causality, is to capture non-linear causalities [3]. In the paper [3], the advantages of STE over the Granger causality are analyzed. There are several techniques for estimating TE from observed data in order to apply it to real-world data problems. However, most of them require a large amount of data, and consequently, their results are commonly biased due to small-sample effects, which limits the use of TE in practical data applications. To avoid this problem, we use the robust and computationally fast technique of symbolization to estimate TE, the so-called Symbolic Transfer Entropy.

## Theoretical approach

Let's now procede to convert the whole time-series (it could be useful not measuring the STE over time-series taken as a whole, but over sliding windows of length <img src="https://render.githubusercontent.com/render/math?math=w \ll N">) into a symbolic representation. A time-series is transformed into symbol sequences, for which an *embedding dimension* <img src="https://render.githubusercontent.com/render/math?math=3 \leq m \leq 7"> must be chosen. Let us consider a simple example of how this works. Imagine we have a signal:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=X = \{120, 74, 203, 167, 92, 148, 174, 47\}">
</p>

(let us ignore sliding windows <img src="https://render.githubusercontent.com/render/math?math=w"> by now). We shall transform this series into symbol series. For simplicity, let us suppose that the embedding dimension <img src="https://render.githubusercontent.com/render/math?math=m = 3">. This quantity determines the amount of symbols that can possibly exist, more precisely <img src="https://render.githubusercontent.com/render/math?math=m!">. See in the figure as an illustration of the possible symbols that can be obtained.

<p align="center"> 
<img src="./images/STE.png" width="400">
</p>

The first step to transform <img src="https://render.githubusercontent.com/render/math?math=X"> into symbol sequences is to sort their subchains of length <img src="https://render.githubusercontent.com/render/math?math=m"> in increasing order. So, we take the first three elements of <img src="https://render.githubusercontent.com/render/math?math=X"> and sort them, which leaves us with <img src="https://render.githubusercontent.com/render/math?math=\{74, 120, 203\}">. We have kept tract of these values' indices, such that the sequence now looks like <img src="https://render.githubusercontent.com/render/math?math=\{2, 1, 3\}">. This first subchain maps to symbol <img src="https://render.githubusercontent.com/render/math?math=D">. From this scheme, we just need to advance one value at a time: the next subchain to consider is <img src="https://render.githubusercontent.com/render/math?math=\{74, 203, 167\}">. Its sorted version is <img src="https://render.githubusercontent.com/render/math?math=\{74, 167, 203\}"> which corresponds to <img src="https://render.githubusercontent.com/render/math?math=\{1, 3, 2\}">, and maps to <img src="https://render.githubusercontent.com/render/math?math=B">. And so on, until to achieve a symbol sequence <img src="https://render.githubusercontent.com/render/math?math=\hat{X} = \{D, B, F, E, A, C\}">. With a similar procedure, the series <img src="https://render.githubusercontent.com/render/math?math=Y"> is transormed into <img src="https://render.githubusercontent.com/render/math?math=\hat{Y}">. In our implementation, we consider integer numbers as symbols, for example the following permutations derived by an embedding dimension <img src="https://render.githubusercontent.com/render/math?math=m = 3"> are associated to: <img src="https://render.githubusercontent.com/render/math?math=\{(1, 2, 3): 0, (1, 3, 2): 1, (2, 1, 3): 2, (2, 3, 1): 3, (3, 1, 2): 4, (3, 2, 1): 5\}">. A way to interpret the meaning of <img src="https://render.githubusercontent.com/render/math?math=m"> is to think of it as the amount of *expressiveness* it allows to the original series. That is, if <img src="https://render.githubusercontent.com/render/math?math=m"> is low, a rich signal (one with many changes in it) is reduced to a small amount of possible symbols. An important feature of symbolic approaches is that they discount the relative magnitude of the time series; this is important in our case because different geographical units (our provinces) can differ largely in population density or other parameters. **Moreover, STE can successfully analyze time-series which may be short and/or non-stationary**. A drawback is that it considers only the order pattern of the time-series hence the information contained in the magnitude of the differences between amplitude values may not be taken into account. For example, <img src="https://render.githubusercontent.com/render/math?math=[1, 100, 2]"> and <img src="https://render.githubusercontent.com/render/math?math=[1, 3, 2]"> have the same permutation pattern <img src="https://render.githubusercontent.com/render/math?math=[0,2,1]">, but they vary greatly in size and tendency.

Given these symbol sequences, the transfer entropy between a pair of signals can be computed.

Let <img src="https://render.githubusercontent.com/render/math?math=\hat{x}_i = \hat{x}(i)"> and <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i = \hat{y}(i)">, <img src="https://render.githubusercontent.com/render/math?math=i = 1, ..., N">, denote sequences of observations from systems <img src="https://render.githubusercontent.com/render/math?math=\hat{X}"> and <img src="https://render.githubusercontent.com/render/math?math=\hat{Y}"> (symbolized time-series). *Transfer entropy* [4] incorporates time dependence by relating previous samples <img src="https://render.githubusercontent.com/render/math?math=\hat{x}_i"> and <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> to predict the next value <img src="https://render.githubusercontent.com/render/math?math=\hat{x}_{i %2B 1}">, and quantifies the deviation from the Markov property, <img src="https://render.githubusercontent.com/render/math?math=p(\hat{x}_{i %2B 1}|\hat{x}_i, \hat{y}_i) = p(\hat{x}_{i %2B 1}|\hat{x}_i)">, where <img src="https://render.githubusercontent.com/render/math?math=p"> denotes the transition probability density. If there is no deviation from the Markov property, <img src="https://render.githubusercontent.com/render/math?math=\hat{Y}"> has no influence on <img src="https://render.githubusercontent.com/render/math?math=\hat{X}">. **Transfer entropy, which is formulated as Kullback-Leibler entropy between <img src="https://render.githubusercontent.com/render/math?math=p(\hat{x}_{i %2B 1}|\hat{x}_i, \hat{y}_i)"> and <img src="https://render.githubusercontent.com/render/math?math=p(\hat{x}_{i %2B 1}|\hat{x}_i)">, quantifies the incorrectness of this assumption**, and is explicitly non-symmetric under the exchange of <img src="https://render.githubusercontent.com/render/math?math=\hat{x}_i"> and <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i">. *The main convenience of such an information theoretic functional designed to detect causality is that, in principle, it does not assume any particular model for the interaction between the two systems of interest*.

Now, we can obtain the pairwise STE computing the joint and conditional probabilities of the sequence indices from the relative frequency of symbols in each sequence, <img src="https://render.githubusercontent.com/render/math?math=\hat{X}"> and <img src="https://render.githubusercontent.com/render/math?math=\hat{Y}">, using the Shannon transfer entropy:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=T_{XY} = \sum p(\hat{y}_{i %2B 1}, \hat{y}_{i}, \hat{x}_{i}) log_2(\frac{p(\hat{y}_{i %2B 1}| \hat{y}_{i}, \hat{x}_{i})}{p(\hat{y}_{i %2B 1}| \hat{y}_{i})}) = \sum p(\hat{y}_{i %2B 1}, \hat{y}_{i}, \hat{x}_{i}) log_2(\frac{p(\hat{y}_{i %2B 1}, \hat{y}_{i}, \hat{x}_{i}) p(\hat{y}_i)}{p(\hat{y}_i, \hat{x}_i) p(\hat{y}_{i %2B 1}, \hat{y}_{i})})">
</p>

where the sum runs over each unique state of the sequences. <img src="https://render.githubusercontent.com/render/math?math=T_{XY}"> (or better <img src="https://render.githubusercontent.com/render/math?math=T_{X \rightarrow Y}">) **measures the information flow from <img src="https://render.githubusercontent.com/render/math?math=X"> to <img src="https://render.githubusercontent.com/render/math?math=Y">**. It is non-negative, and any information transfer between the two variables results in <img src="https://render.githubusercontent.com/render/math?math=0 \leq {T}_{XY} < \infty">. If the state <img src="https://render.githubusercontent.com/render/math?math=\hat{x}_{i}"> has no influence on the transition probabilities from <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_{i}"> to <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_{i %2B 1}">, or if the two time series are completely synchronized, then <img src="https://render.githubusercontent.com/render/math?math={T}_{XY} = 0">. The logarithm has base 2, so that the TE is measured in bits. For example if <img src="https://render.githubusercontent.com/render/math?math=T_{X, Y} = 0.624"> means that the history of the <img src="https://render.githubusercontent.com/render/math?math=X"> process has <img src="https://render.githubusercontent.com/render/math?math=0.624"> bits of additional information for predicting the next value of <img src="https://render.githubusercontent.com/render/math?math=Y">. (i.e., it provides information about the future of <img src="https://render.githubusercontent.com/render/math?math=Y">, in addition to what we know from the history of <img src="https://render.githubusercontent.com/render/math?math=Y">). Since it is non-zero, you can conclude that <img src="https://render.githubusercontent.com/render/math?math=X"> influences <img src="https://render.githubusercontent.com/render/math?math=Y"> in some way. There are actually two equations for the transfer entropy, because it has an inherent asymmetry in it:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=T_{YX} = \sum p(\hat{x}_{i %2B 1}, \hat{x}_{i}, \hat{y}_{i}) log_2(\frac{p(\hat{x}_{i %2B 1}, \hat{x}_{i}, \hat{y}_{i}) p(\hat{x}_i)}{p(\hat{x}_i, \hat{y}_i) p(\hat{x}_{i %2B 1}, \hat{x}_{i})})">
</p>

We can obtain the matrix <img src="https://render.githubusercontent.com/render/math?math=\{T_{XY}\}">, which contains pairwise information about how each component in the system controls (or is controlled by) the others. The matrix <img src="https://render.githubusercontent.com/render/math?math=\{T_{XY}\}"> is asymmetric. 

The transfer entropy in this "discrete" case (or simply into its symbolized version) can be derived using conditional Shannon entropies by expanding the logarithm:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=T_{XY} = H(\hat{y}_{i %2B 1}|\hat{y}_{i}) - H(\hat{y}_{i %2B 1}|\hat{y}_{i}, \hat{x}_{i})">
</p>

where <img src="https://render.githubusercontent.com/render/math?math=H(\hat{y}_{i %2B 1}|\hat{y}_{i}) = -\sum p(\hat{y}_{i %2B 1}, \hat{y}_{i}) log_2(p(\hat{y}_{i %2B 1}| \hat{y}_{i}))"> is the entropy rate (a conditional Shannon entropy) and similarly <img src="https://render.githubusercontent.com/render/math?math=H(\hat{y}_{i %2B 1}|\hat{y}_{i}, \hat{x}_{i})"> a generalised entropy rate. The entropy rate <img src="https://render.githubusercontent.com/render/math?math=H(\hat{y}_{i %2B 1}|\hat{y}_{i})"> accounts for the average number of bits needed to encode one additional state of the system if the previous states is known, while the entropy rate <img src="https://render.githubusercontent.com/render/math?math=H(\hat{y}_{i %2B 1}|\hat{y}_{i}, \hat{x}_{i})"> is the entropy rate capturing the average number of bits required to represent the value of the next destination’s state if source state is included in addition. Since one can always write [5]:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=H(\hat{y}_{i %2B 1}|\hat{y}_{i}) = -\sum p(\hat{y}_{i %2B 1}, \hat{y}_{i}) log_2(p(\hat{y}_{i %2B 1}| \hat{y}_{i})) = -\sum p(\hat{y}_{i %2B 1}, \hat{y}_{i}, \hat{x}_{i}) log_2(p(\hat{y}_{i %2B 1}| \hat{y}_{i}))">
</p>

it is easy to see that the entropy rate <img src="https://render.githubusercontent.com/render/math?math=H(\hat{y}_{i %2B 1}|\hat{y}_{i})"> is equivalent to the rate <img src="https://render.githubusercontent.com/render/math?math=H(\hat{y}_{i %2B 1}|\hat{y}_{i}, \hat{x}_{i})"> when the next state of destination is independent of the source:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=p(\hat{y}_{i %2B 1}|\hat{y}_{i}, \hat{x}_{i}) = p(\hat{y}_{i %2B 1}|\hat{y}_{i})">
</p>

Thus, in this case the transfer entropy reduces to zero.

We can also define a generalized Markov property <img src="https://render.githubusercontent.com/render/math?math=p(\hat{x}_{i %2B 1}|\mathbf{\hat{x}_i^{(k_x)}}, \mathbf{\hat{y}_i^{(k_y)}}) = p(\hat{x}_{i %2B 1}|\mathbf{\hat{x}_i^{(k_x)}})"> relying on the Kullback-Leibler distance, where <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\hat{x}_i^{(k_x)}} = (\hat{x}_i, \hat{x}_{i - 1}, ..., \hat{x}_{i-(k_x-1)})">:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=T_{YX} = \sum p(\hat{x}_{i %2B 1}, \mathbf{\hat{x}_{i}^{(k_x)}}, \mathbf{\hat{y}_{i}^{(k_y)}}) log_2(\frac{p(\hat{x}_{i %2B 1}, \mathbf{\hat{x}_{i}^{(k_x)}}, \mathbf{\hat{y}_{i}^{(k_y)}}) p(\mathbf{\hat{x}_{i}^{(k_x)}})}{p(\mathbf{\hat{x}_{i}^{(k_x)}}, \mathbf{\hat{y}_{i}^{(k_y)}}) p(\hat{x}_{i %2B 1}, \mathbf{\hat{x}_{i}^{(k_x)}})})">
</p>

Also in this case, we can define the transfer entropy as the difference of these two conditional Shannon entropies:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=T_{YX} = H(\hat{x}_{i %2B 1}|\mathbf{\hat{x}_{i}^{(k_x)}}) - H(\hat{x}_{i %2B 1}|\mathbf{\hat{x}_{i}^{(k_x)}}, \mathbf{\hat{y}_{i}^{(k_x)}})">
</p>

In addition to these definitions, it is necessary to set a horizon prediction to the variable <img src="https://render.githubusercontent.com/render/math?math=X"> (or <img src="https://render.githubusercontent.com/render/math?math=Y"> depending on the case). This horizon indicates how far in the future of <img src="https://render.githubusercontent.com/render/math?math=X"> will be analyzed and is symbolized by the parameter <img src="https://render.githubusercontent.com/render/math?math=h">. Let <img src="https://render.githubusercontent.com/render/math?math=x_{i %2B h}"> denote the value of <img src="https://render.githubusercontent.com/render/math?math=X"> at time instant <img src="https://render.githubusercontent.com/render/math?math=i %2B h">, that is, <img src="https://render.githubusercontent.com/render/math?math=h"> steps in the future from <img src="https://render.githubusercontent.com/render/math?math=i">, and <img src="https://render.githubusercontent.com/render/math?math=h"> is referred to as the prediction horizon. That is if <img src="https://render.githubusercontent.com/render/math?math=h = 1"> the method will always verify only one sample ahead of the present, over the whole time analysis. Meaning that, assuming <img src="https://render.githubusercontent.com/render/math?math=i"> instants as the time reference. The method will check whether or not the past of <img src="https://render.githubusercontent.com/render/math?math=Y"> is influencing the behavior of the <img src="https://render.githubusercontent.com/render/math?math=X"> variable in time instant <img src="https://render.githubusercontent.com/render/math?math=i %2B 1">.

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=T_{YX} = \sum p(\hat{x}_{i %2B h}, \mathbf{\hat{x}_{i}^{(k_x)}}, \mathbf{\hat{y}_{i}^{(k_y)}}) log_2(\frac{p(\hat{x}_{i %2B h}, \mathbf{\hat{x}_{i}^{(k_x)}}, \mathbf{\hat{y}_{i}^{(k_y)}}) p(\mathbf{\hat{x}_{i}^{(k_x)}})}{p(\mathbf{\hat{x}_{i}^{(k_x)}}, \mathbf{\hat{y}_{i}^{(k_y)}}) p(\hat{x}_{i %2B h}, \mathbf{\hat{x}_{i}^{(k_x)}})})">
</p>

where for <img src="https://render.githubusercontent.com/render/math?math=h = 1"> and <img src="https://render.githubusercontent.com/render/math?math=k_x = k_y = 1"> is equivalent to the standard form of the transfer entropy based on Markov property not generalized.

**The dominant direction of the information flow can be inferred by calculating the difference between <img src="https://render.githubusercontent.com/render/math?math=T_{YX}"> and <img src="https://render.githubusercontent.com/render/math?math=T_{XY}">**. It is convenient to define the directionality index <img src="https://render.githubusercontent.com/render/math?math=T_{XY}^{S} = T_{YX} - T_{XY}">, which measures the balance of information flow in both directions. **This index quantifies the dominant direction of information flow and is expected to have positive values for undirectional couplings with <img src="https://render.githubusercontent.com/render/math?math=x"> (x-axis) as driver and negative values if <img src="https://render.githubusercontent.com/render/math?math=y"> (y-axis) is driving <img src="https://render.githubusercontent.com/render/math?math=x">**. For symmetric bidirectional couplings, we expect <img src="https://render.githubusercontent.com/render/math?math=T_{XY}^{S}"> to be null. In this case the matrix is symmetric, for this reason I show only a side of the matrix.

### Searching for the best window size of the historical <img src="https://render.githubusercontent.com/render/math?math=X"> (temporal lags) used for the future <img src="https://render.githubusercontent.com/render/math?math=Y"> prediction

Using the generalized formula of the transfer entropy, we can play with <img src="https://render.githubusercontent.com/render/math?math=k_x"> i.e., the window size of the historical predictor <img src="https://render.githubusercontent.com/render/math?math=X"> used for the future target <img src="https://render.githubusercontent.com/render/math?math=Y"> prediction, in order to find to the optimal temporal lags for the predictor. A solution could be the estimation of the optimal <img src="https://render.githubusercontent.com/render/math?math=k_x"> as the minimum positive integer above which the change rate of the TE from <img src="https://render.githubusercontent.com/render/math?math=X"> to <img src="https://render.githubusercontent.com/render/math?math=Y"> decreases significantly [6]. More precisely, we first determine the optimal <img src="https://render.githubusercontent.com/render/math?math=k_y"> as the minimum nonnegative integer above which the change rate of the entropy rate <img src="https://render.githubusercontent.com/render/math?math=H(\hat{y}_{i %2B 1}|\mathbf{\hat{y}_{i}^{(k_y)}})"> decreases significantly.

Our approah is analogous to 

## References

[1]. "Symbolic transfer entropy", M. Staniek and K. Lehnertz, 2008.

[2]. "Equivalence of granger causality and transfer entropy: A generalization", K. Schindlerova, 2011.

[3]. "Transfer entropy as a variable selection methodology of cryptocurrencies in the framework of a high dimensional predictive model", García-Medina, González Farías, 2020.

[4]. "Measuring information transfer", T.Schreiber, 2000.

[5]. "On Thermodynamic Interpretation of Transfer Entropy", Mikhail Prokopenko, Joseph T. Lizier and Don C. Price, 2013.

[6]. "Direct Causality Detection via the Transfer Entropy Approach ", Ping Duan, Fan Yang, 2013.


