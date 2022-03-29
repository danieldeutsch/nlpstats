# nlpstats.correlations
The `nlpstats.correlations` module provides tools for meta-evaluating metrics.

The quality of a metric is quantified by calculating the correlation between its scores and human scores for the outputs of a set of systems on a set of inputs.
The correlation can be calculated in several different ways, each of which is a function of two score matrices.

Let {math}`X \in \mathbb{R}^{m \times n_1}` and {math}`Z \in \mathbb{R}^{m \times n_2}` be the metric and human score matrices in which {math}`x_i^{j_1}` and {math}`z_i^{j_2}` are the respective scores on the output from the {math}`i` th system on the {math}`j_1` th and {math}`j_2` th inputs.
Note that the rows of {math}`X` and {math}`Z` correspond to each other, but this is not necessarily true for their columns.
{math}`X` and {math}`Z` can be used to calculate three different correlations as follows.

The *system-level correlation* quantifies the extent to which the metric scores systems similarly to humans.
It is defined as:

```{math}
r_{\textrm{sys}} = r(\left\{\left(\bar{x}_1, \bar{z}_1\right), \dots, \left(\bar{x}_m, \bar{z}_m\right)\right\})
```

where {math}`r(\cdot)` is some function which calculates the correlation between the paired observations and {math}`\bar{x}_i` and {math}`\bar{z}_i` are the metric and human scores for system {math}`i`:

```{math}
\bar{x}_i = \frac{1}{n_1} \sum_j^{n_1} x_i^j

\bar{z}_i = \frac{1}{n_2} \sum_j^{n_2} z_i^j
```

The *input-level correlation* (also called the *summary-level correlation* in the summarization literature) quantifies how similarly the metric and humans score different outputs for the *same* input.
The input-level correlation requires the columns of {math}`X` and {math}`Z` to be paired (i.e., the {math}`j` th column of {math}`X` and {math}`Z` both correspond to the same input and {math}`n_1 = n_2`).
It is defined as:

```{math}
r_{\textrm{inp}} = \frac{1}{n} \sum_j^n r\left(\left\{(x_1^j, z_2^j), \dots, (x_m^j, z_m^j)\right\}\right)
```

where {math}`n = n_1 = n_2`.
This function calculates the average correlation between the columns of {math}`X` and {math}`Z`

Finally the *global-level correlation* directly calculates the correlation between all of the {math}`x_i^{j}` and {math}`z_i^{j}` pairs.
It also requires the columns to be paired:

```{math}
r_{\textrm{glo}} = r\left(\left\{(x_1^1, z_1^1), \dots, (x_m^1, z_m^1), \dots (x_m^n, z_m^n)\right\}\right)
```

This module provides methods for:

- [calculating these correlations](#calculating-correlations)
- [estimating confidence intervals for correlations](#confidence-intervals)
- [statistical testing the difference between correlations](#hypothesis-testing)

## Calculating Correlations

```{eval-rst}
.. autofunction:: nlpstats.correlations.correlations.correlate

.. autofunction:: nlpstats.correlations.correlations.system_level

.. autofunction:: nlpstats.correlations.correlations.input_level

.. autofunction:: nlpstats.correlations.correlations.global_level
```

## Confidence Intervals

This module contains two methods for calculating confidence intervals: bootstrapping and the Fisher transformation.
The documentation for both functions is described next.

```{eval-rst}
.. autofunction:: nlpstats.correlations.bootstrap.bootstrap

.. autoclass:: nlpstats.correlations.bootstrap.BootstrapResult
    :members:
    :member-order: bysource

.. autofunction:: nlpstats.correlations.fisher.fisher

.. autoclass:: nlpstats.correlations.fisher.FisherResult
    :members:
    :member-order: bysource
```

## Hypothesis Testing

This module contains two methods for hypothesis testing the difference between two correlations: permutation tests and Williams' test.
The documentation for both functions is described next.

```{eval-rst}
.. autofunction:: nlpstats.correlations.permutation.permutation_test

.. autoclass:: nlpstats.correlations.permutation.PermutationResult
    :members:
    :member-order: bysource

.. autofunction:: nlpstats.correlations.williams.williams_test

.. autoclass:: nlpstats.correlations.williams.WilliamsResult
    :members:
    :member-order: bysource
```
