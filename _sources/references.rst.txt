.. _references:

References
==========

Published work behind ``multioutreg``: the methods the code implements today,
and the methods needed for the items in the :doc:`surrogate_roadmap`. Each
table row gives the citation, a link, and a note naming the module the
reference applies to and what to check there.

IDs are a section prefix plus a number (e.g. ``GP3``, ``AL7``). Multi-fidelity
references use ``FID`` so they are not confused with the ``MF`` items in the
audit checklist.

Every DOI and arXiv link was checked to resolve to the cited paper (October
2026).

Where to Start
--------------

**Checks on existing code**

- Auto-Detect reports the CV score of the best model per output, which is
  optimistically biased (MS2).
- ``forest_uncertainty`` adds a noise term that does not vary with *x*
  (TR3, TR4).
- ``bnn_pytorch`` MC Dropout variance has no model-precision (τ⁻¹) term
  (NN2).
- ``quantile_sklearn`` converts quantiles to a symmetric pseudo-std
  (TR10, CP5).
- Reports and GUI labels should quote the same ``1 − 2α`` CV+ guarantee as the
  ``cv_plus`` docstrings (CP4).

**Roadmap "Recommended First Batch"**

1. ``MFKSurrogate``: FID1, FID2, FID4, GP7.
2. PCA output reduction: MO3, VZ3.
3. Multi-task GP: MO1, MO2, GP5.
4. Gaussian copula and joint metrics: MO10, JM1, JM2, UQ1.

Current Codebase
----------------

Methods that ``multioutreg`` implements today, grouped by package.

Gaussian Processes and Bayesian Linear Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - GP1
     - Rasmussen, C. E., & Williams, C. K. I. *Gaussian Processes for Machine Learning*. MIT Press (2006).
     - `book <https://gaussianprocess.org/gpml/>`__
     - ``gp_sklearn``, ``ard_gp_sklearn``: kernel choice, marginal-likelihood fitting and ARD length scales (Ch. 5); predictive variance with and without the noise term.
   * - GP2
     - Rahimi, A., & Recht, B. *Random Features for Large-Scale Kernel Machines*. NeurIPS 20 (2007).
     - `paper <https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html>`__
     - ``rfgp_sklearn``: number of features *D* needed for a given kernel approximation error; RFF variance underestimates away from the data.
   * - GP3
     - Williams, C. K. I., & Seeger, M. *Using the Nyström Method to Speed Up Kernel Machines*. NeurIPS 13 (2001).
     - `paper <https://papers.nips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html>`__
     - ``nystroem_gp_sklearn``: choice and number of landmark points.
   * - GP4
     - Titsias, M. *Variational Learning of Inducing Variables in Sparse Gaussian Processes*. AISTATS, PMLR 5 (2009).
     - `paper <https://proceedings.mlr.press/v5/titsias09a.html>`__
     - ``sparse_gp_gpytorch`` (SGPR): the variational lower bound and inducing-point initialisation.
   * - GP5
     - Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q., & Wilson, A. G. *GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration*. NeurIPS 31 (2018).
     - `arXiv <https://arxiv.org/abs/1809.11165>`__
     - ``sparse_gp_gpytorch``; also the backend for roadmap multi-output #1 (multi-task GP) and multi-fidelity #3 (fidelity as an input).
   * - GP6
     - Bouhlel, M. A., Bartoli, N., Otsmane, A., & Morlier, J. *Improving Kriging Surrogates of High-Dimensional Design Models by Partial Least Squares Dimension Reduction*. Structural and Multidisciplinary Optimization, 53(5) (2016).
     - `doi <https://doi.org/10.1007/s00158-015-1395-9>`__
     - ``kpls_smt``: choosing the number of PLS components; when KPLS beats a full ARD GP.
   * - GP7
     - Saves, P., et al. *SMT 2.0: A Surrogate Modeling Toolbox with a Focus on Hierarchical and Mixed Variables Gaussian Processes*. Advances in Engineering Software, 188 (2024).
     - `doi <https://doi.org/10.1016/j.advengsoft.2023.103571>`__ · `arXiv <https://arxiv.org/abs/2305.13998>`__
     - ``gpx_smt``, ``kpls_smt``, and roadmap multi-fidelity #1 (SMT's ``MFK`` / ``MFKPLS``). Also cite Bouhlel et al. (2019), `doi <https://doi.org/10.1016/j.advengsoft.2019.03.005>`__, for SMT 1.x.
   * - GP8
     - Tipping, M. E. *Sparse Bayesian Learning and the Relevance Vector Machine*. JMLR, 1 (2001).
     - `paper <https://www.jmlr.org/papers/v1/tipping01a.html>`__
     - ``bayesian_ridge_sklearn``, ``polynomial_bayesian_ridge_sklearn``: evidence maximisation and ARD priors.
   * - GP9
     - MacKay, D. J. C. *Bayesian Interpolation*. Neural Computation, 4(3) (1992).
     - `doi <https://doi.org/10.1162/neco.1992.4.3.415>`__
     - Evidence framework behind sklearn's ``BayesianRidge`` hyperparameter updates.

Tree Ensembles, Boosting and Quantile Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - TR1
     - Breiman, L. *Random Forests*. Machine Learning, 45 (2001).
     - `doi <https://doi.org/10.1023/A:1010933404324>`__
     - ``rf_sklearn``, ``forest_uncertainty``: out-of-bag estimates used for ``noise_var``.
   * - TR2
     - Geurts, P., Ernst, D., & Wehenkel, L. *Extremely Randomized Trees*. Machine Learning, 63 (2006).
     - `doi <https://doi.org/10.1007/s10994-006-6226-1>`__
     - ``extra_trees_sklearn``: no bootstrap by default, so ``forest_uncertainty`` falls back to K-fold refits.
   * - TR3
     - Wager, S., Hastie, T., & Efron, B. *Confidence Intervals for Random Forests: The Jackknife and the Infinitesimal Jackknife*. JMLR, 15 (2014).
     - `paper <https://www.jmlr.org/papers/v15/wager14a.html>`__ · `arXiv <https://arxiv.org/abs/1311.4555>`__
     - ``forest_uncertainty``: a bias-corrected alternative to the raw tree-spread variance; compare it with the current ``tree_var + noise_var`` construction.
   * - TR4
     - Meinshausen, N. *Quantile Regression Forests*. JMLR, 7 (2006).
     - `paper <https://www.jmlr.org/papers/v7/meinshausen06a.html>`__
     - Alternative to the constant-in-*x* ``noise_var`` in ``forest_uncertainty``: gives heteroscedastic, asymmetric intervals from the same forest.
   * - TR5
     - Friedman, J. H. *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5) (2001).
     - `doi <https://doi.org/10.1214/aos/1013203451>`__
     - ``gradient_boosting_sklearn``, ``hist_gradient_boosting_sklearn``, ``quantile_sklearn``; also introduces partial-dependence plots (``figures/pdp_plots``).
   * - TR6
     - Chen, T., & Guestrin, C. *XGBoost: A Scalable Tree Boosting System*. KDD (2016).
     - `doi <https://doi.org/10.1145/2939672.2939785>`__ · `arXiv <https://arxiv.org/abs/1603.02754>`__
     - ``xgboost_sklearn``.
   * - TR7
     - Ke, G., et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 30 (2017).
     - `paper <https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html>`__
     - ``lightgbm_sklearn``.
   * - TR8
     - Prokhorenkova, L., et al. *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS 31 (2018).
     - `arXiv <https://arxiv.org/abs/1706.09516>`__
     - ``catboost_sklearn``; also roadmap multi-output #4 (``MultiRMSE`` joint loss).
   * - TR9
     - Duan, T., et al. *NGBoost: Natural Gradient Boosting for Probabilistic Prediction*. ICML (2020).
     - `arXiv <https://arxiv.org/abs/1910.03225>`__
     - ``ngboost_sklearn``: distributional output and how its std is used by ``predict(return_std=True)``.
   * - TR10
     - Koenker, R., & Bassett, G. *Regression Quantiles*. Econometrica, 46(1) (1978).
     - `doi <https://doi.org/10.2307/1913643>`__
     - ``quantile_sklearn``: pinball loss; the code converts quantiles to a std via the inverse normal CDF, which assumes symmetry. Check that against asymmetric intervals.

Neural Networks and Mixtures of Experts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - NN1
     - Lakshminarayanan, B., Pritzel, A., & Blundell, C. *Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles*. NeurIPS 30 (2017).
     - `arXiv <https://arxiv.org/abs/1612.01474>`__
     - ``deep_ensemble_pytorch``: Gaussian NLL heads, mixture-of-Gaussians combination of members, optional adversarial training.
   * - NN2
     - Gal, Y., & Ghahramani, Z. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML (2016).
     - `arXiv <https://arxiv.org/abs/1506.02142>`__
     - ``bnn_network`` / ``bnn_pytorch`` (MC Dropout): the predictive variance needs a model-precision (τ⁻¹) term besides the sample variance.
   * - NN3
     - Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. *Adaptive Mixtures of Local Experts*. Neural Computation, 3(1) (1991).
     - `doi <https://doi.org/10.1162/neco.1991.3.1.79>`__
     - ``moe_surrogate``, ``gating_network``: soft gating vs. the hard-assignment fit in ``gating_network``.
   * - NN4
     - Jordan, M. I., & Jacobs, R. A. *Hierarchical Mixtures of Experts and the EM Algorithm*. Neural Computation, 6(2) (1994).
     - `doi <https://doi.org/10.1162/neco.1994.6.2.181>`__
     - EM training of gate and experts jointly; how the mixture variance (within + between experts) should be computed.
   * - NN5
     - Yuksel, S. E., Wilson, J. N., & Gader, P. D. *Twenty Years of Mixture of Experts*. IEEE TNNLS, 23(8) (2012).
     - `doi <https://doi.org/10.1109/TNNLS.2012.2200299>`__
     - Survey of gating designs and training schemes for ``MixtureOfExpertsSurrogate``.

Multi-Fidelity Surrogates
~~~~~~~~~~~~~~~~~~~~~~~~~

The multi-fidelity models that exist now (``StackedVFMSurrogate``, ``AdditiveCorrectionVFM``). References for the roadmap multi-fidelity candidates continue the numbering under :ref:`references-roadmap`.

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - FID1
     - Kennedy, M. C., & O'Hagan, A. *Predicting the Output from a Complex Computer Code When Fast Approximations Are Available*. Biometrika, 87(1) (2000).
     - `doi <https://doi.org/10.1093/biomet/87.1.1>`__
     - ``AdditiveCorrectionVFM``: the AR1 model it implements with a constant ``rho``; also roadmap multi-fidelity #2 (input-dependent ``rho(x)``).
   * - FID2
     - Forrester, A. I. J., Sóbester, A., & Keane, A. J. *Multi-Fidelity Optimization via Surrogate Modelling*. Proc. R. Soc. A, 463 (2007).
     - `doi <https://doi.org/10.1098/rspa.2007.1900>`__
     - ``AdditiveCorrectionVFM``: co-kriging with ``rho`` estimated jointly by maximum likelihood, an alternative to the current least-squares fit; also co-kriging EI.
   * - FID3
     - Perdikaris, P., Raissi, M., Damianou, A., Lawrence, N. D., & Karniadakis, G. E. *Nonlinear Information Fusion Algorithms for Data-Efficient Multi-Fidelity Modelling*. Proc. R. Soc. A, 473 (2017).
     - `doi <https://doi.org/10.1098/rspa.2016.0751>`__
     - ``StackedVFMSurrogate`` (NARGP-like); also roadmap multi-fidelity #4 (full NARGP kernel with Monte Carlo propagation).

Conformal Prediction
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - CP1
     - Vovk, V., Gammerman, A., & Shafer, G. *Algorithmic Learning in a Random World*. Springer (2005).
     - `doi <https://doi.org/10.1007/b106715>`__
     - Foundations of ``multioutreg.conformal``.
   * - CP2
     - Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. *Inductive Confidence Machines for Regression*. ECML (2002).
     - `doi <https://doi.org/10.1007/3-540-36755-1_29>`__
     - ``split_conformal``: original split/inductive construction and the finite-sample ``⌈(n+1)(1−α)⌉`` quantile.
   * - CP3
     - Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. *Distribution-Free Predictive Inference for Regression*. JASA, 113(523) (2018).
     - `doi <https://doi.org/10.1080/01621459.2017.1307116>`__ · `arXiv <https://arxiv.org/abs/1604.04173>`__
     - ``SplitConformalPredictor`` with normalized scores (locally weighted conformal, §5.2); ``conformal_network_sklearn``.
   * - CP4
     - Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. *Predictive Inference with the Jackknife+*. Annals of Statistics, 49(1) (2021).
     - `doi <https://doi.org/10.1214/20-AOS1965>`__ · `arXiv <https://arxiv.org/abs/1905.02928>`__
     - ``cv_plus``: source of the ``1 − 2α`` worst-case guarantee stated in the docstrings (CV+ with *K* folds adds a small finite-sample term). Check that reports and GUI labels quote the same guarantee, since empirical coverage is usually close to ``1 − α``.
   * - CP5
     - Romano, Y., Patterson, E., & Candès, E. J. *Conformalized Quantile Regression*. NeurIPS 32 (2019).
     - `arXiv <https://arxiv.org/abs/1905.03222>`__
     - Combine ``QuantileRegressionSurrogate`` with ``ConformalMixin`` to get calibrated asymmetric intervals instead of std-based ones.
   * - CP6
     - Angelopoulos, A. N., & Bates, S. *Conformal Prediction: A Gentle Introduction*. Foundations and Trends in ML, 16(4) (2023).
     - `arXiv <https://arxiv.org/abs/2107.07511>`__
     - Practical checklist for ``conformal/metrics`` (coverage, interval width, conditional coverage diagnostics).
   * - CP7
     - Gibbs, I., & Candès, E. *Adaptive Conformal Inference Under Distribution Shift*. NeurIPS 34 (2021).
     - `arXiv <https://arxiv.org/abs/2106.00170>`__
     - Conformal intervals for the time-series forecasters, where exchangeability fails.

Uncertainty Metrics and Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - UQ1
     - Gneiting, T., & Raftery, A. E. *Strictly Proper Scoring Rules, Prediction, and Estimation*. JASA, 102(477) (2007).
     - `doi <https://doi.org/10.1198/016214506000001437>`__
     - ``performance/metrics``: NLL, CRPS and interval score; also the energy score for the roadmap joint metrics.
   * - UQ2
     - Gneiting, T., Balabdaoui, F., & Raftery, A. E. *Probabilistic Forecasts, Calibration and Sharpness*. JRSS B, 69(2) (2007).
     - `doi <https://doi.org/10.1111/j.1467-9868.2007.00587.x>`__
     - The "maximise sharpness subject to calibration" principle behind the calibration and sharpness figures.
   * - UQ3
     - Kuleshov, V., Fenner, N., & Ermon, S. *Accurate Uncertainties for Deep Learning Using Calibrated Regression*. ICML (2018).
     - `arXiv <https://arxiv.org/abs/1807.00263>`__
     - Quantile-calibration curves (``figures/coverage_plots``, ``uct_calibration_extension``) and isotonic recalibration as a post-hoc fix.
   * - UQ4
     - Chung, Y., Char, I., Guo, H., Schneider, J., & Neiswanger, W. *Uncertainty Toolbox: An Open-Source Library for Assessing, Visualizing, and Improving Uncertainty Quantification*. arXiv:2109.10254 (2021).
     - `arXiv <https://arxiv.org/abs/2109.10254>`__
     - ``figures/uncertainty_toolbox_extension``, ``uct_calibration_extension``: definitions of miscalibration area, sharpness and adversarial group calibration.

Model Selection and Statistical Screening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - MS1
     - Arlot, S., & Celisse, A. *A Survey of Cross-Validation Procedures for Model Selection*. Statistics Surveys, 4 (2010).
     - `doi <https://doi.org/10.1214/09-SS054>`__
     - ``model_selection/auto_detect`` and the grid-search CLI/GUI: per-output CV-based selection.
   * - MS2
     - Cawley, G. C., & Talbot, N. L. C. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR, 11 (2010).
     - `paper <https://jmlr.org/papers/v11/cawley10a.html>`__
     - Auto-Detect reports the CV score of the best model *per output*, which is optimistically biased; nested CV or a held-out test set is needed for an unbiased estimate.
   * - MS3
     - Shapiro, S. S., & Wilk, M. B. *An Analysis of Variance Test for Normality (Complete Samples)*. Biometrika, 52(3–4) (1965).
     - `doi <https://doi.org/10.1093/biomet/52.3-4.591>`__
     - ``screening``: normality test on OLS residuals.
   * - MS4
     - D'Agostino, R., & Pearson, E. S. *Tests for Departure from Normality*. Biometrika, 60(3) (1973).
     - `doi <https://doi.org/10.1093/biomet/60.3.613>`__
     - ``screening``: the K² fallback for larger samples.
   * - MS5
     - Ramsey, J. B. *Tests for Specification Errors in Classical Linear Least-Squares Regression Analysis*. JRSS B, 31(2) (1969).
     - `doi <https://doi.org/10.1111/j.2517-6161.1969.tb00796.x>`__
     - ``screening``: RESET linearity test.
   * - MS6
     - Breusch, T. S., & Pagan, A. R. *A Simple Test for Heteroscedasticity and Random Coefficient Variation*. Econometrica, 47(5) (1979).
     - `doi <https://doi.org/10.2307/1911963>`__
     - ``screening``: supports gating Breusch–Pagan on RESET passing (the test assumes the mean model is correct).
   * - MS7
     - Holm, S. *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 6(2) (1979).
     - `jstor <https://www.jstor.org/stable/4615733>`__
     - ``ModelScreener(correction="holm")``: the default joint p-value correction across outputs and tests.

Sampling and Design of Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - DOE1
     - McKay, M. D., Beckman, R. J., & Conover, W. J. *A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code*. Technometrics, 21(2) (1979).
     - `doi <https://doi.org/10.2307/1268522>`__
     - ``sampling/detect``: the one-point-per-stratum property behind the ``strata`` statistic.
   * - DOE2
     - Sobol', I. M. *On the Distribution of Points in a Cube and the Approximate Evaluation of Integrals*. USSR Comput. Math. & Math. Phys., 7(4) (1967).
     - `doi <https://doi.org/10.1016/0041-5553(67)90144-9>`__
     - ``sampling/detect``: the "Sobol" candidate class.
   * - DOE3
     - Halton, J. H. *On the Efficiency of Certain Quasi-Random Sequences of Points in Evaluating Multi-Dimensional Integrals*. Numerische Mathematik, 2 (1960).
     - `doi <https://doi.org/10.1007/BF01386213>`__
     - ``sampling/detect``: Halton is mixed into the low-discrepancy null.
   * - DOE4
     - Owen, A. B. *Scrambling Sobol' and Niederreiter–Xing Points*. Journal of Complexity, 14(4) (1998).
     - `doi <https://doi.org/10.1006/jcom.1998.0487>`__
     - ``sampling/detect``: scrambled vs. unscrambled sequences (the latter are matched exactly).
   * - DOE5
     - Hickernell, F. J. *A Generalized Discrepancy and Quadrature Error Bound*. Mathematics of Computation, 67(221) (1998).
     - `doi <https://doi.org/10.1090/S0025-5718-98-00894-1>`__
     - ``sampling/detect``: the centered L2 discrepancy (``log_cd`` statistic).
   * - DOE6
     - Johnson, M. E., Moore, L. M., & Ylvisaker, D. *Minimax and Maximin Distance Designs*. JSPI, 26(2) (1990).
     - `doi <https://doi.org/10.1016/0378-3758(90)90122-B>`__
     - ``sampling/detect`` (``nn`` statistic); also roadmap active learning #3 (maximin-style batch diversity).
   * - DOE7
     - Massey, F. J. *The Kolmogorov–Smirnov Test for Goodness of Fit*. JASA, 46(253) (1951).
     - `doi <https://doi.org/10.1080/01621459.1951.10500769>`__
     - ``mor_cli/use_ks_for_sample_bias``: marginal KS tests against the YAML distributions.
   * - DOE8
     - Lilliefors, H. W. *On the Kolmogorov–Smirnov Test for Normality with Mean and Variance Unknown*. JASA, 62(318) (1967).
     - `doi <https://doi.org/10.1080/01621459.1967.10482916>`__
     - KS p-values are only valid when the reference distribution is fixed in advance. If the YAML parameters are ever estimated from the same data, a Lilliefors-type correction (or parametric bootstrap) is needed.

Time Series
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - TS1
     - *Chronos: Learning the Language of Time Series*. arXiv:2403.07815 (2024).
     - `abs <https://arxiv.org/abs/2403.07815>`__ · `pdf <https://arxiv.org/pdf/2403.07815>`__
     - ``time_series/chronos_adapter``, ``time_series/financial``, the forecasting CLI and the Streamlit Time Series page: mean scaling and quantisation of the series, and how sampled trajectories become prediction intervals.
   * - TS2
     - Dickey, D. A., & Fuller, W. A. *Distribution of the Estimators for Autoregressive Time Series with a Unit Root*. JASA, 74(366) (1979).
     - `doi <https://doi.org/10.2307/2286348>`__
     - ``ts_suitability``, ``ts_dynamic_fit``: ADF unit-root test.
   * - TS3
     - Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. *Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root*. Journal of Econometrics, 54 (1992).
     - `doi <https://doi.org/10.1016/0304-4076(92)90104-Y>`__
     - ``ts_suitability``: KPSS has the opposite null to ADF; the code runs both together, and this paper gives the reasoning for the combined decision (including the trend-stationary case).
   * - TS4
     - Ljung, G. M., & Box, G. E. P. *On a Measure of Lack of Fit in Time Series Models*. Biometrika, 65(2) (1978).
     - `doi <https://doi.org/10.1093/biomet/65.2.297>`__
     - ``ts_suitability``: autocorrelation test; lag choice.
   * - TS5
     - Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.). OTexts (2021).
     - `book <https://otexts.com/fpp3/>`__
     - ``seasonality`` (ACF-based period detection), ``ts_dynamic_fit/algs/arima``, ``sarima``: model identification and prediction intervals.
   * - TS6
     - Hochreiter, S., & Schmidhuber, J. *Long Short-Term Memory*. Neural Computation, 9(8) (1997).
     - `doi <https://doi.org/10.1162/neco.1997.9.8.1735>`__
     - ``ts_dynamic_fit/algs/lstm``.
   * - TS7
     - Hyndman, R. J., & Koehler, A. B. *Another Look at Measures of Forecast Accuracy*. International Journal of Forecasting, 22(4) (2006).
     - `doi <https://doi.org/10.1016/j.ijforecast.2006.03.001>`__
     - ``time_series/metrics``: MASE (and why sMAPE is problematic near zero).

Figures and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - VZ1
     - Lundberg, S. M., & Lee, S.-I. *A Unified Approach to Interpreting Model Predictions*. NeurIPS 30 (2017).
     - `arXiv <https://arxiv.org/abs/1705.07874>`__
     - ``figures/shap_multioutput``: per-output SHAP values and the background-data choice for KernelSHAP.
   * - VZ2
     - McInnes, L., Healy, J., & Melville, J. *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426 (2018).
     - `arXiv <https://arxiv.org/abs/1802.03426>`__
     - ``figures/umap_plot_classify``: how far distances and cluster sizes in the embedding can be trusted.
   * - VZ3
     - Jolliffe, I. T., & Cadima, J. *Principal Component Analysis: A Review and Recent Developments*. Phil. Trans. R. Soc. A, 374 (2016).
     - `doi <https://doi.org/10.1098/rsta.2015.0202>`__
     - ``figures/pca_plots``, ``utils/pca_utils``; also roadmap multi-output #2 (PCA output reduction).
   * - VZ4
     - Troyanskaya, O., et al. *Missing Value Estimation Methods for DNA Microarrays*. Bioinformatics, 17(6) (2001).
     - `doi <https://doi.org/10.1093/bioinformatics/17.6.520>`__
     - ``utils/imputation``: KNN imputation; choice of *k* and scaling before distance computation.

.. _references-roadmap:

Roadmap Items
-------------

Methods for the candidates in the :doc:`surrogate_roadmap`. "#N" is the row number in the matching roadmap table.

Multi-Fidelity Modelling
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - FID4
     - Le Gratiet, L., & Garnier, J. *Recursive Co-Kriging Model for Design of Computer Experiments with Multiple Levels of Fidelity*. IJUQ, 4(5) (2014).
     - `doi <https://doi.org/10.1615/Int.J.UncertaintyQuantification.2014006914>`__ · `arXiv <https://arxiv.org/abs/1210.0686>`__
     - Roadmap multi-fidelity #1 (``MFKSurrogate``): the recursive formulation SMT's ``MFK`` implements, and its nested-design requirement.
   * - FID5
     - Meliani, M., et al. *Multi-Fidelity Efficient Global Optimization: Methodology and Application to Airfoil Shape Design*. AIAA Aviation Forum (2019).
     - `doi <https://doi.org/10.2514/6.2019-3236>`__
     - Roadmap multi-fidelity #1 (``MFKPLSSurrogate``): MFK combined with PLS reduction.
   * - FID6
     - Meng, X., & Karniadakis, G. E. *A Composite Neural Network That Learns from Multi-Fidelity Data*. Journal of Computational Physics, 401 (2020).
     - `doi <https://doi.org/10.1016/j.jcp.2019.109020>`__ · `arXiv <https://arxiv.org/abs/1903.00104>`__
     - Roadmap multi-fidelity #5 (multi-fidelity neural networks).
   * - FID7
     - Cutajar, K., Pullin, M., Damianou, A., Lawrence, N., & González, J. *Deep Gaussian Processes for Multi-Fidelity Modeling*. arXiv:1903.07320 (2019).
     - `arXiv <https://arxiv.org/abs/1903.07320>`__
     - Roadmap multi-fidelity #6 (multi-fidelity deep GP).
   * - FID8
     - Kandasamy, K., Dasarathy, G., Schneider, J., & Póczos, B. *Multi-Fidelity Bayesian Optimisation with Continuous Approximations*. ICML (2017).
     - `arXiv <https://arxiv.org/abs/1703.06240>`__
     - Roadmap multi-fidelity #3 (fidelity as a continuous input).
   * - FID9
     - Wu, J., Toscano-Palmerin, S., Frazier, P. I., & Wilson, A. G. *Practical Multi-Fidelity Bayesian Optimization for Hyperparameter Tuning*. UAI (2020).
     - `arXiv <https://arxiv.org/abs/1903.04703>`__
     - Roadmap multi-fidelity #3: the downsampling kernel behind BoTorch's ``SingleTaskMultiFidelityGP``.
   * - FID10
     - Peherstorfer, B., Willcox, K., & Gunzburger, M. *Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization*. SIAM Review, 60(3) (2018).
     - `doi <https://doi.org/10.1137/16M1082469>`__
     - Roadmap multi-fidelity supporting methods: accuracy-vs-cost evaluation and model-management strategies.
   * - FID11
     - Fernández-Godino, M. G., Park, C., Kim, N.-H., & Haftka, R. T. *Review of Multi-Fidelity Models*. arXiv:1609.07196 (2016).
     - `arXiv <https://arxiv.org/abs/1609.07196>`__
     - Taxonomy of correction-based vs. fusion-based models; useful for the Integrated Surrogates table.

Multi-Output Modelling
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - MO1
     - Álvarez, M. A., Rosasco, L., & Lawrence, N. D. *Kernels for Vector-Valued Functions: A Review*. Foundations and Trends in ML, 4(3) (2012).
     - `arXiv <https://arxiv.org/abs/1106.6251>`__
     - Roadmap multi-output #1: ICM vs. LMC, and how many latent processes are needed.
   * - MO2
     - Bonilla, E. V., Chai, K. M. A., & Williams, C. K. I. *Multi-Task Gaussian Process Prediction*. NeurIPS 20 (2007).
     - `paper <https://papers.nips.cc/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html>`__
     - Roadmap multi-output #1: the free-form task covariance used by GPyTorch's ``MultitaskKernel``; outputs observed on different rows.
   * - MO3
     - Higdon, D., Gattiker, J., Williams, B., & Rightley, M. *Computer Model Calibration Using High-Dimensional Output*. JASA, 103(482) (2008).
     - `doi <https://doi.org/10.1198/016214507000000888>`__
     - Roadmap multi-output #2 (PCA output reduction): standard reference for emulating field outputs on a principal-component basis, including how variance maps back.
   * - MO4
     - Schölkopf, B., Smola, A., & Müller, K.-R. *Nonlinear Component Analysis as a Kernel Eigenvalue Problem*. Neural Computation, 10(5) (1998).
     - `doi <https://doi.org/10.1162/089976698300017467>`__
     - Roadmap multi-output #2 (kernel PCA variant): note the pre-image problem when mapping predictions back.
   * - MO5
     - Borchani, H., Varando, G., Bielza, C., & Larrañaga, P. *A Survey on Multi-Output Regression*. WIREs Data Mining and Knowledge Discovery, 5(5) (2015).
     - `doi <https://doi.org/10.1002/widm.1157>`__
     - Overview of problem-transformation vs. algorithm-adaptation methods; places the current per-output design and roadmap multi-output #3–#4.
   * - MO6
     - Spyromitros-Xioufis, E., Tsoumakas, G., Groves, W., & Vlahavas, I. *Multi-Target Regression via Input Space Expansion: Treating Targets as Inputs*. Machine Learning, 104 (2016).
     - `doi <https://doi.org/10.1007/s10994-016-5546-z>`__
     - Roadmap multi-output #3 (``RegressorChain``): ensembles of chains and the train/test mismatch from using true vs. predicted targets.
   * - MO7
     - Segal, M., & Xiao, Y. *Multivariate Random Forests*. WIREs Data Mining and Knowledge Discovery, 1(1) (2011).
     - `doi <https://doi.org/10.1002/widm.12>`__
     - Roadmap multi-output #4 (joint-tree random forest).
   * - MO8
     - Wold, S., Sjöström, M., & Eriksson, L. *PLS-Regression: A Basic Tool of Chemometrics*. Chemometrics and Intelligent Laboratory Systems, 58(2) (2001).
     - `doi <https://doi.org/10.1016/S0169-7439(01)00155-1>`__
     - Roadmap multi-output #4 (``PLSRegression``).
   * - MO9
     - Izenman, A. J. *Reduced-Rank Regression for the Multivariate Linear Model*. Journal of Multivariate Analysis, 5(2) (1975).
     - `doi <https://doi.org/10.1016/0047-259X(75)90042-1>`__
     - Roadmap multi-output #4 (reduced-rank regression).
   * - MO10
     - Nelsen, R. B. *An Introduction to Copulas* (2nd ed.). Springer (2006).
     - `doi <https://doi.org/10.1007/0-387-28678-0>`__
     - Roadmap multi-output #5 (Gaussian copula on residuals).

Joint Metrics and Conformal Regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The roadmap's multi-output supporting methods: scoring rules and conformal regions that test whether output correlations are captured.

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - JM1
     - Gneiting, T., Stanberry, L. I., Grimit, E. P., Held, L., & Johnson, N. A. *Assessing Probabilistic Forecasts of Multivariate Quantities*. TEST, 17 (2008).
     - `doi <https://doi.org/10.1007/s11749-008-0114-x>`__
     - Roadmap joint metrics: energy score and multivariate rank histograms.
   * - JM2
     - Scheuerer, M., & Hamill, T. M. *Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities*. Monthly Weather Review, 143(4) (2015).
     - `doi <https://doi.org/10.1175/MWR-D-14-00269.1>`__
     - Roadmap joint metrics: variogram score, which is more sensitive to correlation errors than the energy score.
   * - JM3
     - Messoudi, S., Destercke, S., & Rousseau, S. *Copula-Based Conformal Prediction for Multi-Target Regression*. Pattern Recognition, 120 (2021).
     - `doi <https://doi.org/10.1016/j.patcog.2021.108101>`__ · `arXiv <https://arxiv.org/abs/2101.12002>`__
     - Roadmap joint conformal regions: simultaneous coverage across outputs, an alternative to max-norm scores.
   * - JM4
     - Johnstone, C., & Cox, B. *Conformal Uncertainty Sets for Robust Optimization*. COPA, PMLR 152 (2021).
     - `paper <https://proceedings.mlr.press/v152/johnstone21a.html>`__
     - Roadmap joint conformal regions: the Mahalanobis nonconformity score (ellipsoidal regions).

Active Learning and Sequential Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - AL1
     - Settles, B. *Active Learning Literature Survey*. Computer Sciences Technical Report 1648, University of Wisconsin–Madison (2009).
     - `report <https://minds.wisconsin.edu/handle/1793/60660>`__
     - Roadmap active learning: taxonomy of query strategies (uncertainty sampling, query-by-committee, expected error reduction).
   * - AL2
     - Jones, D. R., Schonlau, M., & Welch, W. J. *Efficient Global Optimization of Expensive Black-Box Functions*. Journal of Global Optimization, 13(4) (1998).
     - `doi <https://doi.org/10.1023/A:1008306431147>`__
     - Roadmap active learning #1: expected improvement (EI).
   * - AL3
     - Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design*. ICML (2010).
     - `abs <https://arxiv.org/abs/0912.3995>`__ · `pdf <https://arxiv.org/pdf/0912.3995>`__
     - Roadmap active learning #1: GP-UCB and the schedule for its exploration weight.
   * - AL4
     - Kushner, H. J. *A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise*. Journal of Basic Engineering, 86(1) (1964).
     - `doi <https://doi.org/10.1115/1.3653121>`__
     - Roadmap active learning #1: probability of improvement (PI).
   * - AL5
     - MacKay, D. J. C. *Information-Based Objective Functions for Active Data Selection*. Neural Computation, 4(4) (1992).
     - `doi <https://doi.org/10.1162/neco.1992.4.4.590>`__
     - Roadmap active learning #1: max-variance / information-gain acquisition for global surrogate accuracy.
   * - AL6
     - Cohn, D. A., Ghahramani, Z., & Jordan, M. I. *Active Learning with Statistical Models*. JAIR, 4 (1996).
     - `arXiv <https://arxiv.org/abs/cs/9603104>`__
     - Roadmap active learning #1: integrated-variance reduction as an alternative to pointwise max variance.
   * - AL7
     - Seung, H. S., Opper, M., & Sompolinsky, H. *Query by Committee*. COLT (1992).
     - `doi <https://doi.org/10.1145/130385.130417>`__
     - Roadmap active learning #2 (query-by-committee).
   * - AL8
     - Burbidge, R., Rowland, J. J., & King, R. D. *Active Learning for Regression Based on Query by Committee*. IDEAL, LNCS 4881 (2007).
     - `doi <https://doi.org/10.1007/978-3-540-77226-2_22>`__
     - Roadmap active learning #2: QBC for regression with bootstrap committees, which suits surrogates with no ``return_std`` (KNN, linear, trees).
   * - AL9
     - Wu, D., Lin, C.-T., & Huang, J. *Active Learning for Regression Using Greedy Sampling*. Information Sciences, 474 (2019).
     - `doi <https://doi.org/10.1016/j.ins.2018.09.060>`__ · `arXiv <https://arxiv.org/abs/1808.04245>`__
     - Roadmap active learning #2–#3: model-free input- and output-space diversity sampling.
   * - AL10
     - González, J., Dai, Z., Hennig, P., & Lawrence, N. *Batch Bayesian Optimization via Local Penalization*. AISTATS (2016).
     - `arXiv <https://arxiv.org/abs/1505.08052>`__
     - Roadmap active learning #3: distance-penalised batch selection.
   * - AL11
     - Arthur, D., & Vassilvitskii, S. *k-means++: The Advantages of Careful Seeding*. SODA (2007).
     - `acm <https://dl.acm.org/doi/10.5555/1283383.1283494>`__
     - Roadmap active learning #3: k-means++-style seeding for diverse batches.
   * - AL12
     - Sener, O., & Savarese, S. *Active Learning for Convolutional Neural Networks: A Core-Set Approach*. ICLR (2018).
     - `arXiv <https://arxiv.org/abs/1708.00489>`__
     - Roadmap active learning #3: k-center greedy coverage as a batch-diversity criterion.
   * - AL13
     - Kirsch, A., van Amersfoort, J., & Gal, Y. *BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning*. NeurIPS 32 (2019).
     - `arXiv <https://arxiv.org/abs/1906.08158>`__
     - Roadmap active learning #3 for the BNN / deep-ensemble surrogates: joint mutual information instead of top-k scoring.
   * - AL14
     - Wang, Z., & Jegelka, S. *Max-Value Entropy Search for Efficient Bayesian Optimization*. ICML (2017).
     - `arXiv <https://arxiv.org/abs/1703.01968>`__
     - Roadmap active learning #5 / multi-fidelity supporting methods: max-value entropy search.
   * - AL15
     - Takeno, S., et al. *Multi-Fidelity Bayesian Optimization with Max-Value Entropy Search and Its Parallelization*. ICML (2020).
     - `arXiv <https://arxiv.org/abs/1901.08275>`__
     - Roadmap active learning #5: cost-aware MES across fidelity levels.
   * - AL16
     - Huang, D., Allen, T. T., Notz, W. I., & Miller, R. A. *Sequential Kriging Optimization Using Multiple-Fidelity Evaluations*. Structural and Multidisciplinary Optimization, 32(5) (2006).
     - `doi <https://doi.org/10.1007/s00158-005-0587-0>`__
     - Roadmap active learning #5: multi-fidelity expected improvement.
   * - AL17
     - Balandat, M., et al. *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*. NeurIPS 33 (2020).
     - `arXiv <https://arxiv.org/abs/1910.06403>`__
     - Roadmap active learning #1, #3, #5: reference implementations (qEI, qUCB, multi-fidelity knowledge gradient) to validate against.
   * - AL18
     - Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. *Taking the Human Out of the Loop: A Review of Bayesian Optimization*. Proceedings of the IEEE, 104(1) (2016).
     - `doi <https://doi.org/10.1109/JPROC.2015.2494218>`__
     - Roadmap active learning: survey of acquisition functions and batch strategies.
   * - AL19
     - Frazier, P. I. *A Tutorial on Bayesian Optimization*. arXiv:1807.02811 (2018).
     - `arXiv <https://arxiv.org/abs/1807.02811>`__
     - Roadmap active learning #1: closed forms for EI and noisy-EI.
   * - AL20
     - Gramacy, R. B. *Surrogates: Gaussian Process Modeling, Design, and Optimization for the Applied Sciences*. CRC Press (2020).
     - `doi <https://doi.org/10.1201/9780367815493>`__ · `online <https://bookdown.org/rbg/surrogates/>`__
     - Roadmap active learning #4: sequential design loops (ALM/ALC), stopping rules and space-filling initial designs.
   * - AL21
     - Liu, H., Ong, Y.-S., & Cai, J. *A Survey of Adaptive Sampling for Global Metamodeling in Support of Simulation-Based Complex Engineering Design*. Structural and Multidisciplinary Optimization, 57(1) (2018).
     - `doi <https://doi.org/10.1007/s00158-017-1739-8>`__
     - Roadmap active learning #4: adaptive sampling for *global accuracy* (the main use case here) rather than optimisation.
   * - AL22
     - Fuhg, J. N., Fau, A., & Nackenhorst, U. *State-of-the-Art and Comparative Review of Adaptive Sampling Methods for Kriging*. Archives of Computational Methods in Engineering, 28 (2021).
     - `doi <https://doi.org/10.1007/s11831-020-09474-6>`__
     - Roadmap active learning #1 and #4: benchmark comparison of variance-, cross-validation- and gradient-based criteria.

Cross-Cutting Additions
~~~~~~~~~~~~~~~~~~~~~~~

The roadmap's cross-cutting candidates, which the multi-fidelity, multi-output
and active-learning items depend on. Model persistence (#5) has no entry: it
is an engineering task with no research literature behind it.

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - XC1
     - Surjanovic, S., & Bingham, D. *Virtual Library of Simulation Experiments: Test Functions and Datasets*. Simon Fraser University (2013).
     - `site <https://www.sfu.ca/~ssurjano/>`__
     - Roadmap cross-cutting #1 (benchmark suite): reference definitions of Forrester, Branin, Currin, Park and Borehole, including multi-fidelity variants. FID2 is the source of the Forrester multi-fidelity function.
   * - XC2
     - Currin, C., Mitchell, T., Morris, M., & Ylvisaker, D. *Bayesian Prediction of Deterministic Functions, with Applications to the Design and Analysis of Computer Experiments*. JASA, 86(416) (1991).
     - `doi <https://doi.org/10.1080/01621459.1991.10475138>`__
     - Roadmap cross-cutting #1: origin of the Currin test function.
   * - XC3
     - Xiong, S., Qian, P. Z. G., & Wu, C. F. J. *Sequential Design and Analysis of High-Accuracy and Low-Accuracy Computer Codes*. Technometrics, 55(1) (2013).
     - `doi <https://doi.org/10.1080/00401706.2012.723572>`__
     - Roadmap cross-cutting #1: the low-fidelity Borehole and Currin variants commonly used as multi-fidelity benchmarks.
   * - XC4
     - Mainini, L., et al. *Analytical Benchmark Problems for Multifidelity Optimization Methods*. arXiv:2204.07867 (2022).
     - `arXiv <https://arxiv.org/abs/2204.07867>`__
     - Roadmap cross-cutting #1: a curated multi-fidelity benchmark set with stated cost ratios between levels, for the accuracy-vs-cost curves. Its 4-level Forrester problem is used in ``examples/mf_surrogates.ipynb``.
   * - XC5
     - Levi, D., Gispan, L., Giladi, N., & Fetaya, E. *Evaluating and Calibrating Uncertainty Prediction in Regression Tasks*. Sensors, 22(15) (2022).
     - `doi <https://doi.org/10.3390/s22155540>`__ · `arXiv <https://arxiv.org/abs/1905.11659>`__
     - Roadmap cross-cutting #2 (std recalibration): single-scalar std scaling fit on held-out data, and why quantile calibration alone can miss mis-scaled std. Read with UQ3 (isotonic recalibration).
   * - XC6
     - Sobol', I. M. *Global Sensitivity Indices for Nonlinear Mathematical Models and Their Monte Carlo Estimates*. Mathematics and Computers in Simulation, 55(1–3) (2001).
     - `doi <https://doi.org/10.1016/S0378-4754(00)00270-6>`__
     - Roadmap cross-cutting #3 (global sensitivity analysis): definitions of first-order and total Sobol indices.
   * - XC7
     - Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. *Variance Based Sensitivity Analysis of Model Output: Design and Estimator for the Total Sensitivity Index*. Computer Physics Communications, 181(2) (2010).
     - `doi <https://doi.org/10.1016/j.cpc.2009.09.018>`__
     - Roadmap cross-cutting #3: the recommended sampling scheme and estimators (Saltelli for first-order, Jansen for total) on ``scipy.stats.qmc`` samples.
   * - XC8
     - Jansen, M. J. W. *Analysis of Variance Designs for Model Output*. Computer Physics Communications, 117(1–2) (1999).
     - `doi <https://doi.org/10.1016/S0010-4655(98)00154-4>`__
     - Roadmap cross-cutting #3: the Jansen total-effect estimator.
   * - XC9
     - Oakley, J. E., & O'Hagan, A. *Probabilistic Sensitivity Analysis of Complex Models: A Bayesian Approach*. JRSS B, 66(3) (2004).
     - `doi <https://doi.org/10.1111/j.1467-9868.2004.05304.x>`__
     - Roadmap cross-cutting #3: sensitivity indices computed on a GP emulator, with uncertainty from the emulator's posterior.
   * - XC10
     - Marrel, A., Iooss, B., Laurent, B., & Roustant, O. *Calculations of Sobol Indices for the Gaussian Process Metamodel*. Reliability Engineering & System Safety, 94(3) (2009).
     - `doi <https://doi.org/10.1016/j.ress.2008.07.008>`__
     - Roadmap cross-cutting #3: repeating the estimators over GP posterior draws to get indices with confidence intervals, as the roadmap proposes.
   * - XC11
     - Bouhlel, M. A., & Martins, J. R. R. A. *Gradient-Enhanced Kriging for High-Dimensional Problems*. Engineering with Computers, 35(1) (2019).
     - `doi <https://doi.org/10.1007/s00366-018-0590-x>`__ · `arXiv <https://arxiv.org/abs/1708.02663>`__
     - Roadmap cross-cutting #4 (``smt`` GEKPLS): how gradients enter the PLS-reduced kernel, and the extra points generated per sample.
   * - XC12
     - Jaworska, J., Nikolova-Jeliazkova, N., & Aldenberg, T. *QSAR Applicability Domain Estimation by Projection of the Training Set in Descriptor Space: A Review*. ATLA, 33(5) (2005).
     - `doi <https://doi.org/10.1177/026119290503300508>`__
     - Roadmap cross-cutting #6 (extrapolation flag): compares range, convex-hull, Mahalanobis and nearest-neighbour domain checks, the options the roadmap lists.
   * - XC13
     - Meyer, H., & Pebesma, E. *Predicting into Unknown Space? Estimating the Area of Applicability of Spatial Prediction Models*. Methods in Ecology and Evolution, 12(9) (2021).
     - `doi <https://doi.org/10.1111/2041-210X.13650>`__
     - Roadmap cross-cutting #6: a nearest-neighbour dissimilarity index with a threshold calibrated by cross-validation rather than chosen by hand.
   * - XC14
     - Wolpert, D. H. *Stacked Generalization*. Neural Networks, 5(2) (1992).
     - `doi <https://doi.org/10.1016/S0893-6080(05)80023-1>`__
     - Roadmap cross-cutting #7 (ensembling Auto-Detect candidates): stacking.
   * - XC15
     - Breiman, L. *Stacked Regressions*. Machine Learning, 24(1) (1996).
     - `doi <https://doi.org/10.1007/BF00117832>`__
     - Roadmap cross-cutting #7: non-negative weights on out-of-fold predictions, which the Auto-Detect CV loop already produces.
   * - XC16
     - Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. *Using Stacking to Average Bayesian Predictive Distributions*. Bayesian Analysis, 13(3) (2018).
     - `doi <https://doi.org/10.1214/17-BA1091>`__
     - Roadmap cross-cutting #7: stacking predictive *distributions* (maximising held-out log score), so the ensemble keeps a usable ``std``.
   * - XC17
     - Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T. *Bayesian Model Averaging: A Tutorial*. Statistical Science, 14(4) (1999).
     - `doi <https://doi.org/10.1214/ss/1009212519>`__
     - Roadmap cross-cutting #7: the model-averaged mixture, whose between-model variance is also the committee estimate for active learning #2.
   * - XC18
     - Forrester, A. I. J., & Keane, A. J. *Recent Advances in Surrogate-Based Optimization*. Progress in Aerospace Sciences, 45(1–3) (2009).
     - `doi <https://doi.org/10.1016/j.paerosci.2008.11.001>`__
     - Roadmap cross-cutting #8 (optimisation on the surrogate): constrained and multi-objective optimisation of a fitted surrogate.
   * - XC19
     - Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*. IEEE Transactions on Evolutionary Computation, 6(2) (2002).
     - `doi <https://doi.org/10.1109/4235.996017>`__
     - Roadmap cross-cutting #8: Pareto front across outputs on the cheap surrogate.
   * - XC20
     - Daulton, S., Balandat, M., & Bakshy, E. *Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization*. NeurIPS 33 (2020).
     - `arXiv <https://arxiv.org/abs/2006.05078>`__
     - Roadmap cross-cutting #8 combined with active learning: choosing new samples to improve the Pareto front (qEHVI, in BoTorch, AL17).
   * - XC21
     - Zhou, Q., Qian, P. Z. G., & Zhou, S. *A Simple Approach to Emulation for Computer Models with Qualitative and Quantitative Factors*. Technometrics, 53(3) (2011).
     - `doi <https://doi.org/10.1198/TECH.2011.10025>`__
     - Roadmap cross-cutting #9 (categorical inputs): hypersphere-parameterised correlation between categorical levels.
   * - XC22
     - Zhang, Y., Tao, S., Chen, W., & Apley, D. W. *A Latent Variable Approach to Gaussian Process Modeling with Qualitative and Quantitative Factors*. Technometrics, 62(3) (2020).
     - `doi <https://doi.org/10.1080/00401706.2019.1638834>`__
     - Roadmap cross-cutting #9: latent-variable kernels, one of the mixed-variable kernels in SMT 2.0 (GP7).
   * - XC23
     - Garrido-Merchán, E. C., & Hernández-Lobato, D. *Dealing with Categorical and Integer-Valued Variables in Bayesian Optimization with Gaussian Processes*. Neurocomputing, 380 (2020).
     - `doi <https://doi.org/10.1016/j.neucom.2019.11.004>`__ · `arXiv <https://arxiv.org/abs/1805.03463>`__
     - Roadmap cross-cutting #9: one-hot inputs need rounding inside the kernel, otherwise acquisition optimisation proposes invalid points.
   * - XC24
     - Little, R. J. A., & Rubin, D. B. *Statistical Analysis with Missing Data* (3rd ed.). Wiley (2019).
     - `doi <https://doi.org/10.1002/9781119482260>`__
     - Roadmap cross-cutting #10 (missing outputs): per-output masking is unbiased only if outputs are missing at random. Simulation runs that fail in particular input regions are not, so check where the failures are.
   * - XC25
     - Bergstra, J., & Bengio, Y. *Random Search for Hyper-Parameter Optimization*. JMLR, 13 (2012).
     - `paper <https://jmlr.org/papers/v13/bergstra12a.html>`__
     - Roadmap cross-cutting #11 (beyond grid search): why random search beats ``GridSearchCV`` when only a few hyperparameters matter.
   * - XC26
     - Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. *Algorithms for Hyper-Parameter Optimization*. NeurIPS 24 (2011).
     - `paper <https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html>`__
     - Roadmap cross-cutting #11: the Tree-structured Parzen Estimator (TPE) used by Optuna and ``hyperopt``.
   * - XC27
     - Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*. JMLR, 18 (2018).
     - `paper <https://jmlr.org/papers/v18/16-558.html>`__ · `arXiv <https://arxiv.org/abs/1603.06560>`__
     - Roadmap cross-cutting #11: early stopping of poor configurations (as in ``ray[tune]``), useful for the PyTorch surrogates.
   * - XC28
     - Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. *Optuna: A Next-Generation Hyperparameter Optimization Framework*. KDD (2019).
     - `doi <https://doi.org/10.1145/3292500.3330701>`__ · `arXiv <https://arxiv.org/abs/1907.10902>`__
     - Roadmap cross-cutting #11: the define-by-run search spaces and pruning API the roadmap would wrap.
