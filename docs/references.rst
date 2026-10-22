.. _references:

References
==========

This page lists key references used in the time-series forecasting integration
and in the planned active-learning work (see the surrogate roadmap).

.. list-table::
   :header-rows: 1
   :widths: 10 58 17 15

   * - ID
     - Citation
     - Link
     - Used In
   * - C1
     - *Chronos: Learning the Language of Time Series*. arXiv:2403.07815 (2024).
     - `abs <https://arxiv.org/abs/2403.07815>`__ · `pdf <https://arxiv.org/pdf/2403.07815>`__
     - Chronos adapter, CLI, Streamlit page
   * - AL1
     - Settles, B. *Active Learning Literature Survey*. Computer Sciences Technical Report 1648, University of Wisconsin–Madison (2009).
     - `report <https://minds.wisconsin.edu/handle/1793/60660>`__
     - Surrogate roadmap: Active Learning (query strategies, query-by-committee)
   * - AL2
     - Jones, D. R., Schonlau, M., & Welch, W. J. *Efficient Global Optimization of Expensive Black-Box Functions*. Journal of Global Optimization, 13(4) (1998).
     - `doi <https://doi.org/10.1023/A:1008306431147>`__
     - Surrogate roadmap: Active Learning (expected improvement)
   * - AL3
     - Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design*. ICML (2010).
     - `abs <https://arxiv.org/abs/0912.3995>`__ · `pdf <https://arxiv.org/pdf/0912.3995>`__
     - Surrogate roadmap: Active Learning (GP-UCB)


.. _references-review:

Literature Survey: References to Review
=======================================

Published work to review against how ``multioutreg`` works. The first part
covers methods the code already implements; the second covers the items in
the :doc:`surrogate_roadmap`. Each row says which module the reference applies
to and what to check there. References in the table above (C1, AL1–AL3) are
not repeated.

Every DOI and arXiv link below was checked to resolve to the cited paper
(October 2026).

Methods in the Current Codebase
-------------------------------

Gaussian Processes and Kernel Approximations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
     - Bouhlel, M. A., Bartoli, N., Otsmane, A., & Morlier, J. *Improving Kriging Surrogates of High-Dimensional Design Models by Partial Least Squares Dimension Reduction*. Structural and Multidisciplinary Optimization, 53(5) (2016).
     - `doi <https://doi.org/10.1007/s00158-015-1395-9>`__
     - ``kpls_smt``: choosing the number of PLS components; when KPLS beats a full ARD GP.
   * - GP6
     - Saves, P., et al. *SMT 2.0: A Surrogate Modeling Toolbox with a Focus on Hierarchical and Mixed Variables Gaussian Processes*. Advances in Engineering Software, 188 (2024).
     - `doi <https://doi.org/10.1016/j.advengsoft.2023.103571>`__ · `arXiv <https://arxiv.org/abs/2305.13998>`__
     - ``gpx_smt``, ``kpls_smt``, and roadmap MF item 1 (``MFK`` / ``MFKPLS``). Also cite Bouhlel et al. (2019), `doi <https://doi.org/10.1016/j.advengsoft.2019.03.005>`__, for SMT 1.x.
   * - GP7
     - Tipping, M. E. *Sparse Bayesian Learning and the Relevance Vector Machine*. JMLR, 1 (2001).
     - `paper <https://www.jmlr.org/papers/v1/tipping01a.html>`__
     - ``bayesian_ridge_sklearn``, ``polynomial_bayesian_ridge_sklearn``: evidence maximisation and ARD priors.
   * - GP8
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
     - ``catboost_sklearn``; also roadmap MO item 4 (``MultiRMSE`` joint loss).
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

Uncertainty Calibration and Scoring
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
     - ``performance/metrics``: NLL, CRPS and interval score; also the energy score for roadmap joint metrics.
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
   * - UQ5
     - Hyndman, R. J., & Koehler, A. B. *Another Look at Measures of Forecast Accuracy*. International Journal of Forecasting, 22(4) (2006).
     - `doi <https://doi.org/10.1016/j.ijforecast.2006.03.001>`__
     - ``time_series/metrics``: MASE (and why sMAPE is problematic near zero).

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

Design of Experiments and Sampling Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
     - ``sampling/detect`` (``nn`` statistic) and roadmap AL item 3 (maximin-style batch diversity).
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
     - Dickey, D. A., & Fuller, W. A. *Distribution of the Estimators for Autoregressive Time Series with a Unit Root*. JASA, 74(366) (1979).
     - `doi <https://doi.org/10.2307/2286348>`__
     - ``ts_suitability``, ``ts_dynamic_fit``: ADF unit-root test.
   * - TS2
     - Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. *Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root*. Journal of Econometrics, 54 (1992).
     - `doi <https://doi.org/10.1016/0304-4076(92)90104-Y>`__
     - ``ts_suitability``: KPSS has the opposite null to ADF; the code runs both together, and this paper gives the reasoning for the combined decision (including the trend-stationary case).
   * - TS3
     - Ljung, G. M., & Box, G. E. P. *On a Measure of Lack of Fit in Time Series Models*. Biometrika, 65(2) (1978).
     - `doi <https://doi.org/10.1093/biomet/65.2.297>`__
     - ``ts_suitability``: autocorrelation test; lag choice.
   * - TS4
     - Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.). OTexts (2021).
     - `book <https://otexts.com/fpp3/>`__
     - ``seasonality`` (ACF-based period detection), ``ts_dynamic_fit/algs/arima``, ``sarima``: model identification and prediction intervals.
   * - TS5
     - Hochreiter, S., & Schmidhuber, J. *Long Short-Term Memory*. Neural Computation, 9(8) (1997).
     - `doi <https://doi.org/10.1162/neco.1997.9.8.1735>`__
     - ``ts_dynamic_fit/algs/lstm``.

Interpretability, Visualisation and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
     - ``figures/pca_plots``, ``utils/pca_utils``; also roadmap MO item 2 (PCA output reduction).
   * - VZ4
     - Troyanskaya, O., et al. *Missing Value Estimation Methods for DNA Microarrays*. Bioinformatics, 17(6) (2001).
     - `doi <https://doi.org/10.1093/bioinformatics/17.6.520>`__
     - ``utils/imputation``: KNN imputation; choice of *k* and scaling before distance computation.

Methods for Roadmap Items
-------------------------

Multi-Fidelity Modelling
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 42 15 35

   * - ID
     - Citation
     - Link
     - Relevance / What to Review
   * - MF1
     - Kennedy, M. C., & O'Hagan, A. *Predicting the Output from a Complex Computer Code When Fast Approximations Are Available*. Biometrika, 87(1) (2000).
     - `doi <https://doi.org/10.1093/biomet/87.1.1>`__
     - Current ``AdditiveCorrectionVFM`` (AR1 with constant ``rho``) and roadmap MF item 2 (input-dependent ``rho(x)``).
   * - MF2
     - Le Gratiet, L., & Garnier, J. *Recursive Co-Kriging Model for Design of Computer Experiments with Multiple Levels of Fidelity*. IJUQ, 4(5) (2014).
     - `doi <https://doi.org/10.1615/Int.J.UncertaintyQuantification.2014006914>`__ · `arXiv <https://arxiv.org/abs/1210.0686>`__
     - Roadmap MF item 1 (``MFKSurrogate``): the recursive formulation SMT's ``MFK`` implements, and its nested-design requirement.
   * - MF3
     - Meliani, M., et al. *Multi-Fidelity Efficient Global Optimization: Methodology and Application to Airfoil Shape Design*. AIAA Aviation Forum (2019).
     - `doi <https://doi.org/10.2514/6.2019-3236>`__
     - Roadmap MF item 1 (``MFKPLSSurrogate``): MFK combined with PLS reduction.
   * - MF4
     - Forrester, A. I. J., Sóbester, A., & Keane, A. J. *Multi-Fidelity Optimization via Surrogate Modelling*. Proc. R. Soc. A, 463 (2007).
     - `doi <https://doi.org/10.1098/rspa.2007.1900>`__
     - Co-kriging with joint ML estimation of ``rho``, an alternative to the current least-squares ``rho`` fit; co-kriging EI.
   * - MF5
     - Perdikaris, P., Raissi, M., Damianou, A., Lawrence, N. D., & Karniadakis, G. E. *Nonlinear Information Fusion Algorithms for Data-Efficient Multi-Fidelity Modelling*. Proc. R. Soc. A, 473 (2017).
     - `doi <https://doi.org/10.1098/rspa.2016.0751>`__
     - Current ``StackedVFMSurrogate`` (NARGP-like) and roadmap MF item 4 (proper NARGP kernel with Monte Carlo propagation).
   * - MF6
     - Meng, X., & Karniadakis, G. E. *A Composite Neural Network That Learns from Multi-Fidelity Data*. Journal of Computational Physics, 401 (2020).
     - `doi <https://doi.org/10.1016/j.jcp.2019.109020>`__ · `arXiv <https://arxiv.org/abs/1903.00104>`__
     - Roadmap MF item 5 (multi-fidelity neural networks).
   * - MF7
     - Cutajar, K., Pullin, M., Damianou, A., Lawrence, N., & González, J. *Deep Gaussian Processes for Multi-Fidelity Modeling*. arXiv:1903.07320 (2019).
     - `arXiv <https://arxiv.org/abs/1903.07320>`__
     - Roadmap MF item 6 (multi-fidelity deep GP).
   * - MF8
     - Kandasamy, K., Dasarathy, G., Schneider, J., & Póczos, B. *Multi-Fidelity Bayesian Optimisation with Continuous Approximations*. ICML (2017).
     - `arXiv <https://arxiv.org/abs/1703.06240>`__
     - Roadmap MF item 3 (fidelity as a continuous input).
   * - MF9
     - Wu, J., Toscano-Palmerin, S., Frazier, P. I., & Wilson, A. G. *Practical Multi-Fidelity Bayesian Optimization for Hyperparameter Tuning*. UAI (2020).
     - `arXiv <https://arxiv.org/abs/1903.04703>`__
     - Roadmap MF item 3: the downsampling kernel behind BoTorch's ``SingleTaskMultiFidelityGP``.
   * - MF10
     - Peherstorfer, B., Willcox, K., & Gunzburger, M. *Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization*. SIAM Review, 60(3) (2018).
     - `doi <https://doi.org/10.1137/16M1082469>`__
     - Roadmap MF supporting methods: accuracy-vs-cost evaluation and model-management strategies.
   * - MF11
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
     - Roadmap MO item 1: ICM vs. LMC, and how many latent processes are needed.
   * - MO2
     - Bonilla, E. V., Chai, K. M. A., & Williams, C. K. I. *Multi-Task Gaussian Process Prediction*. NeurIPS 20 (2007).
     - `paper <https://papers.nips.cc/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html>`__
     - Roadmap MO item 1: the free-form task covariance used by GPyTorch's ``MultitaskKernel``; outputs observed on different rows.
   * - MO3
     - Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q., & Wilson, A. G. *GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration*. NeurIPS 31 (2018).
     - `arXiv <https://arxiv.org/abs/1809.11165>`__
     - Current ``sparse_gp_gpytorch`` and roadmap MO item 1 / MF item 3 (GPyTorch implementations).
   * - MO4
     - Higdon, D., Gattiker, J., Williams, B., & Rightley, M. *Computer Model Calibration Using High-Dimensional Output*. JASA, 103(482) (2008).
     - `doi <https://doi.org/10.1198/016214507000000888>`__
     - Roadmap MO item 2 (PCA output reduction): standard reference for emulating field outputs on a principal-component basis, including how variance maps back.
   * - MO5
     - Schölkopf, B., Smola, A., & Müller, K.-R. *Nonlinear Component Analysis as a Kernel Eigenvalue Problem*. Neural Computation, 10(5) (1998).
     - `doi <https://doi.org/10.1162/089976698300017467>`__
     - Roadmap MO item 2 (kernel PCA variant): note the pre-image problem when mapping predictions back.
   * - MO6
     - Borchani, H., Varando, G., Bielza, C., & Larrañaga, P. *A Survey on Multi-Output Regression*. WIREs Data Mining and Knowledge Discovery, 5(5) (2015).
     - `doi <https://doi.org/10.1002/widm.1157>`__
     - Overview of problem-transformation vs. algorithm-adaptation methods; places the current per-output design and roadmap MO items 3–4.
   * - MO7
     - Spyromitros-Xioufis, E., Tsoumakas, G., Groves, W., & Vlahavas, I. *Multi-Target Regression via Input Space Expansion: Treating Targets as Inputs*. Machine Learning, 104 (2016).
     - `doi <https://doi.org/10.1007/s10994-016-5546-z>`__
     - Roadmap MO item 3 (``RegressorChain``): ensembles of chains and the train/test mismatch from using true vs. predicted targets.
   * - MO8
     - Segal, M., & Xiao, Y. *Multivariate Random Forests*. WIREs Data Mining and Knowledge Discovery, 1(1) (2011).
     - `doi <https://doi.org/10.1002/widm.12>`__
     - Roadmap MO item 4 (joint-tree random forest).
   * - MO9
     - Wold, S., Sjöström, M., & Eriksson, L. *PLS-Regression: A Basic Tool of Chemometrics*. Chemometrics and Intelligent Laboratory Systems, 58(2) (2001).
     - `doi <https://doi.org/10.1016/S0169-7439(01)00155-1>`__
     - Roadmap MO item 4 (``PLSRegression``).
   * - MO10
     - Izenman, A. J. *Reduced-Rank Regression for the Multivariate Linear Model*. Journal of Multivariate Analysis, 5(2) (1975).
     - `doi <https://doi.org/10.1016/0047-259X(75)90042-1>`__
     - Roadmap MO item 4 (reduced-rank regression).
   * - MO11
     - Nelsen, R. B. *An Introduction to Copulas* (2nd ed.). Springer (2006).
     - `doi <https://doi.org/10.1007/0-387-28678-0>`__
     - Roadmap MO item 5 (Gaussian copula on residuals).
   * - MO12
     - Gneiting, T., Stanberry, L. I., Grimit, E. P., Held, L., & Johnson, N. A. *Assessing Probabilistic Forecasts of Multivariate Quantities*. TEST, 17 (2008).
     - `doi <https://doi.org/10.1007/s11749-008-0114-x>`__
     - Roadmap joint metrics: energy score and multivariate rank histograms.
   * - MO13
     - Scheuerer, M., & Hamill, T. M. *Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities*. Monthly Weather Review, 143(4) (2015).
     - `doi <https://doi.org/10.1175/MWR-D-14-00269.1>`__
     - Roadmap joint metrics: variogram score, which is more sensitive to correlation errors than the energy score.
   * - MO14
     - Messoudi, S., Destercke, S., & Rousseau, S. *Copula-Based Conformal Prediction for Multi-Target Regression*. Pattern Recognition, 120 (2021).
     - `doi <https://doi.org/10.1016/j.patcog.2021.108101>`__ · `arXiv <https://arxiv.org/abs/2101.12002>`__
     - Roadmap joint conformal regions: simultaneous coverage across outputs, an alternative to max-norm scores.
   * - MO15
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
   * - AL4
     - Kushner, H. J. *A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise*. Journal of Basic Engineering, 86(1) (1964).
     - `doi <https://doi.org/10.1115/1.3653121>`__
     - Roadmap AL item 1: probability of improvement (PI).
   * - AL5
     - MacKay, D. J. C. *Information-Based Objective Functions for Active Data Selection*. Neural Computation, 4(4) (1992).
     - `doi <https://doi.org/10.1162/neco.1992.4.4.590>`__
     - Roadmap AL item 1: max-variance / information-gain acquisition for global surrogate accuracy.
   * - AL6
     - Cohn, D. A., Ghahramani, Z., & Jordan, M. I. *Active Learning with Statistical Models*. JAIR, 4 (1996).
     - `arXiv <https://arxiv.org/abs/cs/9603104>`__
     - Roadmap AL item 1: integrated-variance reduction as an alternative to pointwise max variance.
   * - AL7
     - Seung, H. S., Opper, M., & Sompolinsky, H. *Query by Committee*. COLT (1992).
     - `doi <https://doi.org/10.1145/130385.130417>`__
     - Roadmap AL item 2 (query-by-committee).
   * - AL8
     - Burbidge, R., Rowland, J. J., & King, R. D. *Active Learning for Regression Based on Query by Committee*. IDEAL, LNCS 4881 (2007).
     - `doi <https://doi.org/10.1007/978-3-540-77226-2_22>`__
     - Roadmap AL item 2: QBC for regression with bootstrap committees, which suits surrogates with no ``return_std`` (KNN, linear, trees).
   * - AL9
     - Wu, D., Lin, C.-T., & Huang, J. *Active Learning for Regression Using Greedy Sampling*. Information Sciences, 474 (2019).
     - `doi <https://doi.org/10.1016/j.ins.2018.09.060>`__ · `arXiv <https://arxiv.org/abs/1808.04245>`__
     - Roadmap AL items 2–3: model-free input- and output-space diversity sampling.
   * - AL10
     - González, J., Dai, Z., Hennig, P., & Lawrence, N. *Batch Bayesian Optimization via Local Penalization*. AISTATS (2016).
     - `arXiv <https://arxiv.org/abs/1505.08052>`__
     - Roadmap AL item 3: distance-penalised batch selection.
   * - AL11
     - Arthur, D., & Vassilvitskii, S. *k-means++: The Advantages of Careful Seeding*. SODA (2007).
     - `acm <https://dl.acm.org/doi/10.5555/1283383.1283494>`__
     - Roadmap AL item 3: k-means++-style seeding for diverse batches.
   * - AL12
     - Sener, O., & Savarese, S. *Active Learning for Convolutional Neural Networks: A Core-Set Approach*. ICLR (2018).
     - `arXiv <https://arxiv.org/abs/1708.00489>`__
     - Roadmap AL item 3: k-center greedy coverage as a batch-diversity criterion.
   * - AL13
     - Kirsch, A., van Amersfoort, J., & Gal, Y. *BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning*. NeurIPS 32 (2019).
     - `arXiv <https://arxiv.org/abs/1906.08158>`__
     - Roadmap AL item 3 for the BNN / deep-ensemble surrogates: joint mutual information instead of top-k scoring.
   * - AL14
     - Wang, Z., & Jegelka, S. *Max-Value Entropy Search for Efficient Bayesian Optimization*. ICML (2017).
     - `arXiv <https://arxiv.org/abs/1703.01968>`__
     - Roadmap AL item 5 / MF supporting methods: max-value entropy search.
   * - AL15
     - Takeno, S., et al. *Multi-Fidelity Bayesian Optimization with Max-Value Entropy Search and Its Parallelization*. ICML (2020).
     - `arXiv <https://arxiv.org/abs/1901.08275>`__
     - Roadmap AL item 5: cost-aware MES across fidelity levels.
   * - AL16
     - Huang, D., Allen, T. T., Notz, W. I., & Miller, R. A. *Sequential Kriging Optimization Using Multiple-Fidelity Evaluations*. Structural and Multidisciplinary Optimization, 32(5) (2006).
     - `doi <https://doi.org/10.1007/s00158-005-0587-0>`__
     - Roadmap AL item 5: multi-fidelity expected improvement.
   * - AL17
     - Balandat, M., et al. *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*. NeurIPS 33 (2020).
     - `arXiv <https://arxiv.org/abs/1910.06403>`__
     - Roadmap AL items 1, 3, 5: reference implementations (qEI, qUCB, multi-fidelity knowledge gradient) to validate against.
   * - AL18
     - Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. *Taking the Human Out of the Loop: A Review of Bayesian Optimization*. Proceedings of the IEEE, 104(1) (2016).
     - `doi <https://doi.org/10.1109/JPROC.2015.2494218>`__
     - Roadmap AL: survey of acquisition functions and batch strategies.
   * - AL19
     - Frazier, P. I. *A Tutorial on Bayesian Optimization*. arXiv:1807.02811 (2018).
     - `arXiv <https://arxiv.org/abs/1807.02811>`__
     - Roadmap AL item 1: closed forms for EI and noisy-EI.
   * - AL20
     - Gramacy, R. B. *Surrogates: Gaussian Process Modeling, Design, and Optimization for the Applied Sciences*. CRC Press (2020).
     - `doi <https://doi.org/10.1201/9780367815493>`__ · `online <https://bookdown.org/rbg/surrogates/>`__
     - Roadmap AL item 4: sequential design loops (ALM/ALC), stopping rules and space-filling initial designs.
   * - AL21
     - Liu, H., Ong, Y.-S., & Cai, J. *A Survey of Adaptive Sampling for Global Metamodeling in Support of Simulation-Based Complex Engineering Design*. Structural and Multidisciplinary Optimization, 57(1) (2018).
     - `doi <https://doi.org/10.1007/s00158-017-1739-8>`__
     - Roadmap AL item 4: adaptive sampling for *global accuracy* (the main use case here) rather than optimisation.
   * - AL22
     - Fuhg, J. N., Fau, A., & Nackenhorst, U. *State-of-the-Art and Comparative Review of Adaptive Sampling Methods for Kriging*. Archives of Computational Methods in Engineering, 28 (2021).
     - `doi <https://doi.org/10.1007/s11831-020-09474-6>`__
     - Roadmap AL items 1 and 4: benchmark comparison of variance-, cross-validation- and gradient-based criteria.

Suggested Reading Order
-----------------------

For the "Recommended First Batch" in the :doc:`surrogate_roadmap`:

1. ``MFKSurrogate``: MF1, MF2, MF4, GP6.
2. PCA output reduction: MO4, VZ3.
3. Multi-task GP: MO1, MO2, MO3.
4. Gaussian copula and joint metrics: MO11, MO12, MO13, UQ1.

For checks on code that already exists: MS2 (Auto-Detect score optimism),
TR3/TR4 (``forest_uncertainty``), NN2 (MC Dropout variance has no τ⁻¹ term in
``bnn_pytorch``), TR10/CP5 (``quantile_sklearn`` pseudo-std) and CP4 (coverage
wording in reports).
