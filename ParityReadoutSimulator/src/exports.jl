# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

export
  ##### Units.jl #####
  # Module
  Units,
  # Constants
  μeV,
  kBmK,

  # StatsBase
  kurtosis,
  mean,

  ##### Noise.jl #####
  TelegraphTrajectory,
  generate_noise_trajectory,

  ##### SpectralFunctions.jl #####
  PhenomenologicalRelaxation,
  evaluate_gω,

  ##### BathCorrelationFunctions.jl #####
  BathCorrelationFunction,
  evaluate_gω,
  evaluate_Γ_Rudner,
  evaluate_τ_Rudner,

  ##### BathCouplings.jl #####
  AbstractBathCoupling,
  StaticBathCoupling,
  PeriodicChargeNoiseBathCoupling,

  ##### LindbladOperators.jl #####
  build_Lindblad_operator_fourier_component,
  Lindblad_Fourier_components,
  calculate_quasistatic_Lindblad_matrix,
  get_Lindblad_time_grid,

  ##### QuantumCapacitance.jl #####
  calculate_CQ_prefactor,
  periodic_steady_CQ,
  DrivenHamiltonian,

  ##### TimeTraces.jl #####
  TimeTracesParameters,
  σR_to_σC,
  experiment_duration,
  generate_time_traces,
  generate_long_one_over_f_trajectory,
  generate_telegraph,
  get_time_coords,
  get_sweep_parameters,
  get_number_of_point_per_trace,
  number_of_traces,
  number_of_sweep_parameters,
  build_interpolators_from_xarray,

  ##### Models #####
  ##### models/MPR_TQD.jl #####
  MPR_TQD_Hamiltonian,
  number_operator,
  evaluate_CQ_TQD_model,
  cache_bath_correlation_function,

  ##### models/DQD.jl #####
  evaluate_CQ_DQD_model,

  # Sweeps.jl
  sweep_parameters,
  to_xarray,
  convert_DQD_results_to_xarray,
  convert_TQD_results_to_xarray,
  generate_default_parameters,
  generate_default_charge_noise_parameters,
  generate_default_ng_grid,

  # Plotting.jl
  make_CQ_vs_flux_histograms,
  mix_histograms
