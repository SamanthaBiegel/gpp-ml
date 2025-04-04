library(rsofun)
library(dplyr)
library(tidyr)
library(lubridate)
library(readr)
library(here)

set.seed(42)

new_driver <- readRDS('drivers.rds')

# Prepare validation data
validation <- new_driver %>%
  unnest(forcing) %>%
  select(sitename, date, gpp) %>%
  group_by(sitename) %>%
  nest(data = c(date, gpp))

splits <- list()

splits[[1]] <- list()
splits[[1]]$train_sitename <- c('AT-Neu', 'AU-ASM', 'AU-Cow', 'AU-Cum', 'AU-DaS', 'AU-GWW', 'AU-Gin', 'AU-Tum', 'AU-Ync', 'BE-Maa', 'CA-Ca1', 'CA-Cbo', 'CA-Qfo', 'CA-TP1', 'CA-TP3', 'CA-TPD', 'CH-Aws', 'CH-Cha', 'CH-Dav', 'CH-Fru', 'CH-Oe1', 'CZ-BK1', 'CZ-RAJ', 'CZ-Stn', 'DE-HoH', 'DE-Obe', 'DE-RuW', 'DK-Sor', 'ES-Abr', 'ES-Agu', 'ES-LJu', 'ES-LM1', 'FI-Hyy', 'FI-Let', 'FI-Sod', 'FI-Var', 'FR-Bil', 'FR-FBn', 'FR-Fon', 'FR-LBr', 'FR-Pue', 'IT-Col', 'IT-Cpz', 'IT-Lav', 'IT-Lsn', 'IT-MBo', 'IT-Noe', 'IT-Ren', 'IT-Ro1', 'IT-Ro2', 'IT-SR2', 'IT-Tor', 'NL-Loo', 'RU-Fyo', 'SE-Htm', 'SE-Nor', 'SE-Ros', 'US-Blo', 'US-Fmf', 'US-GLE', 'US-Ha1', 'US-Ho2', 'US-ICh', 'US-ICt', 'US-Jo1', 'US-KFS', 'US-KLS', 'US-MMS', 'US-MOz', 'US-Me2', 'US-NR1', 'US-PFa', 'US-Ro4', 'US-Rwf', 'US-Rws', 'US-SRG', 'US-SRM', 'US-Seg', 'US-Ses', 'US-Syv', 'US-Ton', 'US-UMB', 'US-UMd', 'US-Var', 'US-WCr', 'US-Whs', 'US-Wkg')
splits[[1]]$test_sitename <- c('AU-How', 'AU-Stp', 'BE-Bra', 'BE-Dor', 'BE-Vie', 'CA-Ca2', 'CA-Gro', 'CH-Lae', 'CZ-Lnz', 'DE-Gri', 'DE-Hai', 'DE-RuR', 'DE-Tha', 'ES-LM2', 'IL-Yat', 'RU-Fy2', 'US-BZS', 'US-Bar', 'US-Jo2', 'US-Mpj', 'US-Rms', 'US-Wjs')

splits[[2]] <- list()
splits[[2]]$train_sitename <- c('AT-Neu', 'AU-ASM', 'AU-Cow', 'AU-Cum', 'AU-DaS', 'AU-Gin', 'AU-How', 'AU-Stp', 'AU-Tum', 'AU-Ync', 'BE-Bra', 'BE-Dor', 'BE-Maa', 'BE-Vie', 'CA-Ca1', 'CA-Ca2', 'CA-Gro', 'CA-Qfo', 'CA-TP1', 'CH-Cha', 'CH-Dav', 'CH-Fru', 'CH-Lae', 'CH-Oe1', 'CZ-BK1', 'CZ-Lnz', 'CZ-RAJ', 'CZ-Stn', 'DE-Gri', 'DE-Hai', 'DE-HoH', 'DE-RuR', 'DE-RuW', 'DE-Tha', 'DK-Sor', 'ES-Abr', 'ES-Agu', 'ES-LJu', 'ES-LM1', 'ES-LM2', 'FI-Let', 'FI-Sod', 'FR-Bil', 'FR-FBn', 'FR-Fon', 'FR-LBr', 'FR-Pue', 'IL-Yat', 'IT-Col', 'IT-Cpz', 'IT-Lsn', 'IT-MBo', 'IT-Noe', 'IT-Ren', 'IT-Ro1', 'IT-Tor', 'RU-Fy2', 'RU-Fyo', 'SE-Htm', 'SE-Nor', 'SE-Ros', 'US-BZS', 'US-Bar', 'US-Blo', 'US-GLE', 'US-Ha1', 'US-Ho2', 'US-ICh', 'US-ICt', 'US-Jo2', 'US-KLS', 'US-MOz', 'US-Me2', 'US-Mpj', 'US-PFa', 'US-Rms', 'US-Ro4', 'US-Rwf', 'US-SRM', 'US-Seg', 'US-Ses', 'US-Ton', 'US-UMd', 'US-Var', 'US-WCr', 'US-Whs', 'US-Wjs')
splits[[2]]$test_sitename <- c('AU-GWW', 'CA-Cbo', 'CA-TP3', 'CA-TPD', 'CH-Aws', 'DE-Obe', 'FI-Hyy', 'FI-Var', 'IT-Lav', 'IT-Ro2', 'IT-SR2', 'NL-Loo', 'US-Fmf', 'US-Jo1', 'US-KFS', 'US-MMS', 'US-NR1', 'US-Rws', 'US-SRG', 'US-Syv', 'US-UMB', 'US-Wkg')

splits[[3]] <- list()
splits[[3]]$train_sitename <- c('AT-Neu', 'AU-Cow', 'AU-Cum', 'AU-DaS', 'AU-GWW', 'AU-Gin', 'AU-How', 'AU-Stp', 'AU-Ync', 'BE-Bra', 'BE-Dor', 'BE-Maa', 'BE-Vie', 'CA-Ca1', 'CA-Ca2', 'CA-Cbo', 'CA-Gro', 'CA-Qfo', 'CA-TP1', 'CA-TP3', 'CA-TPD', 'CH-Aws', 'CH-Fru', 'CH-Lae', 'CZ-Lnz', 'CZ-RAJ', 'DE-Gri', 'DE-Hai', 'DE-HoH', 'DE-Obe', 'DE-RuR', 'DE-Tha', 'DK-Sor', 'ES-Abr', 'ES-Agu', 'ES-LM2', 'FI-Hyy', 'FI-Sod', 'FI-Var', 'FR-Bil', 'FR-FBn', 'FR-Fon', 'FR-LBr', 'IL-Yat', 'IT-Col', 'IT-Cpz', 'IT-Lav', 'IT-MBo', 'IT-Ro1', 'IT-Ro2', 'IT-SR2', 'IT-Tor', 'NL-Loo', 'RU-Fy2', 'RU-Fyo', 'SE-Htm', 'SE-Nor', 'SE-Ros', 'US-BZS', 'US-Bar', 'US-Blo', 'US-Fmf', 'US-GLE', 'US-Ha1', 'US-Ho2', 'US-ICh', 'US-ICt', 'US-Jo1', 'US-Jo2', 'US-KFS', 'US-MMS', 'US-MOz', 'US-Mpj', 'US-NR1', 'US-Rms', 'US-Ro4', 'US-Rwf', 'US-Rws', 'US-SRG', 'US-Seg', 'US-Ses', 'US-Syv', 'US-Ton', 'US-UMB', 'US-Var', 'US-Wjs', 'US-Wkg')
splits[[3]]$test_sitename <- c('AU-ASM', 'AU-Tum', 'CH-Cha', 'CH-Dav', 'CH-Oe1', 'CZ-BK1', 'CZ-Stn', 'DE-RuW', 'ES-LJu', 'ES-LM1', 'FI-Let', 'FR-Pue', 'IT-Lsn', 'IT-Noe', 'IT-Ren', 'US-KLS', 'US-Me2', 'US-PFa', 'US-SRM', 'US-UMd', 'US-WCr', 'US-Whs')

splits[[4]] <- list()
splits[[4]]$train_sitename <- c('AU-ASM', 'AU-Cum', 'AU-GWW', 'AU-Gin', 'AU-How', 'AU-Stp', 'AU-Tum', 'AU-Ync', 'BE-Bra', 'BE-Dor', 'BE-Maa', 'BE-Vie', 'CA-Ca1', 'CA-Ca2', 'CA-Cbo', 'CA-Gro', 'CA-TP3', 'CA-TPD', 'CH-Aws', 'CH-Cha', 'CH-Dav', 'CH-Fru', 'CH-Lae', 'CH-Oe1', 'CZ-BK1', 'CZ-Lnz', 'CZ-RAJ', 'CZ-Stn', 'DE-Gri', 'DE-Hai', 'DE-Obe', 'DE-RuR', 'DE-RuW', 'DE-Tha', 'ES-Abr', 'ES-LJu', 'ES-LM1', 'ES-LM2', 'FI-Hyy', 'FI-Let', 'FI-Sod', 'FI-Var', 'FR-FBn', 'FR-Pue', 'IL-Yat', 'IT-Col', 'IT-Lav', 'IT-Lsn', 'IT-MBo', 'IT-Noe', 'IT-Ren', 'IT-Ro2', 'IT-SR2', 'NL-Loo', 'RU-Fy2', 'RU-Fyo', 'SE-Ros', 'US-BZS', 'US-Bar', 'US-Blo', 'US-Fmf', 'US-GLE', 'US-ICt', 'US-Jo1', 'US-Jo2', 'US-KFS', 'US-KLS', 'US-MMS', 'US-MOz', 'US-Me2', 'US-Mpj', 'US-NR1', 'US-PFa', 'US-Rms', 'US-Ro4', 'US-Rws', 'US-SRG', 'US-SRM', 'US-Seg', 'US-Syv', 'US-UMB', 'US-UMd', 'US-Var', 'US-WCr', 'US-Whs', 'US-Wjs', 'US-Wkg')
splits[[4]]$test_sitename <- c('AT-Neu', 'AU-Cow', 'AU-DaS', 'CA-Qfo', 'CA-TP1', 'DE-HoH', 'DK-Sor', 'ES-Agu', 'FR-Bil', 'FR-Fon', 'FR-LBr', 'IT-Cpz', 'IT-Ro1', 'IT-Tor', 'SE-Htm', 'SE-Nor', 'US-Ha1', 'US-Ho2', 'US-ICh', 'US-Rwf', 'US-Ses', 'US-Ton')

splits[[5]] <- list()
splits[[5]]$train_sitename <- c('AT-Neu', 'AU-ASM', 'AU-Cow', 'AU-DaS', 'AU-GWW', 'AU-How', 'AU-Stp', 'AU-Tum', 'BE-Bra', 'BE-Dor', 'BE-Vie', 'CA-Ca2', 'CA-Cbo', 'CA-Gro', 'CA-Qfo', 'CA-TP1', 'CA-TP3', 'CA-TPD', 'CH-Aws', 'CH-Cha', 'CH-Dav', 'CH-Lae', 'CH-Oe1', 'CZ-BK1', 'CZ-Lnz', 'CZ-Stn', 'DE-Gri', 'DE-Hai', 'DE-HoH', 'DE-Obe', 'DE-RuR', 'DE-RuW', 'DE-Tha', 'DK-Sor', 'ES-Agu', 'ES-LJu', 'ES-LM1', 'ES-LM2', 'FI-Hyy', 'FI-Let', 'FI-Var', 'FR-Bil', 'FR-Fon', 'FR-LBr', 'FR-Pue', 'IL-Yat', 'IT-Cpz', 'IT-Lav', 'IT-Lsn', 'IT-Noe', 'IT-Ren', 'IT-Ro1', 'IT-Ro2', 'IT-SR2', 'IT-Tor', 'NL-Loo', 'RU-Fy2', 'SE-Htm', 'SE-Nor', 'US-BZS', 'US-Bar', 'US-Fmf', 'US-Ha1', 'US-Ho2', 'US-ICh', 'US-Jo1', 'US-Jo2', 'US-KFS', 'US-KLS', 'US-MMS', 'US-Me2', 'US-Mpj', 'US-NR1', 'US-PFa', 'US-Rms', 'US-Rwf', 'US-Rws', 'US-SRG', 'US-SRM', 'US-Ses', 'US-Syv', 'US-Ton', 'US-UMB', 'US-UMd', 'US-WCr', 'US-Whs', 'US-Wjs', 'US-Wkg')
splits[[5]]$test_sitename <- c('AU-Cum', 'AU-Gin', 'AU-Ync', 'BE-Maa', 'CA-Ca1', 'CH-Fru', 'CZ-RAJ', 'ES-Abr', 'FI-Sod', 'FR-FBn', 'IT-Col', 'IT-MBo', 'RU-Fyo', 'SE-Ros', 'US-Blo', 'US-GLE', 'US-ICt', 'US-MOz', 'US-Ro4', 'US-Seg', 'US-Var')


# Loop over splits and perform calibration and testing
for (i in 1:length(splits)) {
  cat("Processing Split", i, "\n")

  train_sitename <- splits[[i]]$train_sitename
  test_sitename <- splits[[i]]$test_sitename

  # Driver data for training set
  train_driver <- new_driver %>%
    filter(sitename %in% train_sitename)

  # Observations data for training set
  train_validation <- validation %>%
    filter(sitename %in% train_sitename)

  settings <- list(
    method = "GenSA",
    metric = cost_rmse_pmodel,
    control = list(maxit = 100),
    par = list(
      kphio = list(lower = 0.03, upper = 0.2, init = 0.05),
      kphio_par_a = list(lower = -0.0004, upper = 0.001, init = -0.0025),
      kphio_par_b = list(lower = 10, upper = 30, init = 20),
      soilm_thetastar = list(lower = 0, upper = 240, init = 144)
    )
  )

  # Model training (calibration)
  pars_50 <- calib_sofun(
    drivers = train_driver,
    obs = train_validation,
    settings = settings,
    targets = "gpp",
    par_fixed = list(
      soilm_betao = 0.0,
      beta_unitcostratio = 146.0,
      rd_to_vcmax = 0.014,
      tau_acclim = 30.0,
      kc_jmax = 0.41
    )
  )

  # Assign training site names
  pars_50$sitename <- train_sitename

  # Save calibration results with split number
  output_path <- here::here(paste0("data/calibration_new_data_50_split", i, ".rds"))
  write_rds(pars_50, output_path)

  cat("Calibration for Split", i, "completed and saved to", output_path, "\n")

  # Extract optimized parameters
  optimized_params <- pars_50$par
  print(optimized_params)

  # Prepare parameters for model
  params_modl <- list(
    kphio           = optimized_params[["kphio"]],
    kphio_par_a     = optimized_params[["kphio_par_a"]],
    kphio_par_b     = optimized_params[["kphio_par_b"]],
    soilm_thetastar = optimized_params[["soilm_thetastar"]],
    soilm_betao        = 0.0,
    beta_unitcostratio = 146.0,
    rd_to_vcmax        = 0.014,
    tau_acclim         = 30.0,
    kc_jmax            = 0.41
  )

  # Driver data for test set
  test_driver <- new_driver %>%
    filter(sitename %in% test_sitename)

  # Run the model on test sites
  test_results <- runread_pmodel_f(
        drivers = test_driver,
        par = params_modl,
        makecheck = TRUE,
        parallel = TRUE
      )

  # Save test results
  test_output_path <- here::here(paste0("test_results_split", i, ".rds"))
  write_rds(test_results, test_output_path)

  cat("Model run for test sites in Split", i, "completed and saved to", test_output_path, "\n\n")
}

# Define the file paths
file_paths <- list(
  "test_results_split1.rds",
  "test_results_split2.rds",
  "test_results_split3.rds",
  "test_results_split4.rds",
  "test_results_split5.rds"
)

# Load and combine all test results
test_results <- do.call(rbind, lapply(file_paths, readRDS))

test_results <- subset(test_results, select = -site_info)

test_results_unnest <- test_results %>%
  unnest(data)

# Export to CSV
write.csv(test_results_unnest, "pmodel_global.csv", row.names = FALSE)








# Collect all test_sitename sets into a single list with split numbers
test_sites_list <- lapply(seq_along(splits), function(i) {
  data.frame(sitename = splits[[i]]$test_sitename, split = i, stringsAsFactors = FALSE)
})

# Combine all test_sitename sets into a single data frame
all_test_sites <- do.call(rbind, test_sites_list)

# Find overlapping sitenames across splits
overlapping_sites <- unique(all_test_sites$sitename[duplicated(all_test_sites$sitename)])

# If overlaps exist, determine the splits they appear in
if (length(overlapping_sites) > 0) {
  overlap_info <- lapply(overlapping_sites, function(site) {
    splits_with_site <- all_test_sites$split[all_test_sites$sitename == site]
    data.frame(sitename = site, splits = paste(sort(unique(splits_with_site)), collapse = ","))
  })

  # Combine results into a single data frame
  overlap_info_df <- do.call(rbind, overlap_info)

  cat("The following test sitenames are present in multiple splits along with their split numbers:\n")
  print(overlap_info_df)
} else {
  cat("No overlapping test sitenames found between the splits.\n")
}




# Flatten all train sitename lists into one vector
all_train_sites <- unlist(lapply(splits, function(split) unique(split$train_sitename)))

# Count occurrences of each site in train_sitename
site_counts <- table(all_train_sites)

# Verify each site appears exactly 4 times
if (all(site_counts == 4)) {
  cat("Each site appears exactly 4 times in the train sets.\n")
} else {
  cat("Some sites do not appear exactly 4 times in the train sets.\n")
  print(site_counts[site_counts != 4])
}






library(dplyr)

test_results %>%
  group_by(sitename) %>%
  summarise(entry_count = n()) %>%
  print()







test <- driver %>%
  unnest(forcing)
na_counts <- colSums(is.na(test))
print(na_counts)

