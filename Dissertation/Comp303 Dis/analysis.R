# ============================================================
# Statistical analysis — PPO vs Q-Learning maze navigation
# This script tests the primary and secondary hypotheses
# derived from the research question. 
# Data files required (produced by data_prep.py):
#   sim_master.csv          — 80 rows, one per simulation run
#   hardware_master.csv     — 40 rows, one per hardware trial
#   sim_episodes_clean.csv  — 40000 rows, one per episode
# ============================================================

library(readr)   # CSV loading
library(dplyr)   # Data manipulation
library(effsize) # Cohen's d effect sizes

# All three CSV files must be in this folder
setwd("C:/Users/adith/Desktop/Uni/Gituni/Y3/2306514_COMP303/Mujoco/Final Experiment")

# Load data
sim <- read_csv("sim_master.csv",         show_col_types = FALSE)
hw  <- read_csv("hardware_master.csv",    show_col_types = FALSE)
ep  <- read_csv("sim_episodes_clean.csv", show_col_types = FALSE)

# Set factor levels so PPO always appears before Q-Learning in outputs
sim$algorithm <- factor(sim$algorithm, levels = c("PPO", "Q-Learning"))
sim$maze      <- factor(sim$maze,      levels = c("U-maze", "Z-maze"))
hw$algorithm  <- factor(hw$algorithm)
hw$maze       <- factor(hw$maze,       levels = c("U-maze", "Z-maze"))
ep$algorithm  <- factor(ep$algorithm,  levels = c("PPO", "Q-Learning"))
ep$maze       <- factor(ep$maze,       levels = c("U-maze", "Z-maze"))

# Subset by maze for all per-maze analyses
u_sim <- sim %>% filter(maze == "U-maze")
z_sim <- sim %>% filter(maze == "Z-maze")
u_hw  <- hw  %>% filter(maze == "U-maze")
z_hw  <- hw  %>% filter(maze == "Z-maze")

cat("\n Data loaded\n")
cat("Simulation runs:      ", nrow(sim), "\n")
cat("Hardware trials:      ", nrow(hw),  "\n")
cat("Episode rows:         ", nrow(ep),  "\n")

# DESCRIPTIVE STATISTICS
cat("\n Descriptive statistics, simulation \n")

desc_sim <- sim %>%
  group_by(algorithm, maze) %>%
  summarise(
    n                = n(),
    mean_collisions  = round(mean(mean_collisions),         2),
    sd_collisions    = round(sd(mean_collisions),           2),
    mean_time_s      = round(mean(mean_elapsed_s),          3),
    sd_time_s        = round(sd(mean_elapsed_s),            3),
    mean_converge    = round(mean(episodes_to_converge),    1),
    sd_converge      = round(sd(episodes_to_converge),      1),
    success_rate_pct = round(mean(success_rate) * 100,      1),
    .groups = "drop"
  )
print(as.data.frame(desc_sim))

cat("\n Descriptive Satistics, Hardware \n")
desc_hw <- hw %>%
  group_by(maze) %>%
  summarise(
    n               = n(),
    mean_collisions = round(mean(mean_collisions, na.rm = TRUE), 2),
    sd_collisions   = round(sd(mean_collisions,   na.rm = TRUE), 2),
    mean_time_s     = round(mean(mean_elapsed_s,  na.rm = TRUE), 3),
    sd_time_s       = round(sd(mean_elapsed_s,    na.rm = TRUE), 3),
    success_rate    = round(mean(success_rate,    na.rm = TRUE), 3),
    .groups = "drop"
  )
print(as.data.frame(desc_hw))


# ============================================================
# Normalility check 
# Required before choosing between parametric ( t-test)
# and non-parametric (Mann-Whitney U) follow-up tests.
# If p < 0.05 the distribution is non-normal and Mann-Whitney
# should be used for that metric instead of Welch t-test.
# Groups where all values are identical (e.g. PPO
# episodes_to_converge = 500 throughout) are skipped —
# a constant cannot be tested for normality.
# ============================================================
cat("\n normality test \n")
cat("(p < 0.05 = non-normal, use Mann-Whitney instead of t-test)\n\n")

metrics <- c("mean_collisions", "mean_elapsed_s", "episodes_to_converge")

for (mz in c("U-maze", "Z-maze")) {
  for (alg in c("PPO", "Q-Learning")) {
    for (metric in metrics) {
      vals <- sim %>%
        filter(maze == mz, algorithm == alg) %>%
        pull(!!sym(metric))
      
      # Skip if all values are identical — it requires variance
      if (length(unique(vals)) < 2) {
        cat(sprintf("%-10s | %-12s | %-25s | SKIPPED — all values = %s\n",
                    mz, alg, metric, unique(vals)))
        next
      }
      
      sw   <- shapiro.test(vals)
      flag <- ifelse(sw$p.value < 0.05, " NON-NORMAL", "  normal")
      cat(sprintf("%-10s | %-12s | %-25s | W=%.4f | p=%.4f%s\n",
                  mz, alg, metric, sw$statistic, sw$p.value, flag))
    }
  }
}


# H1 — PRIMARY HYPOTHESIS: MANOVA
# H1:  Navigation performance differs significantly between
#      PPO and Q-Learning across collision frequency,
#      completion time, and episodes to convergence.
# H0:  No significant difference.
#
# Test: MANOVA with Pillai's trace
# Run separately for U-maze and Z-maze to assess whether
# the algorithmic difference is consistent across both
# maze geometries.
# Decision rule: reject H0 if p < 0.05.

cat("\n H1: PRIMARY MANOVA — PPO vs Q-Learning \n")

cat("\n U-maze \n")
manova_u <- manova(
  cbind(mean_collisions, mean_elapsed_s, episodes_to_converge) ~ algorithm,
  data = u_sim
)
print(summary(manova_u, test = "Pillai"))

cat("\n Z-maze \n")
manova_z <- manova(
  cbind(mean_collisions, mean_elapsed_s, episodes_to_converge) ~ algorithm,
  data = z_sim
)
print(summary(manova_z, test = "Pillai"))

cat("\n interpratation:\n")
cat("If Pr(>F) < 0.05 for both mazes: H1 is supported.\n")

# ============================================================
cat("\n H1a: COLLISION FREQUENCY \n")

cat("\n U-maze (Mann-Whitney — Q-Learning non-normal)\n")
mw_col_u <- wilcox.test(mean_collisions ~ algorithm, data = u_sim)
d_col_u  <- cohen.d(mean_collisions ~ algorithm, data = u_sim)
print(mw_col_u)
cat("Cohen's d:", round(d_col_u$estimate, 3),
    "| 95% CI [", round(d_col_u$conf.int[1], 3),
    ",", round(d_col_u$conf.int[2], 3), "]\n")

cat("\n Z-maze (Welch t-test — both groups normal) \n")
t_col_z <- t.test(mean_collisions ~ algorithm, data = z_sim, var.equal = FALSE)
d_col_z <- cohen.d(mean_collisions ~ algorithm, data = z_sim)
print(t_col_z)
cat("Cohen's d:", round(d_col_z$estimate, 3),
    "| 95% CI [", round(d_col_z$conf.int[1], 3),
    ",", round(d_col_z$conf.int[2], 3), "]\n")

# ============================================================
# H1b — COMPLETION TIME
# Both mazes: mean_elapsed_s is NON-NORMAL for both groups
# ( p < 0.001 for all elapsed_s groups)
# Therefore Mann-Whitney U is used for both mazes.
# ============================================================
cat("\n H1b: COMPLETION TIME \n")

cat("\n U-maze (Mann-Whitney — both groups non-normal) \n")
mw_time_u <- wilcox.test(mean_elapsed_s ~ algorithm, data = u_sim)
d_time_u  <- cohen.d(mean_elapsed_s ~ algorithm, data = u_sim)
print(mw_time_u)
cat("Cohen's d:", round(d_time_u$estimate, 3),
    "| 95% CI [", round(d_time_u$conf.int[1], 3),
    ",", round(d_time_u$conf.int[2], 3), "]\n")

cat("\n Z-maze (Mann-Whitney — both groups non-normal) \n")
mw_time_z <- wilcox.test(mean_elapsed_s ~ algorithm, data = z_sim)
d_time_z  <- cohen.d(mean_elapsed_s ~ algorithm, data = z_sim)
print(mw_time_z)
cat("Cohen's d:", round(d_time_z$estimate, 3),
    "| 95% CI [", round(d_time_z$conf.int[1], 3),
    ",", round(d_time_z$conf.int[2], 3), "]\n")

# ============================================================
# H1c — EPISODES TO CONVERGENCE
# Alpha = 0.05/3 = 0.017 (Bonferroni correction)
# U-maze only: Q-Learning converged (mean=190.5, sd=90.9)
#              PPO never converged (all values = 500)
# Z-maze: BOTH algorithms never converged (all = 500)
#         H1c is not testable for Z-maze — reported as finding.
#
# Q-Learning U-maze episodes_to_converge is NON-NORMAL
# PPO is constant (500) so Mann-Whitney is used.
# ============================================================
cat("\n H1c: EPISODES TO CONVERGENCE \n")
cat("Bonferroni corrected alpha = 0.05/3 = 0.0167\n")

cat("\n U-maze \n")
cat("PPO: all values = 500 (never converged)\n")
cat("Q-Learning: mean =", round(mean(u_sim$episodes_to_converge[u_sim$algorithm=="Q-Learning"]),1),
    "SD =", round(sd(u_sim$episodes_to_converge[u_sim$algorithm=="Q-Learning"]),1), "\n\n")

mw_conv_u <- wilcox.test(episodes_to_converge ~ algorithm, data = u_sim)
d_conv_u  <- cohen.d(episodes_to_converge ~ algorithm, data = u_sim)
print(mw_conv_u)
cat("Cohen's d:", round(d_conv_u$estimate, 3),
    "| 95% CI [", round(d_conv_u$conf.int[1], 3),
    ",", round(d_conv_u$conf.int[2], 3), "]\n")

cat("\n Z-maze \n")
cat("FINDING: Neither PPO nor Q-Learning converged on the Z-maze.\n")
cat("All episodes_to_converge values = 500 for both algorithms.\n")
cat("H1c is not statistically testable for the Z-maze.\n")
cat("This is reported as a substantive finding: the Z-maze\n")
cat("exceeded the episode budget for both algorithms.\n")

# ============================================================
# H1d — TIME-COLLISION CORRELATION (Pearson's r)
# Tests whether longer completion times are associated with
# higher collision counts across all simulation runs.
# Run separately per maze on all 40 simulation runs combined
# (both algorithms together) to capture the full relationship.
# Also run per algorithm to show within-group patterns.
# ============================================================
cat("\n H1d: TIME-COLLISION CORRELATION \n")

cat("\n U-maze (all 40 runs combined) \n")
cor_u <- cor.test(u_sim$mean_elapsed_s, u_sim$mean_collisions,
                  method = "pearson")
print(cor_u)

cat("\n Z-maze (all 40 runs combined) \n")
cor_z <- cor.test(z_sim$mean_elapsed_s, z_sim$mean_collisions,
                  method = "pearson")
print(cor_z)

cat("\n U-maze PPO only \n")
ppo_u <- u_sim %>% filter(algorithm == "PPO")
cor_ppo_u <- cor.test(ppo_u$mean_elapsed_s, ppo_u$mean_collisions,
                      method = "pearson")
print(cor_ppo_u)

cat("\n U-maze Q-Learning only \n")
ql_u <- u_sim %>% filter(algorithm == "Q-Learning")
cor_ql_u <- cor.test(ql_u$mean_elapsed_s, ql_u$mean_collisions, method = "pearson")
print(cor_ql_u)

# ============================================================
# MD1 — Maze difficulty, U and Z Maze
# Does the algorithm x maze interaction show that the
# performance gap between PPO and Q-Learning changes
# depending on maze complexity?
# ============================================================
cat("\n MD1: TWO-WAY ANOVA — Algorithm x Maze Interaction \n")

cat("\n Collisions \n")
aov_col <- aov(mean_collisions ~ algorithm * maze, data = sim)
print(summary(aov_col))

cat("\n Completion time \n")
aov_time <- aov(mean_elapsed_s ~ algorithm * maze, data = sim)
print(summary(aov_time))

# ============================================================
# MD2 — hardware baseline vs simulation 
# Does the physical UR7e show significantly more collisions
# and longer completion time on the Z-maze vs U-maze?
# Both metrics non-normal likely — check first.
# ============================================================
cat("\n MD2: HARDWARE BASELINE — U-maze vs Z-maze \n")

cat("\nNormality checks:\n")
for (metric in c("mean_collisions", "mean_elapsed_s")) {
  for (mz in c("U-maze", "Z-maze")) {
    vals <- hw %>% filter(maze == mz) %>% pull(!!sym(metric))
    if (length(unique(vals)) < 2) {
      cat(sprintf("%-15s | %-8s | SKIPPED — constant\n", metric, mz))
      next
    }
    sw <- shapiro.test(vals)
    flag <- ifelse(sw$p.value < 0.05, "NON-NORMAL", "normal")
    cat(sprintf("%-20s | %-8s | W=%.4f | p=%.4f | %s\n",
                metric, mz, sw$statistic, sw$p.value, flag))
  }
}

cat("\n Collisions: U-maze vs Z-maze hardware \n")
t_hw_col <- t.test(mean_collisions ~ maze, data = hw, var.equal = FALSE)
d_hw_col <- cohen.d(mean_collisions ~ maze, data = hw)
print(t_hw_col)
cat("Cohen's d:", round(d_hw_col$estimate, 3),
    "| 95% CI [", round(d_hw_col$conf.int[1], 3),
    ",", round(d_hw_col$conf.int[2], 3), "]\n")

cat("\n Completion time: U-maze vs Z-maze hardware \n")
t_hw_time <- t.test(mean_elapsed_s ~ maze, data = hw, var.equal = FALSE)
d_hw_time <- cohen.d(mean_elapsed_s ~ maze, data = hw)
print(t_hw_time)
cat("Cohen's d:", round(d_hw_time$estimate, 3),
    "| 95% CI [", round(d_hw_time$conf.int[1], 3),
    ",", round(d_hw_time$conf.int[2], 3), "]\n")

