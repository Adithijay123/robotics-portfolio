library(dplyr)
library(readr)
library(ggplot2)
library(zoo)
library(scales)

setwd("C:/Users/adith/Desktop/Uni/Gituni/Y3/2306514_COMP303/Mujoco/Final Experiment")

sim <- read_csv("sim_master.csv", show_col_types = FALSE)
hw  <- read_csv("hardware_master.csv", show_col_types = FALSE)
ep  <- read_csv("sim_episodes_clean.csv", show_col_types = FALSE)

sim$algorithm <- factor(sim$algorithm, levels = c("PPO", "Q-Learning"))
sim$maze      <- factor(sim$maze,      levels = c("U-maze", "Z-maze"))
hw$maze       <- factor(hw$maze,       levels = c("U-maze", "Z-maze"))
ep$algorithm  <- factor(ep$algorithm,  levels = c("PPO", "Q-Learning"))
ep$maze       <- factor(ep$maze,       levels = c("U-maze", "Z-maze"))

sim_box <- sim %>% mutate(source = paste(algorithm, "(sim)"))
hw_box  <- hw  %>% mutate(source = "Hardcoded (hw)", algorithm = "Hardcoded")

# ── Shared theme ──────────────────────────────────────────────
base_theme <- theme_minimal(base_size = 12, base_family = "serif") +
  theme(
    plot.title    = element_text(size = 13, face = "bold",
                                 hjust = 0.5, margin = margin(b = 8)),
    plot.caption  = element_text(size = 9, hjust = 0.5,
                                 face = "italic", margin = margin(t = 8)),
    axis.title    = element_text(size = 11),
    axis.text     = element_text(size = 10),
    axis.text.x   = element_text(angle = 25, hjust = 1),
    strip.text    = element_text(size = 11, face = "bold"),
    legend.title  = element_text(size = 11, face = "bold"),
    legend.text   = element_text(size = 10),
    legend.position = "right",
    panel.grid.minor = element_blank(),
    plot.margin   = margin(10, 15, 10, 15)
  )

# ── Colour palette ────────────────────────────────────────────
algo_colours <- c("PPO" = "#E07B5D", "Q-Learning" = "#5DB8C8")
cond_colours <- c("PPO (sim)" = "#E07B5D",
                  "Q-Learning (sim)" = "#5DB8C8",
                  "Hardcoded (hw)" = "#7D9B76")

# ── Figure 1: Collision boxplot ───────────────────────────────
all_box <- bind_rows(
  sim_box %>% select(source, maze, mean_collisions),
  hw_box  %>% select(source, maze, mean_collisions)
)
all_box$source <- factor(all_box$source,
                         levels = c("PPO (sim)", "Q-Learning (sim)", "Hardcoded (hw)"))

p1 <- ggplot(all_box,
             aes(x = source, y = mean_collisions, fill = source)) +
  geom_boxplot(alpha = 0.85, outlier.shape = 16,
               outlier.size = 2, width = 0.55) +
  scale_fill_manual(values = cond_colours, guide = "none") +
  facet_wrap(~maze) +
  labs(
    title   = "Mean Collisions per Episode by Condition",
    x       = NULL,
    y       = "Mean collisions per episode"
  ) +
  base_theme +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

ggsave("fig1_collisions.png", p1, width = 9, height = 5, dpi = 300)
cat("Figure 1 saved.\n")

# ── Figure 2: Time boxplot (y-axis capped at 40s) ────────────
all_time <- bind_rows(
  sim_box %>% select(source, maze, mean_elapsed_s),
  hw_box  %>% select(source, maze, mean_elapsed_s)
)
all_time$source <- factor(all_time$source,
                          levels = c("PPO (sim)", "Q-Learning (sim)", "Hardcoded (hw)"))

p2 <- ggplot(all_time,
             aes(x = source, y = mean_elapsed_s, fill = source)) +
  geom_boxplot(alpha = 0.85, outlier.shape = 16,
               outlier.size = 2, width = 0.55) +
  scale_fill_manual(values = cond_colours, guide = "none") +
  facet_wrap(~maze) +
  coord_cartesian(ylim = c(0, 40)) +
  labs(
    title   = "Mean Elapsed Time per Episode by Condition",
    x       = NULL,
    y       = "Mean elapsed time (s)",
    caption = "Note: one PPO Z-maze outlier (219.8 s) excluded from view for readability"
  ) +
  base_theme +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

ggsave("fig2_time.png", p2, width = 9, height = 5, dpi = 300)
cat("Figure 2 saved.\n")

# ── Figure 3: Rolling success rate learning curve ────────────
ep$goal_reached <- as.numeric(ep$goal_reached)
ep$trial        <- as.numeric(ep$trial)

ep_success <- ep %>%
  group_by(algorithm, maze, trial) %>%
  summarise(mean_success = mean(goal_reached, na.rm = TRUE),
            .groups = "drop") %>%
  group_by(algorithm, maze) %>%
  arrange(trial) %>%
  mutate(roll = rollmean(mean_success, k = 50,
                         fill = NA, align = "right"))

p3 <- ggplot(ep_success,
             aes(x = trial, y = roll, colour = algorithm)) +
  geom_line(linewidth = 1.0, na.rm = TRUE) +
  scale_colour_manual(values = algo_colours) +
  scale_y_continuous(labels = percent_format()) +
  facet_wrap(~maze) +
  labs(
    title   = "Rolling Success Rate over Training Episodes",
    x       = "Episode",
    y       = "Success rate",
    colour  = "Algorithm",
    caption = "50-episode rolling window averaged across 20 runs per condition"
  ) +
  base_theme

ggsave("fig3_learning_curve.png", p3, width = 9, height = 5, dpi = 300)
cat("Figure 3 saved.\n")

# ── Figure 4: Scatter time vs collisions ─────────────────────
p4 <- ggplot(sim,
             aes(x = mean_collisions, y = mean_elapsed_s,
                 colour = algorithm)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 0.9) +
  scale_colour_manual(values = algo_colours) +
  facet_wrap(~maze) +
  coord_cartesian(ylim = c(0, 40)) +
  labs(
    title   = "Completion Time vs Collision Count",
    x       = "Mean collisions per episode",
    y       = "Mean elapsed time (s)",
    colour  = "Algorithm",
    caption = "Note: one PPO Z-maze outlier (219.8 s) excluded from view for readability"
  ) +
  base_theme

ggsave("fig4_scatter.png", p4, width = 9, height = 5, dpi = 300)
cat("Figure 4 saved.\n")

cat("\nAll four figures saved to Final Experiment folder.\n")