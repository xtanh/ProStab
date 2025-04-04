# Create dataframe from the PCC table data
df <- data.frame(
  Dataset = rep(c("S2648", "S4346", "S461", "S571", "S669", "S783", "S8754", "HIF", "megascale", "P53"), each = 6),
  Model = rep(c("Prostab", "w/o wild seq feature" ,"w/o delta feature", "w/o fusion feature", "w/o transformer encoder", "w/o structure feature"), times = 10),
  Value = c(
    # S2648
    0.6814, 0.6768, 0.6713, 0.5711, 0.5743,  0.5818,
    # S4346
    0.5790, 0.5609, 0.5615, 0.5143, 0.5102,  0.4817,
    # S461
    0.6930, 0.6382,0.6381, 0.6442, 0.6582, 0.6682,
    # S571
    0.3864, 0.3201,0.3656, 0.3832, 0.3752, 0.4208,
    # S669
    0.5010,0.4797, 0.4552, 0.5014, 0.5062, 0.4810,
    # S783
    0.7065,0.7025, 0.7042, 0.6460, 0.6387,  0.6344,
    # S8754
    0.5844,0.5746, 0.5558, 0.5658, 0.5619,  0.5275,
    # HIF
    0.6414,0.6676,  0.6368, 0.5758, 0.5658,  0.5544,
    # megascale
    0.7631, 0.7636,0.7684, 0.6398, 0.6392,  0.6951,
    # P53
    0.80,0.7955, 0.7990, 0.6534, 0.6524,  0.6873
  )
)

# Set model as a factor with specific order
df$Model <- factor(df$Model, 
                   levels = c("Prostab", "w/o wild seq feature", "w/o delta feature", "w/o fusion feature", 
                              "w/o transformer encoder", "w/o structure feature"))

# Calculate values for Prostab model for each dataset, for sorting
dataset_values <- tapply(df$Value[df$Model == "Prostab"], df$Dataset[df$Model == "Prostab"], identity)

# Order datasets by Prostab values from high to low
ordered_datasets <- names(sort(dataset_values, decreasing = TRUE))

# Set dataset factor levels to order by Prostab values
df$Dataset <- factor(df$Dataset, levels = ordered_datasets)

# Create the plot
library(ggplot2)

# Color palette - using the same colors from your original code
model_colors <- c("Prostab" = "#80558c", 
                  "w/o delta feature" = "#af7ab3",
                  "w/o fusion feature" = "#cba0ae", 
                  "w/o transformer encoder" = "#d8b9a0",
                  "w/o wild seq feature" = "#e4d192",
                  "w/o structure feature" = "#e47e8c")

p <- ggplot(df, aes(x = Dataset, y = Value, fill = Model)) + 
  geom_bar(stat = "identity", position = position_dodge(width = 0.85), width = 0.8) + 
  scale_fill_manual(values = model_colors) + 
  scale_y_continuous(expand = c(0, 0)) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_minimal() + 
  theme(
    legend.position = c(0.99, 0.99),
    legend.justification = c(1, 1),
    panel.border = element_rect(colour = "black", fill = NA, size = 1),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(face = "bold", size = 14),
    axis.title.y = element_text(face = "bold", size = 14),
    axis.ticks.y = element_line(color = "black"),
    axis.ticks.x = element_line(color = "black"),
    legend.text = element_text(size = 12, face = "bold"),
    legend.title = element_blank(),
    axis.text.x = element_text(face = "bold", size = 11, hjust = 0.5, color = 'black'),
    axis.text.y = element_text(face = "bold", size = 12),
    plot.margin = unit(c(1, 1, 1, 1), "lines")
  ) + 
  labs(x = "Dataset", y = "PCC")

# Display the plot
print(p)

