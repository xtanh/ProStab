# Create dataframe from your updated table data
df <- data.frame(
  Dataset = rep(c("S2648", "S4346", "S461", "S571", "S669", "S783", "S8754", "Fireprot", "Megascale", "P53"), each = 6),
  Model = rep(c("Prostab", "w/o wild seq feature","w/o delta feature", "w/o fusion feature", "w/o transformer encoder","w/o structure feature"), times = 10),
  Value = c(
    # S2648
    0.6904, 0.6832, 0.682, 0.5699, 0.5725,0.5965,
    # S4346
    0.6295, 0.62,0.609, 0.5223, 0.5217,0.4952,
    # S461
    0.7102, 0.66,0.6597, 0.6437, 0.6581,0.6741,
    # S571
    0.4625, 0.46,0.4331, 0.4316, 0.4226,0.4562,
    # S669
    0.5308, 0.5293,0.4998, 0.5196, 0.5253,0.5054,
    # S783
    0.7086, 0.69,0.71025, 0.6539, 0.6363,0.6288,
    # S8754
    0.6386,0.61, 0.6206, 0.5686, 0.5721,0.5609,
    # RIE
    0.6512, 0.66,0.6465, 0.5452, 0.5352,0.5461,
    # megascale
    0.7427, 0.7286,0.7472, 0.5945, 0.6023,0.6529,
    # P53
    0.8185,0.81,0.8152, 0.5878, 0.5740,0.6731
  )
)

# Set model as a factor with specific order
df$Model <- factor(df$Model, 
                   levels = c("Prostab", "w/o wild seq feature","w/o delta feature", "w/o fusion feature", "w/o transformer encoder","w/o structure feature"))

# 计算每个数据集的Prostab模型的值，用于排序
dataset_values <- tapply(df$Value[df$Model == "Prostab"], df$Dataset[df$Model == "Prostab"], identity)
# 按照Prostab模型的值从高到低对数据集进行排序
ordered_datasets <- names(sort(dataset_values, decreasing = TRUE))

# 设置数据集的因子水平，使其按照Prostab值的降序排列
df$Dataset <- factor(df$Dataset, levels = ordered_datasets)

# Create the plot
library(ggplot2)

# Create a color palette - you can adjust these colors as needed
model_colors <- c("Prostab" = "#80558c", 
                  "w/o wild seq feature" = "#e4d192",
                  "w/o delta feature" = "#af7ab3",
                  "w/o fusion feature" = "#cba0ae", 
                  "w/o transformer encoder" = "#d8b9a0",
                  "w/o structure feature" = "#e47e8c")

p <- ggplot(df, aes(x = Dataset, y = Value, fill = Model)) + 
  geom_bar(stat = "identity", position = position_dodge(width = 0.85), width = 0.8) + 
  scale_fill_manual(values = model_colors) + 
  scale_y_continuous(expand = c(0, 0)) +
  coord_cartesian(ylim = c(0, 1)) +  # Adjusted y-axis range to fit the new data
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
    axis.text.x = element_text(face = "bold", size = 11, hjust = 0.5,color = 'black'),
    axis.text.y = element_text(face = "bold", size = 12),
    plot.margin = unit(c(1, 1, 1, 1), "lines")
  ) + 
  labs(x = "Dataset", y = "Spearman ρ")

# Display the plot
print(p)

