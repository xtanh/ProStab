# 加载必要的库
library(ggplot2)
library(reshape2)
library(dplyr)

# 创建数据集
data <- data.frame(
  PDB_code = c("2OCJ", "3Q05", "2AC0"),
  Resolution = c(2.05, 2.40, 1.80),
  Biological_assembly = c("Homodimer", "Homotetramer complexed with a DNA helix", "Homotetramer complexed with two DNA helices"),
  Prostab_spearson = c(0.84, 0.84, 0.83),
  Without_delta_feature = c(0.82, 0.84, 0.81)
)

# 将数据从宽格式转换为长格式
melted_data <- melt(data, 
                    id.vars = c("PDB_code", "Resolution", "Biological_assembly"),
                    measure.vars = c("Prostab_spearson", "Without_delta_feature"),
                    variable.name = "Method",
                    value.name = "Stability")

# 按分辨率排序PDB结构
melted_data$PDB_code <- factor(melted_data$PDB_code, 
                               levels = data$PDB_code[order(data$Resolution)])

# 设置自定义的标签 - 添加到melted_data中
melted_data$Resolution_label <- paste0(melted_data$PDB_code, "\n(", melted_data$Resolution, "Å)")

# 创建计算delta的数据集
delta_data <- data %>%
  mutate(
    Resolution_label = paste0(PDB_code, "\n(", Resolution, "Å)"),
    Delta = Prostab_spearson - Without_delta_feature,
    Label_y = pmin(Prostab_spearson, Without_delta_feature) - 0.005
  )

# 创建组合散点图和线图
p <- ggplot(melted_data, aes(x = Resolution_label, y = Stability, color = Method, group = Method)) +
  # 添加点
  geom_point(size = 4, position = position_dodge(width = 0.3)) +
  # 添加线
  geom_line(position = position_dodge(width = 0.3), linewidth = 1) +
  # 添加平均值的水平虚线
  geom_hline(yintercept = mean(melted_data$Stability[melted_data$Method == "Prostab_spearson"]), 
             color = "#1E88E5", linetype = "dashed", alpha = 0.7) +
  geom_hline(yintercept = mean(melted_data$Stability[melted_data$Method == "Without_delta_feature"]), 
             color = "#D81B60", linetype = "dashed", alpha = 0.7) +
  # 添加差值标签 - 使用预先计算好的delta_data
  geom_text(data = delta_data, 
            aes(x = Resolution_label, 
                y = Label_y,
                label = sprintf("Δ = %.3f", Delta)),
            color = "black", size = 3.5, vjust = 1.5) +
  # 设置标题和标签
  labs(
    title = "不同精度PDB结构下模型的稳定性比较",
    subtitle = "比较使用与不使用delta feature的鲁棒性",
    x = "PDB结构 (分辨率)",
    y = "稳定性指标 (Spearson相关性)",
    caption = paste("平均值：使用delta feature =", round(mean(data$Prostab_spearson), 3), 
                    "，不使用delta feature =", round(mean(data$Without_delta_feature), 3),
                    "，平均提升 =", round(mean(data$Prostab_spearson - data$Without_delta_feature), 3))
  ) +
  # 自定义颜色和图例
  scale_color_manual(
    values = c("Prostab_spearson" = "#1E88E5", "Without_delta_feature" = "#D81B60"),
    labels = c("使用delta feature", "不使用delta feature"),
    name = NULL
  ) +
  # 设置y轴范围，稍微扩大一点以便于观察
  scale_y_continuous(limits = c(0.8, 0.85), breaks = seq(0.8, 0.85, 0.01)) +
  # 应用主题
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(color = "gray30"),
    axis.text.x = element_text(angle = 0, hjust = 0.5, face = "bold"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray80", fill = NA)
  )

# 打印图表
print(p)

# 保存图表（如果需要）
# ggsave("pdb_stability_comparison.png", p, width = 9, height = 6, dpi = 300)