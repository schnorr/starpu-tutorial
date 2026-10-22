# Space-time view, submitted and ready tasks of two gemm_starpu traces
# side by side, on the same time axis (Figure img.starvz of the chapter).
# Run StarVZ phase 1 on each trace directory first, then:
#   Rscript starvz_compare.R starvz_compare.yaml <trace NB=2> <trace NB=8> <out.png>
# Two optional arguments replace the panel titles ("NB = 2", "NB = 8").
suppressMessages({ library(starvz); library(ggplot2); library(patchwork) })
args <- commandArgs(trailingOnly = TRUE)
titles <- if (length(args) >= 6) args[5:6] else c("NB = 2", "NB = 8")
traces <- lapply(args[2:3], starvz_read, config_file = args[1])
# common time axis: from 0 to the longest of the two makespans
end <- max(sapply(traces, function(d) max(d$Application$End)))
no_x <- theme(axis.title.x = element_blank(), axis.text.x = element_blank())
column <- function(data, title, left) {
  data$config$limits <- list(start = 0, end = end)
  # the makespan and ABE labels overlap when both are close (NB = 8)
  data$config$st$makespan <- left
  panels <- list(panel_st(data) + ggtitle(title) + no_x,
                 panel_submitted(data) + no_x,
                 panel_ready(data))
  if (!left)
    panels <- lapply(panels, function(p) p + theme(axis.title.y = element_blank()))
  wrap_plots(panels, ncol = 1, heights = c(3, 1, 1))
}
p <- column(traces[[1]], titles[1], TRUE) | column(traces[[2]], titles[2], FALSE)
ggsave(args[4], p, width = 12, height = 6.5, dpi = 150)
