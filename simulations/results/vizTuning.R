library(tidyverse)

df <- data.frame("file" = dir()) %>%
    filter(grepl("prec", file)) %>%
    # map read these csvs into nested dataframe
    mutate(data = map(file, read_csv)) %>%
    # unnest them
    unnest(data)
# group by file



# vs alpha, beta
df %>%
    ggplot(aes(x = meanBeta1, fill = factor(prec))) +
    geom_density(alpha = 0.3) +
    geom_vline(aes(xintercept = 3)) +
    # facet
    facet_grid(rows = vars(bet), cols = vars(alph), labeller = labeller(.rows = label_both, .cols = label_both))


df %>%
    ggplot(aes(x = meanBetakclust1)) +
    geom_density(alpha = 0.3) +
    geom_vline(aes(xintercept = 3))
# facet
