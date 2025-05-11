library(tidyverse)
library(patchwork)

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

temp <- df %>%
    reframe(
        # type 2 error
        mixT2 = mean(zeroInDPM),
        kT2 = mean(kclustIn0),

        # coverage
        mixCov = mean(commonInDPM),
        kCov = mean(kclustInCommon),
        .by = c(alph, bet, prec)
    ) %>%
    pivot_longer(cols = c(mixT2:kCov)) %>%
    mutate(
        stat = ifelse(grepl("T2", name), "T2", "Coverage"),
        model = ifelse(grepl("k", name), "kclust", "mixDPM"),
    )
p1 <- temp %>%
    filter(stat == "T2") %>%
    ggplot(aes(x = alph, y = bet, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(
        low = "forestgreen", mid = "white", high = "red",
        midpoint = 0.75
    ) +
    facet_wrap(~prec) +
    ggtitle("Type 2 error")
p2 <- temp %>%
    filter(stat == "Coverage") %>%
    ggplot(aes(x = alph, y = bet, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(
        low = "red", mid = "white", high = "forestgreen",
        midpoint = 0.75
    ) +
    facet_wrap(~prec) +
    ggtitle("Coverage")

# mse
temp2 <- df %>%
    reframe(mseMix = sqrt(mean((common - meanBeta1)^2)), mseK = sqrt(mean((common - meanBetakclust1)^2)), .by = c(alph, bet, prec))

p3 <- temp2 %>%
    ggplot(aes(x = alph, y = bet, fill = mseMix)) +
    geom_tile() +
    scale_fill_gradient2(
        low = "forestgreen", mid = "white", high = "red",
        midpoint = 2.4
    ) +
    facet_wrap(~prec) +
    ggtitle("MSE")

p1 / p2 / p3
