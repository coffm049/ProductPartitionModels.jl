library(tidyverse)
library(patchwork)


df <- data.frame("file" = c(
    "c8_inter2.0_common2.0_xd0.0_v1.0_dim1_prec0.01alph1.0bet0.1.csv",
    "c8_inter2.0_common3.0_xd0.0_v1.0_dim1_prec0.01alph1.0bet0.1.csv"
)) %>%
    # map read these csvs into nested dataframe
    mutate(
        data = map(file, read_csv,
            col_types = cols(
                .default = col_double(),
                zeroInDPM = col_logical()
            )
        )
    ) %>%
    # unnest them
    unnest(data)
# group by file


plot(c(1, 2, 3), c(1, 2, 3))

# vs alpha, beta
df %>%
    pivot_longer(c(
        rind_Mix, adjrind_Mix, midMix, meanBeta1, rind_DPM, adjrind_DPM,
        midDPM, adjrind_K, rind_K, kmean_MSE, meanBetakclust1
    )) %>%
    select(name, value, common) %>%
    drop_na() %>%
    ggplot(aes(x = value)) +
    geom_density() +
    ggh4x::facet_grid2(rows = vars(name), cols = vars(common), scales = "free", independent = "all")



temp <- df %>%
    reframe(
        # type 2 error
        mixT2 = mean(zeroInDPM),
        kT2 = mean(kclustIn0),
        slrT2 = mean(zeroInSLR),

        # coverage
        mixCov = mean(commonInDPM),
        kCov = mean(kclustInCommon),
        slrCov = mean(commonInSLR),
        .by = c(common)
    ) %>%
    pivot_longer(cols = c(mixT2:kCov)) %>%
    mutate(
        stat = ifelse(grepl("T2", name), "T2", "Coverage"),
        model = case_when(
            grepl("k", name) ~ "kclust",
            grepl("mix", name) ~ "mixDPM",
            grepl("slr", name) ~ "slr",
        )
    ) %>%
    drop_na(common)
p1 <- temp %>%
    
    filter(stat == "T2") %>%
    ggplot(aes(x = value)) +
    geom_density() +
    facet_wrap(~common) +
    ggtitle("Type 2 error")
p2 <- temp %>%
    filter(stat == "Coverage") %>%
    ggplot(aes(x = alph, y = bet, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(
        low = "red", mid = "white", high = "forestgreen",
        midpoint = 0.85
    ) +
    facet_wrap(~prec) +
    ggtitle("Coverage")

# mse
temp2 <- df %>%
    reframe(
        mseMix = sqrt(mean((common - meanBeta1)^2)),
        mseK = sqrt(mean((common - meanBetakclust1)^2)),
        mseSLR = sqrt(mean((common - meanBetaSLR)^2)),
        .by = c(common)
    )

p3 <- temp2 %>%
    ggplot(aes(x = alph, y = bet, fill = mseMix)) +
    geom_tile() +
    scale_fill_gradient2(
        low = "forestgreen", mid = "white", high = "red",
        midpoint = 3
    ) +
    facet_wrap(~prec) +
    ggtitle("MSE")

p1 / p2 / p3

ggsave("tuningResults.png")
