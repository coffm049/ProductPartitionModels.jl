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


temp <- df %>%
    reframe(
        # type 2 error
        mixT2 = mean(zeroInDPM, na.rm = T),
        kT2 = mean(kclustIn0, na.rm = T),
        slrT2 = mean(zeroInSLR, na.rm = T),

        # coverage
        mixCov = mean(commonInDPM, na.rm = T),
        kCov = mean(kclustInCommon, na.rm = T),
        slrCov = mean(commonInSLR, na.rm = T),

        # bias
        biasMix = mean(common - meanBeta1, na.rm = T),
        biasK = mean(common - meanBetakclust1, na.rm = T),
        biasSLR = mean(common - meanBetaSLR, na.rm = T),

        # MSE
        mseMix = sqrt(mean((common - meanBeta1)^2, na.rm = T)),
        mseK = sqrt(mean((common - meanBetakclust1)^2, na.rm = T)),
        mseSLR = sqrt(mean((common - meanBetaSLR)^2, na.rm = T)),
        .by = c(common)
    ) %>%
    drop_na(common)

# common
df %>%
    pivot_longer(c(
        meanBeta1, meanBetakclust1, meanBetaSLR
    )) %>%
    select(name, value, common) %>%
    drop_na() %>%
    ggplot(aes(x = value)) +
    geom_density() +
    facet_grid(rows = vars(name), cols = vars(common), scales = "free")

# bias
temp %>%
    select(common, starts_with("bias"))


# MSE
temp %>%
    select(common, starts_with("mse"))

# T2
temp %>%
    select(common, contains("T2"))

# Cov
temp %>%
    select(common, contains("Cov"))
