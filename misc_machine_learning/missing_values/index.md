In this project I research some methods to mitigate negative impacts of missing values.
Different imputation methods are experimented on, and effect on mean is explored.

```r
iris_data <- read.table("iris.csv", header = TRUE, sep = ";", dec = ",")
head(iris_data)
```

    ##   Petal.Length Petal.Width Species
    ## 1          1.4         0.2  setosa
    ## 2          1.7         0.4  setosa
    ## 3          1.5          NA  setosa
    ## 4          1.5         0.4  setosa
    ## 5          1.7         0.2  setosa
    ## 6          1.6         0.2  setosa

```r
plot(iris_data$Petal.Width,iris_data$Petal.Length)
```

![](missing-values_files/figure-markdown_github/cars-1.png)

Fitting linear model.

```r
lm_fit <- lm(Petal.Width ~ Petal.Length, data = iris_data)
summary(lm_fit)
```

    ##
    ## Call:
    ## lm(formula = Petal.Width ~ Petal.Length, data = iris_data)
    ##
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max
    ## -0.3518 -0.1518  0.0254  0.1391  0.3810
    ##
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)  -0.46809    0.10416  -4.494    2e-04 ***
    ## Petal.Length  0.45906    0.02738  16.769 1.23e-13 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ##
    ## Residual standard error: 0.2149 on 21 degrees of freedom
    ##   (7 observations deleted due to missingness)
    ## Multiple R-squared:  0.9305, Adjusted R-squared:  0.9272
    ## F-statistic: 281.2 on 1 and 21 DF,  p-value: 1.231e-13

```r
lm_fit$coefficients
```

    ##  (Intercept) Petal.Length
    ##   -0.4680856    0.4590629

Here mean can’t be calculated with NA values.

```r
mean(iris_data$Petal.Length)
```

    ## [1] NA

```r
mean(iris_data$Petal.Width)
```

    ## [1] NA

```r
summary(iris_data)
```

    ##   Petal.Length  Petal.Width      Species
    ##  Min.   :1.2   Min.   :0.200   Length:30
    ##  1st Qu.:1.6   1st Qu.:0.375   Class :character
    ##  Median :4.2   Median :1.350   Mode  :character
    ##  Mean   :3.4   Mean   :1.289
    ##  3rd Qu.:4.7   3rd Qu.:2.025
    ##  Max.   :5.7   Max.   :2.500
    ##  NA's   :5     NA's   :2

We have set omission as method of dealing with missing values. I iris
data we have 23 complete observations -\> no missing values

```r
getOption('na.action')
```

    ## [1] "na.omit"

```r
sum(complete.cases(iris_data))
```

    ## [1] 23

with na removal.

```r
(mean_pl <- mean(iris_data$Petal.Length, na.rm = TRUE))
```

    ## [1] 3.4

```r
(mean_pw <- mean(iris_data$Petal.Width, na.rm = TRUE))
```

    ## [1] 1.289286

No na removal, only complete rows, complete pairwise

```r
cov(iris_data$Petal.Length, iris_data$Petal.Width) #returns NA
```

    ## [1] NA

```r
cov(iris_data$Petal.Length, iris_data$Petal.Width, use = "complete.obs") # case wise deletion
```

    ## [1] 1.286047

```r
cov(iris_data$Petal.Length, iris_data$Petal.Width, use = "pairwise.complete.obs") # calculated using all pair
```

    ## [1] 1.286047

mean imputation for missing values.

```r
mean_imputation <- iris_data

idx_pw_na <- is.na(mean_imputation$Petal.Width) #find na
mean_imputation$Petal.Width[idx_pw_na] <- mean(mean_pw)
mean_pw_meanimpute <- mean(mean_imputation$Petal.Width)

idx_pl_na <- is.na(mean_imputation$Petal.Length) #find na
mean_imputation$Petal.Length[idx_pl_na] <- mean(mean_pl)
mean_pl_meanimpute <- mean(mean_imputation$Petal.Length)
mean_pl_meanimpute
```

    ## [1] 3.4

stratified mean imputation for missing values.

```r
strat_mean_imputation <- iris_data

#petal width
mean_pw_setosa <- mean(subset(iris_data, Species == "setosa")$Petal.Width, na.rm = TRUE)
strat_mean_imputation$Petal.Width[is.na(strat_mean_imputation$Petal.Width) & strat_mean_imputation$Species == "setosa"] <- mean_pw_setosa

mean_pw_virginica <- mean(subset(iris_data, Species == "virginica")$Petal.Width, na.rm = TRUE)
strat_mean_imputation$Petal.Width[is.na(strat_mean_imputation$Petal.Width) & strat_mean_imputation$Species == "virginica"] <- mean_pw_virginica

mean_pw_versicolor <- mean(subset(iris_data, Species == "versicolor")$Petal.Width, na.rm = TRUE)
strat_mean_imputation$Petal.Width[is.na(strat_mean_imputation$Petal.Width) & strat_mean_imputation$Species == "versicolor"] <- mean_pw_versicolor

summary(strat_mean_imputation)
```

    ##   Petal.Length  Petal.Width      Species
    ##  Min.   :1.2   Min.   :0.200   Length:30
    ##  1st Qu.:1.6   1st Qu.:0.325   Class :character
    ##  Median :4.2   Median :1.306   Mode  :character
    ##  Mean   :3.4   Mean   :1.256
    ##  3rd Qu.:4.7   3rd Qu.:1.975
    ##  Max.   :5.7   Max.   :2.500
    ##  NA's   :5

```r
mean_pw_strat <- mean(strat_mean_imputation$Petal.Width)

#petal length
mean_pl_virginica <- mean(subset(iris_data, Species == "virginica")$Petal.Length, na.rm = TRUE)
strat_mean_imputation$Petal.Length[is.na(strat_mean_imputation$Petal.Length) & strat_mean_imputation$Species == "virginica"] <- mean_pl_virginica

summary(strat_mean_imputation)
```

    ##   Petal.Length    Petal.Width      Species
    ##  Min.   :1.200   Min.   :0.200   Length:30
    ##  1st Qu.:1.625   1st Qu.:0.325   Class :character
    ##  Median :4.400   Median :1.306   Mode  :character
    ##  Mean   :3.730   Mean   :1.256
    ##  3rd Qu.:5.275   3rd Qu.:1.975
    ##  Max.   :5.700   Max.   :2.500

```r
mean_pl_strat <-  mean(strat_mean_imputation$Petal.Length)
mean_pl_strat
```

    ## [1] 3.73

regression imputation for na values.

```r
regr_imputation <- iris_data
summary(regr_imputation)
```

    ##   Petal.Length  Petal.Width      Species
    ##  Min.   :1.2   Min.   :0.200   Length:30
    ##  1st Qu.:1.6   1st Qu.:0.375   Class :character
    ##  Median :4.2   Median :1.350   Mode  :character
    ##  Mean   :3.4   Mean   :1.289
    ##  3rd Qu.:4.7   3rd Qu.:2.025
    ##  Max.   :5.7   Max.   :2.500
    ##  NA's   :5     NA's   :2

```r
#petal width
subset(regr_imputation,is.na(Petal.Width))
```

    ##    Petal.Length Petal.Width    Species
    ## 3           1.5          NA     setosa
    ## 18          4.5          NA versicolor

```r
# fit the linear model and impute predictions

lm_fit_width <- lm(Petal.Width ~ Petal.Length, data = iris_data)
regr_imputation$Petal.Width[is.na(regr_imputation$Petal.Width)] <- predict(lm_fit_width, subset(regr_imputation, is.na(Petal.Width)))

mean_pw_regr <- mean(regr_imputation$Petal.Width)
mean_pw_regr
```

    ## [1] 1.26394

```r
#petal length
subset(regr_imputation,is.na(Petal.Length))
```

    ##    Petal.Length Petal.Width   Species
    ## 21           NA         2.5 virginica
    ## 22           NA         2.1 virginica
    ## 26           NA         1.8 virginica
    ## 27           NA         1.9 virginica
    ## 28           NA         2.3 virginica

```r
lm_fit_length <- lm(Petal.Length ~ Petal.Width, data = iris_data)
regr_imputation$Petal.Length[is.na(regr_imputation$Petal.Length)] <- predict(lm_fit_length, subset(regr_imputation, is.na(Petal.Length)))
summary(regr_imputation)
```

    ##   Petal.Length    Petal.Width      Species
    ##  Min.   :1.200   Min.   :0.200   Length:30
    ##  1st Qu.:1.625   1st Qu.:0.325   Class :character
    ##  Median :4.400   Median :1.350   Mode  :character
    ##  Mean   :3.747   Mean   :1.264
    ##  3rd Qu.:5.085   3rd Qu.:1.975
    ##  Max.   :6.255   Max.   :2.500

```r
mean_pl_regr <- mean(regr_imputation$Petal.Length)
mean_pl_regr
```

    ## [1] 3.747445

maximum likelihood for na values.

```r
if (!require(mclust)) {
  install.packages("mclust")
  require(mclust)
}
```

    ## Loading required package: mclust

    ## Package 'mclust' version 5.4.10
    ## Type 'citation("mclust")' for citing this R package in publications.

```r
iris_ml <- iris_data
trainingdata <- iris_data[complete.cases(iris_data),]

dens <- densityMclust(trainingdata[,c(1,2)])
```

![](missing-values_files/figure-markdown_github/unnamed-chunk-9-1.png)

```r
summary(dens, parameters = TRUE)
```

    ## -------------------------------------------------------
    ## Density estimation via Gaussian finite mixture modeling
    ## -------------------------------------------------------
    ##
    ## Mclust EEE (ellipsoidal, equal volume, shape and orientation) model with 5
    ## components:
    ##
    ##  log-likelihood  n df       BIC       ICL
    ##        3.332066 23 17 -46.63927 -46.65523
    ##
    ## Mixing probabilities:
    ##          1          2          3          4          5
    ## 0.39130435 0.26068872 0.08713737 0.08692027 0.17394930
    ##
    ## Means:
    ##                   [,1]     [,2]     [,3]     [,4]     [,5]
    ## Petal.Length 1.4888889 4.433393 3.651447 4.949938 5.449927
    ## Petal.Width  0.2666667 1.316722 1.050387 1.899959 2.324932
    ##
    ## Variances:
    ## [,,1]
    ##              Petal.Length Petal.Width
    ## Petal.Length  0.028012791 0.005388134
    ## Petal.Width   0.005388134 0.006131798
    ## [,,2]
    ##              Petal.Length Petal.Width
    ## Petal.Length  0.028012791 0.005388134
    ## Petal.Width   0.005388134 0.006131798
    ## [,,3]
    ##              Petal.Length Petal.Width
    ## Petal.Length  0.028012791 0.005388134
    ## Petal.Width   0.005388134 0.006131798
    ## [,,4]
    ##              Petal.Length Petal.Width
    ## Petal.Length  0.028012791 0.005388134
    ## Petal.Width   0.005388134 0.006131798
    ## [,,5]
    ##              Petal.Length Petal.Width
    ## Petal.Length  0.028012791 0.005388134
    ## Petal.Width   0.005388134 0.006131798

```r
#plot(dens, what = "density", data = trainingdata[,c(1,2)])
#plot(dens, what = "density", type = "persp")

summary(iris_ml)
```

    ##   Petal.Length  Petal.Width      Species
    ##  Min.   :1.2   Min.   :0.200   Length:30
    ##  1st Qu.:1.6   1st Qu.:0.375   Class :character
    ##  Median :4.2   Median :1.350   Mode  :character
    ##  Mean   :3.4   Mean   :1.289
    ##  3rd Qu.:4.7   3rd Qu.:2.025
    ##  Max.   :5.7   Max.   :2.500
    ##  NA's   :5     NA's   :2

```r
#width estimates
widths <- seq(0,4,0.1)

lengths <- rep(iris_data$Petal.Length[18], length(widths))
ml_est <- seq(0,4,0.1)[which.max(predict(dens, data.frame(Petal.Length = lengths, Petal.Width = widths)))]
iris_ml$Petal.Width[18] <- ml_est

lengths <- rep(iris_data$Petal.Length[3], length(widths))
ml_est <- seq(0,4,0.1)[which.max(predict(dens, data.frame(Petal.Length = lengths, Petal.Width = widths)))]
iris_ml$Petal.Width[3] <- ml_est

iris_ml
```

    ##    Petal.Length Petal.Width    Species
    ## 1           1.4         0.2     setosa
    ## 2           1.7         0.4     setosa
    ## 3           1.5         0.3     setosa
    ## 4           1.5         0.4     setosa
    ## 5           1.7         0.2     setosa
    ## 6           1.6         0.2     setosa
    ## 7           1.6         0.2     setosa
    ## 8           1.2         0.2     setosa
    ## 9           1.3         0.3     setosa
    ## 10          1.4         0.3     setosa
    ## 11          4.7         1.4 versicolor
    ## 12          4.5         1.3 versicolor
    ## 13          3.5         1.0 versicolor
    ## 14          4.4         1.4 versicolor
    ## 15          4.8         1.8 versicolor
    ## 16          4.4         1.4 versicolor
    ## 17          3.8         1.1 versicolor
    ## 18          4.5         1.3 versicolor
    ## 19          4.4         1.2 versicolor
    ## 20          4.2         1.2 versicolor
    ## 21           NA         2.5  virginica
    ## 22           NA         2.1  virginica
    ## 23          5.1         2.0  virginica
    ## 24          5.3         2.3  virginica
    ## 25          5.7         2.3  virginica
    ## 26           NA         1.8  virginica
    ## 27           NA         1.9  virginica
    ## 28           NA         2.3  virginica
    ## 29          5.6         2.4  virginica
    ## 30          5.2         2.3  virginica

```r
mean_pw_ml <- mean(iris_ml$Petal.Width)
mean_pw_ml
```

    ## [1] 1.256667

```r
#Length estimates
lengths <- seq(1,7,0.1)

widths <- rep(iris_data$Petal.Width[21], length(lengths))
ml_est <- seq(1,7,0.1)[which.max(predict(dens, data.frame(Petal.Length = lengths, Petal.Width = widths)))]
iris_ml$Petal.Length[21] <- ml_est

widths <- rep(iris_data$Petal.Width[22], length(lengths))
ml_est <- seq(1,7,0.1)[which.max(predict(dens, data.frame(Petal.Length = lengths, Petal.Width = widths)))]
iris_ml$Petal.Length[22] <- ml_est

widths <- rep(iris_data$Petal.Width[26], length(lengths))
ml_est <- seq(1,7,0.1)[which.max(predict(dens, data.frame(Petal.Length = lengths, Petal.Width = widths)))]
iris_ml$Petal.Length[26] <- ml_est

widths <- rep(iris_data$Petal.Width[27], length(lengths))
ml_est <- seq(1,7,0.1)[which.max(predict(dens, data.frame(Petal.Length = lengths, Petal.Width = widths)))]
iris_ml$Petal.Length[27] <- ml_est

widths <- rep(iris_data$Petal.Width[28], length(lengths))
ml_est <- seq(1,7,0.1)[which.max(predict(dens, data.frame(Petal.Length = lengths, Petal.Width = widths)))]
iris_ml$Petal.Length[28] <- ml_est

iris_ml
```

    ##    Petal.Length Petal.Width    Species
    ## 1           1.4         0.2     setosa
    ## 2           1.7         0.4     setosa
    ## 3           1.5         0.3     setosa
    ## 4           1.5         0.4     setosa
    ## 5           1.7         0.2     setosa
    ## 6           1.6         0.2     setosa
    ## 7           1.6         0.2     setosa
    ## 8           1.2         0.2     setosa
    ## 9           1.3         0.3     setosa
    ## 10          1.4         0.3     setosa
    ## 11          4.7         1.4 versicolor
    ## 12          4.5         1.3 versicolor
    ## 13          3.5         1.0 versicolor
    ## 14          4.4         1.4 versicolor
    ## 15          4.8         1.8 versicolor
    ## 16          4.4         1.4 versicolor
    ## 17          3.8         1.1 versicolor
    ## 18          4.5         1.3 versicolor
    ## 19          4.4         1.2 versicolor
    ## 20          4.2         1.2 versicolor
    ## 21          5.6         2.5  virginica
    ## 22          5.2         2.1  virginica
    ## 23          5.1         2.0  virginica
    ## 24          5.3         2.3  virginica
    ## 25          5.7         2.3  virginica
    ## 26          4.9         1.8  virginica
    ## 27          4.9         1.9  virginica
    ## 28          5.4         2.3  virginica
    ## 29          5.6         2.4  virginica
    ## 30          5.2         2.3  virginica

```r
mean_pl_ml <- mean(iris_ml$Petal.Length)
mean_pl_ml
```

    ## [1] 3.7

[back](https://kaimhall.github.io/portfolio/misc_machine_learning)
