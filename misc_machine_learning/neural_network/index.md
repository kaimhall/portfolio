```r
load('data_PCA_E7.RData')
summary(X)
```

    ##       Var1            Var2            Var3            Var4
    ##  Min.   :505.0   Min.   :18900   Min.   :0.073   Min.   : 12000
    ##  1st Qu.:610.0   1st Qu.:26900   1st Qu.:0.145   1st Qu.:168000
    ##  Median :645.0   Median :37200   Median :0.178   Median :219000
    ##  Mean   :635.3   Mean   :36101   Mean   :0.418   Mean   :203439
    ##  3rd Qu.:665.0   3rd Qu.:41900   3rd Qu.:0.246   3rd Qu.:229000
    ##  Max.   :725.0   Max.   :51200   Max.   :1.950   Max.   :756000

```r
x.pca<-prcomp(X)
print(x.pca)
```

    ## Standard deviations (1, .., p=4):
    ## [1] 1.196981e+05 6.207397e+03 4.548899e+01 4.669683e-01
    ##
    ## Rotation (n x k) = (4 x 4):
    ##                PC1           PC2           PC3           PC4
    ## Var1 -2.306577e-05  1.185786e-03 -9.999895e-01 -4.427279e-03
    ## Var2  4.770627e-02  9.988607e-01  1.183337e-03  2.167368e-06
    ## Var3 -3.746301e-07 -3.057227e-06  4.427278e-03 -9.999902e-01
    ## Var4 -9.988614e-01  4.770621e-02  7.960712e-05  5.808034e-07

```r
plot(x.pca,type='l')
```

![](exercise7_files/figure-markdown_github/unnamed-chunk-2-1.png)

```r
print(summary(x.pca))
```

    ## Importance of components:
    ##                              PC1       PC2   PC3   PC4
    ## Standard deviation     1.197e+05 6.207e+03 45.49 0.467
    ## Proportion of Variance 9.973e-01 2.680e-03  0.00 0.000
    ## Cumulative Proportion  9.973e-01 1.000e+00  1.00 1.000

```r
#z-score standardization
x.pca <- prcomp(X,center=TRUE,scale=TRUE)
print(x.pca)
```

    ## Standard deviations (1, .., p=4):
    ## [1] 1.3076326 1.1684277 0.7876379 0.5518154
    ##
    ## Rotation (n x k) = (4 x 4):
    ##             PC1        PC2       PC3        PC4
    ## Var1 -0.1547026 -0.7010420 0.6670785 -0.1990313
    ## Var2 -0.6832255  0.1556693 0.2086951  0.6822143
    ## Var3  0.2508370  0.6504772 0.7078286 -0.1137494
    ## Var4  0.6680949 -0.2473593 0.1021336  0.6942847

```r
plot(x.pca,type='l')
```

![](exercise7_files/figure-markdown_github/unnamed-chunk-3-1.png)

```r
print(summary(x.pca))
```

    ## Importance of components:
    ##                           PC1    PC2    PC3     PC4
    ## Standard deviation     1.3076 1.1684 0.7876 0.55182
    ## Proportion of Variance 0.4275 0.3413 0.1551 0.07613
    ## Cumulative Proportion  0.4275 0.7688 0.9239 1.00000

#NN without PCA

```r
library(nnet)
rm(X)
rm(y)
load('data_NN_E7.RData')
summary(X)
```

    ##       Var1             Var2            Var3             Var4
    ##  Min.   : 22.00   Min.   :505.0   Min.   :0.0000   Min.   :0.1830
    ##  1st Qu.: 40.00   1st Qu.:605.0   1st Qu.:0.0000   1st Qu.:0.3230
    ##  Median : 55.00   Median :630.0   Median :1.0000   Median :0.3680
    ##  Mean   : 63.63   Mean   :628.9   Mean   :0.6826   Mean   :0.3562
    ##  3rd Qu.: 80.00   3rd Qu.:655.0   3rd Qu.:1.0000   3rd Qu.:0.4120
    ##  Max.   :200.00   Max.   :725.0   Max.   :1.0000   Max.   :0.5120
    ##       Var5              Var6             Var7             Var8
    ##  Min.   :0.00500   Min.   :0.0730   Min.   :0.0120   Min.   :0.00400
    ##  1st Qu.:0.00900   1st Qu.:0.1510   1st Qu.:0.1720   1st Qu.:0.01000
    ##  Median :0.01100   Median :0.1950   Median :0.2210   Median :0.01200
    ##  Mean   :0.01249   Mean   :0.5973   Mean   :0.2181   Mean   :0.01515
    ##  3rd Qu.:0.01500   3rd Qu.:1.3710   3rd Qu.:0.2300   3rd Qu.:0.01600
    ##  Max.   :0.03000   Max.   :2.6000   Max.   :0.7640   Max.   :0.06000
    ##       Var9             Var10
    ##  Min.   :0.00000   Min.   :0.0000
    ##  1st Qu.:0.00000   1st Qu.:0.0000
    ##  Median :0.00000   Median :1.0000
    ##  Mean   :0.01709   Mean   :0.5289
    ##  3rd Qu.:0.00000   3rd Qu.:1.0000
    ##  Max.   :1.00000   Max.   :1.0000

```r
#min-max scaling

minx=apply(X,2,min)
maxx=apply(X,2,max)

miny=min(y)
maxy=max(y)

X_S=scale(X,minx,maxx-minx)
y_s=scale(y,miny,maxy-miny)
```

```r
# NN model
model_nn <- nnet(X_S, y_s, size=20,
                 maxit=300, decay=0.03, linout=TRUE, reltol=1.e-6, MaxNWts=100000)
```

    ## # weights:  241
    ## initial  value 12074.926569
    ## iter  10 value 53.040354
    ## iter  20 value 39.697988
    ## iter  30 value 34.312506
    ## iter  40 value 31.351618
    ## iter  50 value 28.171835
    ## iter  60 value 26.879735
    ## iter  70 value 25.852764
    ## iter  80 value 25.353068
    ## iter  90 value 25.059211
    ## iter 100 value 24.893339
    ## iter 110 value 24.763462
    ## iter 120 value 24.611410
    ## iter 130 value 24.505750
    ## iter 140 value 24.425072
    ## iter 150 value 24.307079
    ## iter 160 value 24.157097
    ## iter 170 value 23.999051
    ## iter 180 value 23.881802
    ## iter 190 value 23.792141
    ## iter 200 value 23.710257
    ## iter 210 value 23.643505
    ## iter 220 value 23.589224
    ## iter 230 value 23.534901
    ## iter 240 value 23.494990
    ## iter 250 value 23.469023
    ## iter 260 value 23.439473
    ## iter 270 value 23.415806
    ## iter 280 value 23.392344
    ## iter 290 value 23.275909
    ## iter 300 value 23.187084
    ## final  value 23.187084
    ## stopped after 300 iterations

```r
# produce the predictions
y_s.predict <- predict(model_nn, X_S)
print(cor(y_s.predict,y_s))
```

    ##           [,1]
    ## [1,] 0.9388577

```r
str(X)
```

    ## 'data.frame':    7196 obs. of  10 variables:
    ##  $ Var1 : num  90 90 90 90 38 38 38 38 65 90 ...
    ##  $ Var2 : num  710 710 710 710 580 580 580 580 655 620 ...
    ##  $ Var3 : num  1 1 1 1 1 1 1 1 0 1 ...
    ##  $ Var4 : num  0.463 0.463 0.463 0.463 0.247 0.247 0.247 0.247 0.428 0.337 ...
    ##  $ Var5 : num  0.011 0.011 0.011 0.011 0.013 0.013 0.013 0.013 0.011 0.008 ...
    ##  $ Var6 : num  0.151 0.151 0.151 0.151 0.208 ...
    ##  $ Var7 : num  0.027 0.027 0.027 0.027 0.219 0.219 0.219 0.219 0.172 0.226 ...
    ##  $ Var8 : num  0.011 0.011 0.011 0.011 0.01 0.01 0.01 0.01 0.008 0.016 ...
    ##  $ Var9 : num  1 1 1 1 0 0 0 0 0 0 ...
    ##  $ Var10: num  0 0 0 0 1 1 1 1 1 0 ...

# NN with PCA

```r
# remove binary variables and standardize X

miny=min(y)
maxy=max(y)
y_s=scale(y,miny,maxy-miny)

x.pca <- prcomp(X[,c(1,2,4,5,6,7,8)],center=TRUE,scale=TRUE)
pred_pca <- predict(x.pca)

print(pred_pca[1:5,])
```

    ##            PC1        PC2        PC3        PC4        PC5       PC6        PC7
    ## [1,]  2.769676  0.9674481  0.5816383 -1.7773290 -0.4168766 0.9762095 0.78995243
    ## [2,]  2.769676  0.9674481  0.5816383 -1.7773290 -0.4168766 0.9762095 0.78995243
    ## [3,]  2.769676  0.9674481  0.5816383 -1.7773290 -0.4168766 0.9762095 0.78995243
    ## [4,]  2.769676  0.9674481  0.5816383 -1.7773290 -0.4168766 0.9762095 0.78995243
    ## [5,] -1.412512 -0.0040778 -1.3465561  0.7394715 -1.0735162 0.2160057 0.03617177

```r
print(summary(x.pca))
```

    ## Importance of components:
    ##                          PC1    PC2    PC3    PC4    PC5     PC6     PC7
    ## Standard deviation     1.328 1.1657 1.1342 0.9897 0.8571 0.77783 0.52149
    ## Proportion of Variance 0.252 0.1941 0.1838 0.1399 0.1049 0.08643 0.03885
    ## Cumulative Proportion  0.252 0.4461 0.6299 0.7698 0.8747 0.96115 1.00000

```r
# binary variables are added back to the data

N=7
x_data <- data.frame(X[,c(3,9,10)],pred_pca[,1:N])
```

```r
# NN model with pca

model_pcann <- nnet(x_data, y_s, size=20,
                 maxit=300, decay=0.03, linout=TRUE, reltol=1.e-6, MaxNWts=100000)
```

    ## # weights:  241
    ## initial  value 2548.425480
    ## iter  10 value 60.234397
    ## iter  20 value 29.770896
    ## iter  30 value 25.437188
    ## iter  40 value 23.035419
    ## iter  50 value 21.986917
    ## iter  60 value 20.768992
    ## iter  70 value 20.073042
    ## iter  80 value 19.625506
    ## iter  90 value 19.356206
    ## iter 100 value 19.097833
    ## iter 110 value 18.938737
    ## iter 120 value 18.784620
    ## iter 130 value 18.635220
    ## iter 140 value 18.516993
    ## iter 150 value 18.435954
    ## iter 160 value 18.361717
    ## iter 170 value 18.281183
    ## iter 180 value 18.209129
    ## iter 190 value 18.142622
    ## iter 200 value 18.071377
    ## iter 210 value 18.010881
    ## iter 220 value 17.943529
    ## iter 230 value 17.878284
    ## iter 240 value 17.816297
    ## iter 250 value 17.788753
    ## iter 260 value 17.768241
    ## iter 270 value 17.752451
    ## iter 280 value 17.736956
    ## iter 290 value 17.719340
    ## iter 300 value 17.698402
    ## final  value 17.698402
    ## stopped after 300 iterations

```r
# predict and eval
y_s.pred <- predict(model_pcann, x_data)
print(cor(y_s.pred, y_s))
```

    ##           [,1]
    ## [1,] 0.9514857

```r
plot(y_s.pred,y_s)
```

![](exercise7_files/figure-markdown_github/unnamed-chunk-12-1.png)

[back](https://kaimhall.github.io/portfolio/misc_machine_learning)
