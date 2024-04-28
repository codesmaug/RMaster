---
title: "Model Training and tunig"
output: html_notebook
---
```{r}
str(iris)
```

```{r}
library(AppliedPredictiveModeling)
transparentTheme(trans = .4)
library(caret)
featurePlot(x = iris[, 1:4], 
            y = iris$Species, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))
```
```{r}
featurePlot(x=iris[, 1:4],
            y=iris$Species,
            plot = "ellipse",
            auto.key =list(columns =3))
```
```{r}
transparentTheme(trans = .9)
featurePlot(x = iris[, 1:4],
            y = iris$Species,
            plot = "density",
            ## Pass in options to xyplot() to 
            ##make it prettier
            scales = list(x = list(relation = "free"),
                          y = list(relation = "free")),
            adjust = 1.5,
            pch = "|",
            layout = c(4, 1),
            auto.key =list(columns = 3))
```


```{r}
featurePlot(x = iris[, 1:4],
            y = iris$Species,
            plot = "box",
            
            ##Pass in options to bwplot()
            scales = list(y = list(relation ="free"),
                          x = list(rot = 90)),
            layout = c(4, 1 ),
            autoo.key = list(columns = 2))
```



```{r}
library(mlbench)
data(BostonHousing)
regVar <- c("age", "lstat", "tax")
str(BostonHousing[, regVar])
```
```{r}
theme1 <- trellis.par.get()

theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2

trellis.par.set(theme1)
featurePlot(x = BostonHousing[,regVar],
            y = BostonHousing$medv,
            plot = "scatter",
            layout = c(3, 1))
```
```{r}
featurePlot(x = BostonHousing[, regVar], 
            y = BostonHousing$medv, 
            plot = "scatter",
            type = c("p", "smooth"),
            span = .5,
            layout = c(3, 1))
```


```{r}
library(earth)
data("etitanic")
head(model.matrix(survived ~ ., data = etitanic))
```
```{r}
dummies <- dummyVars(survived ~ ., data = etitanic)
head(predict(dummies, newdata = etitanic))
```
```{r}
data(mdrr)
data.frame(table(mdrrDescr$nR11))
```
```{r}
nzv <- nearZeroVar(mdrrDescr, saveMetrics= TRUE)
nzv[nzv$nzv,][1:10,]
```
```{r}
dim(mdrrDescr)
nzv <- nearZeroVar(mdrrDescr)
filteredDescr <- mdrrDescr[, -nzv]
dim(filteredDescr)
```
```{r}
library(caret)
set.seed(3456)
trainIndex <- createDataPartition(iris$Species,
                                  list = FALSE,
                                  times = 1)
irisTrain <- iris[ trainIndex,]
irisTest  <- iris[-trainIndex,] 
```
```{r}
library(mlbench)
data(BostonHousing)

testing <- scale(BostonHousing[, c("age", "nox")])
set.seed(5)
##A random sample of 5 data points
startSet <- sample(1:dim(testing)[1], 5)
samplePool <- testing[- startSet,]
start <- testing[startSet,]
newSamp <- maxDissim(start, samplePool, n = 20)
head(newSamp)
```
```{r}
library(mlbench)
data("Sonar")
str(Sonar[, 1:10])
```
```{r}
library(caret)
set.seed(998)

inTraning <- createDataPartition(Sonar$Class, p = .75, list = FALSE)
training  <- Sonar[ inTraning,]
testing   <- Sonar[-inTraning,]
```
```{r}
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10)
```
```{r}
set.seed(825)

gbmfit1 <- train(Class~., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 ##This last option is actually one
                 ##for gbm () that passes through
                 verbose = FALSE)
```
```{r}
gbmGrid <- expand.grid(interaction.depth = c(1, 5, 9),
                                   n.trees = (1:30)*50,
                                   shrinkage = 0.1,
                                   n.minobsinnode = 20)
nrow(gbmGrid)

set.seed(825)

gbmFit2 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 tuneGrid = gbmGrid)

gbmFit2
          
                      
```
```{r}
trellis.par.set(caretTheme())
plot(gbmFit2)
```
```{r}
trellis.par.set(caretTheme())
plot(gbmFit2, metric = "Kappa", plotType = "level",
     scales = list(x = list(rot = 90)))
```
```{r}
ggplot(gbmFit2)
```
```{r}
head(twoClassSummary)
```
```{r}
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ##Estimando probabilidade de classes
                           classProbs = TRUE,
                           ##Avaliando performance usando
                           ## a função a seguir
                           summaryFunction = twoClassSummary)
set.seed(825)
gbmFit3 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 tuneGrid = gbmGrid,
                 ##Especificando qual a métrica
                 ## a ser otimizada
                 metric = "ROC")
gbmFit3
```
```{r}
whichTwoPct <- tolerance(gbmFit3$results, metric = "ROC",
                         tol = 2, maximize = TRUE)
cat("best model within 2 pct of best :\n")

gbmFit3$results[whichTwoPct, 1:6]
```
```{r}
predict(gbmFit3, newdata = head(testing), type = "prob")
```
```{r}
trellis.par.set(caretTheme())
densityplot(gbmFit3, pch = "|", resamples = "all")
```
```{r}
set.seed(825)

svmFit <- train(Class ~., data = training,
                method = "svmRadial", 
                trControl = fitControl,
                preProc = c("center", "scale"),
                tuneLength = 8,
                metric = "ROC")
svmFit
```
```{r}
set.seed(825)
rdaFit <- train(Class ~ ., data = training, 
                 method = "rda", 
                 trControl = fitControl, 
                 tuneLength = 4,
                 metric = "ROC")
rdaFit     
```
```{r}
resamps <- resamples(list(GBM = gbmFit3,
                          SVM = svmFit,
                          RDA = rdaFit))
resamps
```
```{r}
summary(resamps)
```
```{r}
theme1 <- trellis.par.get()

theme1$plot.symbol$col = rgb(.2, .2, .2, .4)

theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps, layout = c(3, 1))
```
```{r}
trellis.par.set(caretTheme())
dotplot(resamps, metric = "ROC")
```
```{r}
trellis.par.set(theme1)
xyplot(resamps, what = "BlandAltman")
```
```{r}
splom(resamps)
```

