#Machine Learning Models for TCP Prediction 4/25/22 ####

# load the required packages
library(caret) # version 6.0-86
library(data.table)
library(ggplot2)
library(randomForest)
library(ranger)
library(dplyr)
library(tidyverse) # version 1.3.0
library(gbm)
library(writexl)
library(xgboost) # version 1.2.0.1 required to reproduce model results exactly
library(hydroGOF) # version 0.4-0
library(recipes) # version 0.1.15
library(raster) # version 3.4-5

# Read in df 
setwd("C:/Users/bhauptman/Box/Ch2/R")
df1000 <- read.csv(file = 'data/EV1000_HH.csv')
df1000

# Take a look at df structure and summary
str(df1000)
summary(df1000)
dim(df1000)

# Count number of rows with TCP level == 0
colSums(df1000 == 0)

# Find annoying NA values in df
colSums(is.na(df1000))
colnames(df1000)

#Remove ID column #1 and other uneeded columns
df1000[c(1, 2, 3, 4)] <- NULL
colnames(df)
dim(df1000)

# Remove any NA
df1000[df1000 == "?"] <- NA 
df1000 <- df1000[!(df1000$Age %in% c(NA)),]
df1000 <- df1000[!(df1000$Drainage %in% c(NA)),]
df1000 <- df1000[!(df1000$HSG %in% c(NA)),]
df1000 <- df1000[!(df1000$LU2005 %in% c(NA)),]
df1000 <- df1000[!(df1000$LU1990 %in% c(NA)),]
df1000 <- df1000[!(df1000$LU1975 %in% c(NA)),]
df1000 <- df1000[!(df1000$LU1960 %in% c(NA)),]
df1000 <- df1000[!(df1000$LU1945 %in% c(NA)),]
df1000 <- df1000[!(df1000$Precip2005 %in% c(NA)),]
df1000 <- df1000[!(df1000$Precip1990 %in% c(NA)),]
df1000 <- df1000[!(df1000$Precip1975 %in% c(NA)),]
df1000 <- df1000[!(df1000$Precip1960 %in% c(NA)),]
df1000 <- df1000[!(df1000$Irrig2005 %in% c(NA)),]
df1000 <- df1000[!(df1000$Irrig1990 %in% c(NA)),]
df1000 <- df1000[!(df1000$Irrig1975 %in% c(NA)),]
df1000 <- df1000[!(df1000$Irrig1960 %in% c(NA)),]
df1000 <- df1000[!(df1000$NO3_180 %in% c(NA)),]
df1000 <- df1000[!(df1000$NO3_400 %in% c(NA)),]
df1000 <- df1000[!(df1000$DO5 %in% c(NA)),]
df1000 <- df1000[!(df1000$DO2 %in% c(NA)),]

dim(df1000)
any(is.na(df1000))

# Make categorical df into factors
as.data.frame(df1000)

df1000 <- transform(
  df1000,
  LU19410=as.factor(LU19410),
  LU1960=as.factor(LU1960),
  LU19710=as.factor(LU19710),
  LU1990=as.factor(LU1990),
  LU20010=as.factor(LU20010),
  HSG=as.factor(HSG),
  Drainage=as.factor(Drainage)
)
dim(df1000)
str(df1000)

# df Partition - split df in train and test set (80/20 split)
set.seed(123)

# Create partition index to split df into training and testing set - next step follows...
index <- createDataPartition(df1000$logTCP, p = .80, list = FALSE)

# Subset that df with index from above into a train and test object
dftrain1000 <- df1000[index,]
dftest1000 <- df1000[-index,]

write.csv(dftrain1000, 'C:/Users/bhauptman/Box/Ch2/R/data/dftrain1000.csv')
write.csv(dftest1000, 'C:/Users/bhauptman/Box/Ch2/R/data/dftest1000.csv')

#check the dimensions of the test and training and look at distribution
dim(dftrain1000)
dim(dftest1000)

# Create a plot of training and testing (copy first) just to make sure their distributions are similar
hist_dftrain1000 <- copy(dftrain1000)
hist_dftest1000 <- copy(dftest1000)

hist_dftrain1000$set <- 'training'
hist_dftest1000$set <- 'testing'

dfgg1000 <- rbind(hist_dftrain1000, hist_dftest1000)
df <- dfgg1000 %>% mutate(TCP = 10^(logTCP)) %>% select(TCP, set)

#graph test and training data raw
ggplot(df, aes(x = TCP, fill = set)) + 
  geom_histogram(bins = 100) + 
  xlim(0,1) + 
  ylim(0, 200) + 
  geom_density(alpha = 0.5) + 
  labs(x="TCP concentration (ppb)", y="count", set="Data") + 
  theme_bw() +
  theme(legend.title=element_blank()) + 
  scale_fill_discrete(labels=c("Training data (n=10,652; 80%)", "Testing data (n=2,662; 20%)"))

#graph test and training data log
ggplot(dfgg1000, aes(x = logTCP, fill = set)) + 
  geom_histogram(bins = 100) + 
  xlim(-3,2) + geom_density(alpha = 0.1)

#graph test and training data log
ggplot(dfgg1000, aes(x = set, y = logTCP, fill = set)) + 
  geom_boxplot() + ylim(-3,2)

##################################################################################################
trainctrl <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3)
################################################################################################

####1 CART MODEL with Caret##################################################################
set.seed(111)

#tunable 
modelLookup("rpart")

# fit the model
rpart_tree1000 <- train(logTCP ~ ., 
                        data=dftrain1000, 
                        method="rpart", 
                        tuneLength = 500,   #automatically finding the best cp value and tuning model with that value
                        trControl = trainctrl)

#cp = 0.0009897651.

rpart_train1000 <- predict(rpart_tree1000, newdata=dftrain1000)
rpart_test1000 <- predict(rpart_tree1000, newdata=dftest1000)

plot(rpart_tree1000)  #optimal cp = 0.0009897651.

#save predictions
write.csv(rpart_train1000, "C:/Users/bhauptman/Box/Ch2/R/data/rpart_train1000.csv")
write.csv(rpart_test1000, "C:/Users/bhauptman/Box/Ch2/R/data/rpart_test1000.csv")

#save model to file
saveRDS(rpart_tree1000, "FinalModels/rpart_tree1000.Rds")

# to read model and training later 
rpart_tree1000 <- readRDS("FinalModels/rpart_tree1000.Rds")

####2 RANDOM FOREST MODEL with CARET ##################################################################
set.seed(222)

#to see which parameters can be tuned
#modelLookup("rf")

#create a grid with different hyperparmeters the only tunable parmeter in rf caret is mtry
rftunegrid1000 <- expand.grid(mtry=5) #best is 5

#Now train with above hyperparameters
rfcaret_1000 <- train(logTCP ~ ., method="rf", data = dftrain1000, tuneGrid = rftunegrid1000, trControl = trainctrl)
rfcaret_1000

varImp(rfcaret_tree1000)

#Now predict to train and test data
rfcaret_train1000 <- predict(rfcaret_1000, newdata = dftrain1000)
rfcaret_train1000

rfcaret_test1000 <- predict(rfcaret_1000, newdata = dftest1000)
rfcaret_test1000

write.csv(rfcaret_test1000, "C:/Users/bhauptman/Box/Ch2/R/data/rfcaret_test1000.csv")
write.csv(rfcaret_train1000, "C:/Users/bhauptman/Box/Ch2/R/data/rfcaret_train1000.csv")

rfcaret_test1000
plot(10^(rfcaret_test1000), 10^(dftest1000$logTCP))
obs_pred <- data.frame(pred = 10^(rfcaret_test1000), obs = 10^(dftest1000$logTCP))

obs_predPlot <- ggplot(obs_pred, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")

obs_predPlot

#save model to file
saveRDS(rfcaret_1000, "FinalModels/rfcaret_1000.Rds")

# to read later 
#rfCaret <- readRDS("FinalModels/rfcaret_1000.Rds")

#Since this is the best model we will also look at a variable importance table
caretVarImp <- varImp(rfcaret_1000)
plot(caretVarImp)

imp <- do.call(rbind.data.frame, caretVarImp)
imp

####3 xgb GRADIENT BOOSTED MODEL with caret##################################################################
set.seed(444)

modelLookup("gbm")

gbm_bestgrid1000 <- expand.grid(n.trees = c(800),
                            interaction.depth = c(10, 12),
                            shrinkage = c(0.1, 0.3),
                            n.minobsinnode = c(3,5))

gbm_tree1000 <- train(logTCP ~ .,
                      data=dftrain1000,
                      method="gbm",
                      distribution="gaussian",
                      trControl = trainctrl,
                      tuneGrid = gbm_bestgrid1000,
                      verbose = FALSE)
gbm_tree1000

gbm_train1000 <- predict(gbm_tree1000, newdata = dftrain1000)
gbm_test1000 <- predict(gbm_tree1000, newdata = dftest1000)

write.csv(gbm_train1000, "C:/Users/bhauptman/Box/Ch2/R/data/gbm_train1000.csv")
write.csv(gbm_test1000, "C:/Users/bhauptman/Box/Ch2/R/data/gbm_test1000.csv")

#save file
saveRDS(gbm_tree1000, "FinalModels/gbm_tree1000.Rds")

# to read later 
#gbm <- readRDS("FinalModels/gbm_tree1000.Rds")

#### 4 xgb_tree in CARET package ##################################################################  
set.seed(888)

# Tune model xgboost in Caret lets just start with some simple and suggested parameters here: https://www.kaggle.com/code/pelkoja/visual-xgboost-tuning-with-caret/report
#We going to start the tuning "the bigger knobs" by setting up the maximum number of trees:
  
nrounds <- 1000

tune_grid <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)
  
  # train  the model with the best parameters
  xgb_tune <- caret::train(
    x = as.matrix(dftrain1000[, -1]),
    y = as.numeric(dftrain1000$logTCP),
    trControl = tune_control,
    tuneGrid = tune_grid,
    method = "xgbTree",
    verbosity = 0
  )
  
  # helper function for the plots
  tuneplot <- function(x, probs = .90) {
    ggplot(x) +
      coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
      theme_bw()
  }
  
  tuneplot(xgb_tune)
  xgb_tune$bestTune
  
#Try again now tuning Maximum Depth and Minimum Child Weight
  tune_grid2 <- expand.grid(
    nrounds = seq(from = 50, to = nrounds, by = 50),
    eta = xgb_tune$bestTune$eta,
    max_depth = ifelse(xgb_tune$bestTune$max_depth == 2,
                       c(xgb_tune$bestTune$max_depth:4),
                       xgb_tune$bestTune$max_depth - 1:xgb_tune$bestTune$max_depth + 1),
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = c(1, 2, 3),
    subsample = 1
  )
  
  xgb_tune2 <- caret::train(
    x = as.matrix(dftrain1000[, -1]),
    y = as.numeric(dftrain1000$logTCP),
    trControl = tune_control,
    tuneGrid = tune_grid2,
    method = "xgbTree",
    verbosity = 0
  )
  
  tuneplot(xgb_tune2)
  xgb_tune2$bestTune
  
# now tune column row and sampling
  tune_grid3 <- expand.grid(
    nrounds = seq(from = 50, to = nrounds, by = 50),
    eta = xgb_tune$bestTune$eta,
    max_depth = xgb_tune2$bestTune$max_depth,
    gamma = 0,
    colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
    min_child_weight = xgb_tune2$bestTune$min_child_weight,
    subsample = c(0.5, 0.75, 1.0)
  )
  
  xgb_tune3 <- caret::train(
    x = as.matrix(dftrain1000[, -1]),
    y = as.numeric(dftrain1000$logTCP),
    trControl = tune_control,
    tuneGrid = tune_grid3,
    method = "xgbTree",
    verbosity = 0
  )
  
  tuneplot(xgb_tune3, probs = .95)
  xgb_tune3$bestTune
  
  
#now gamma
  tune_grid4 <- expand.grid(
    nrounds = seq(from = 50, to = nrounds, by = 50),
    eta = xgb_tune$bestTune$eta,
    max_depth = xgb_tune2$bestTune$max_depth,
    gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
    colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
    min_child_weight = xgb_tune2$bestTune$min_child_weight,
    subsample = xgb_tune3$bestTune$subsample
  )
  
  xgb_tune4 <- caret::train(
    x = as.matrix(dftrain1000[, -1]),
    y = as.numeric(dftrain1000$logTCP),
    trControl = tune_control,
    tuneGrid = tune_grid4,
    method = "xgbTree",
    verbosity = 0
  )
  
  tuneplot(xgb_tune4)
  
#Now, we have tuned the hyperparameters and can start reducing the learning rate to get to the final model:
  tune_grid5 <- expand.grid(
    nrounds = seq(from = 100, to = 10000, by = 100),
    eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
    max_depth = xgb_tune2$bestTune$max_depth,
    gamma = xgb_tune4$bestTune$gamma,
    colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
    min_child_weight = xgb_tune2$bestTune$min_child_weight,
    subsample = xgb_tune3$bestTune$subsample
  )
  
  xgb_tune5 <- caret::train(
    x = as.matrix(dftrain1000[, -1]),
    y = as.numeric(dftrain1000$logTCP),
    trControl = tune_control,
    tuneGrid = tune_grid5,
    method = "xgbTree",
    verbosity = 0
  )
  
  tuneplot(xgb_tune5)
  xgb_tune5$bestTune
  
  #now the final model
  final_grid <- expand.grid(
    nrounds = xgb_tune5$bestTune$nrounds,
    eta = xgb_tune5$bestTune$eta,
    max_depth = xgb_tune5$bestTune$max_depth,
    gamma = xgb_tune5$bestTune$gamma,
    colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
    min_child_weight = xgb_tune5$bestTune$min_child_weight,
    subsample = xgb_tune5$bestTune$subsample
  )
  
  xgb_model <- caret::train(
    x = as.matrix(dftrain1000[, -1]),
    y = as.numeric(dftrain1000$logTCP),
    trControl = tune_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbosity = 0
  )
  
  #save file
  saveRDS(xgb_model, "FinalModels/xgb_model.Rds")
  
  # to read later 
  #xgb <- readRDS("FinalModels/xgb_train_1.Rds")
  
  xgb_model$bestTune
  
  #write and/or load the model to a file and to load file next time: 
  # save(xgb_train_1, file= "xgb_train_1000.rda")
  # load(file = "xgb_train_1000.rda")
  
  #xgbCaret_grid_1$best
  
  xgb_train1000 <- predict(xgb_model, newdata = dftrain1000[, -1])
  xgb_train1000
  xgb_test1000 <- predict(xgb_model, newdata = dftest1000[, -1])
  xgb_test1000
  
  write.csv(xgb_train1000, "C:/Users/bhauptman/Box/Ch2/R/data/xgb_train1000.csv")
  write.csv(xgb_test1000, "C:/Users/bhauptman/Box/Ch2/R/data/xgb_test1000.csv")

####Error Analysis##############################################################################

# MAE(predicted, original)#######################################
#CART
MAE_CARTtrain1000 <- MAE(rpart_train1000, dftrain1000$logTCP)
MAE_CARTtest1000 <- MAE(rpart_test1000, dftest1000$logTCP)

#RF caret
MAE_RFtrain1000 <- MAE(rfcaret_train1000, dftrain1000$logTCP)
MAE_RFtest1000 <- MAE(rfcaret_test1000, dftest1000$logTCP)

#GBM
MAE_GBMtrain1000 <- MAE(gbm_train1000, dftrain1000$logTCP)
MAE_GBMtest1000 <- MAE(gbm_test1000, dftest1000$logTCP)

#XGB
MAE_XGBtrain1000 <- MAE(xgb_train1000, dftrain1000$logTCP)
MAE_XGBtest1000 <- MAE(xgb_test1000, dftest1000$logTCP)

# RMSE(predicted, original)#######################################
#CART
RMSE_CARTtrain1000 <- RMSE(rpart_train1000, dftrain1000$logTCP)
RMSE_CARTtest1000 <- RMSE(rpart_test1000, dftest1000$logTCP)

#RF caret
RMSE_RFtrain1000 <- RMSE(rfcaret_train1000, dftrain1000$logTCP)
RMSE_RFtest1000 <- RMSE(rfcaret_test1000, dftest1000$logTCP)

#XGB
RMSE_GBMtrain1000 <- RMSE(gbm_train1000, dftrain1000$logTCP)
RMSE_GBMtest1000 <- RMSE(gbm_test1000, dftest1000$logTCP)

#XGB
RMSE_XGBtrain1000 <- RMSE(xgb_train1000, dftrain1000$logTCP)
RMSE_XGBtest1000 <- RMSE(xgb_test1000, dftest1000$logTCP)

# R2(predicted, original)#######################################
#CART
R2_CARTtrain1000 <- R2(rpart_train1000, dftrain1000$logTCP)
R2_CARTtest1000 <- R2(rpart_test1000, dftest1000$logTCP)

#RF caret
R2_RFtrain1000 <- R2(rfcaret_train1000, dftrain1000$logTCP)
R2_RFtest1000 <- R2(rfcaret_test1000, dftest1000$logTCP)

#XGB
R2_GBMtrain1000 <- R2(gbm_train1000, dftrain1000$logTCP)
R2_GBMtest1000 <- R2(gbm_test1000, dftest1000$logTCP)

#XGB
R2_XGBtrain1000 <- R2(xgb_train1000, dftrain1000$logTCP)
R2_XGBtest1000 <- R2(xgb_test1000, dftest1000$logTCP)

# # Error diagnostics for all models
# #error_AllModels <- data.frame(mae_p1_cart, rmse_p1_cart, r2_p1_cart))

a1000 <- c('CART', MAE_CARTtrain1000, RMSE_CARTtrain1000, R2_CARTtrain1000, MAE_CARTtest1000, RMSE_CARTtest1000, R2_CARTtest1000)

b1000 <- c('RFcaret', MAE_RFtrain1000, RMSE_RFtrain1000, R2_RFtrain1000, MAE_RFtest1000, RMSE_RFtest1000, R2_RFtest1000)

c1000 <- c('GBM', MAE_GBMtrain1000, RMSE_GBMtrain1000, R2_GBMtrain1000, MAE_GBMtest1000, RMSE_GBMtest1000, R2_GBMtest1000)

d1000 <- c('XGB', MAE_XGBtrain1000, RMSE_XGBtrain1000, R2_XGBtrain1000, MAE_XGBtest1000, RMSE_XGBtest1000, R2_XGBtest1000)

error_4models1000 <- rbind(a1000,b1000,c1000, d1000)
error_4models1000 <- as.data.table(error_4models1000)
class(error_4models1000)

names(error_4models1000) <- c('Model', 'MAE Train', 'RMSE Train', 'R2 Train', 'MAE Test', 'RMSE Test', 'R2 Test')
error_4models1000

write.csv(error_4models1000, file = "error_4models1000.csv", row.names=FALSE)

###### BIAS CORRECTION############################################################
# bias correct the predictions 

#try this first with rf caret training: 

# The training data (ML and obs) are sorted
sort_train_obs <- sort(dftrain1000$logTCP) #sorting the training data by size
sort_train_pred <- sort(rfcaret_train1000)

# For each value predicted by the xgb model, find the matching observed value (based on sorted vectors)
train_pred_corrected <- approx(sort_train_pred, sort_train_obs, rfcaret_train1000, ties = mean)$y     #y is the predicted matched value

# get the slope of the fitted line between predicted and corrected values
ltf_slope <- coef(lm(rfcaret_train1000 ~ train_pred_corrected))[[2]]

# Apply method to hold-out data.HoldoutPred_corrected is new output of model
# For a given value of uncorrected hold out predictions, we interpolate using the ordered (x,y) pairs: (PredEst, train_pred_corrected)
holdout_pred_corrected <- approx(rfcaret_train1000, train_pred_corrected, rfcaret_test1000, ties = mean)$y
summary(holdout_pred_corrected)

# Second step: use QQ_slope to adjust the extreme values that led to nulls (this has no effect on the hold out predictions because none outside range)
holdout_pred_corrected <- ifelse(rfcaret_test1000 < min(rfcaret_train1000), min(train_pred_corrected) - ltf_slope *(min(rfcaret_train1000) - rfcaret_test1000), holdout_pred_corrected)
holdout_pred_corrected <- ifelse(rfcaret_test1000 > max(rfcaret_train1000), max(train_pred_corrected) + ltf_slope *(rfcaret_test1000 - max(rfcaret_train1000)), holdout_pred_corrected)
summary(holdout_pred_corrected)

summary(holdout_pred_corrected)
summary(rfcaret_train1000)

#RF caret uncorrected 
R2_Reg <- R2(rfcaret_test1000, dftest1000$logTCP)
R2_Reg

#RF caret bias corrected
R2_bias <- R2(holdout_pred_corrected, dftest1000$logTCP)
R2_bias
