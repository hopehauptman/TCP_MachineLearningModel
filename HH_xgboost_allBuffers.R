############ Machine Learning Models for TCP Prediction using the caret package ####

# load the required packages
library(caret) # version 6.0-86
library(xgboost) # version 1.2.0.1 required to reproduce model results exactly
library(tidyverse) # version 1.3.0
library(hydroGOF) # version 0.4-0
library(recipes) # version 0.1.15
library(raster) # version 3.4-5
library(data.table)
library(ggplot2)
library(dplyr)
library(tidyverse)

#set working directory to source file loation
setwd("C:/Users/Hope Hauptman/Box/Ch2/R")

# Read in df 
df1000 <- read.csv(file = 'data/EV1000_HH.csv')

# Take a look at df structure and summary
str(df1000)
summary(df1000)
dim(df1000x)

# Count number of rows with TCP level == 0
colSums(df10 == 0)

# Find annoying NA values in df
colSums(is.na(df1000x))
colnames(df1000x)

#Remove ID column #1 and other unneeded columns
df1000x[c(1, 2, 3, 4)] <- NULL
colnames(df)
dim(df1000x)

# Remove any NA
df1000x[df1000x == "?"] <- NA 
df1000x <- df1000x[!(df1000x$Age %in% c(NA)),]
df1000x <- df1000x[!(df1000x$Drainage %in% c(NA)),]
df1000x <- df1000x[!(df1000x$HSG %in% c(NA)),]
df1000x <- df1000x[!(df1000x$LU2005 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$LU1990 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$LU1975 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$LU1960 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$LU1945 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$Precip2005 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$Precip1990 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$Precip1975 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$Precip1960 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$Irrig2005 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$Irrig1990 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$Irrig1975 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$Irrig1960 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$NO3_180 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$NO3_400 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$DO5 %in% c(NA)),]
df1000x <- df1000x[!(df1000x$DO2 %in% c(NA)),]

dim(df1000x)
any(is.na(df1000x))

# Make categorical df into factors
as.data.frame(df1000x)

df1000x <- transform(
  df1000x,
  LU19410=as.factor(LU19410),
  LU1960=as.factor(LU1960),
  LU19710=as.factor(LU19710),
  LU1990=as.factor(LU1990),
  LU20010=as.factor(LU20010),
  HSG=as.factor(HSG),
  Drainage=as.factor(Drainage)
)
dim(df1000x)
str(df1000x)

# df Partition - split df in train and test set (80/20 split)
set.seed(123)

# Create partition index to split df into training and testing set - next step follows...
index <- createDataPartition(df1000x$logTCP, p = .80, list = FALSE)

# Subset that df with index from above into a train and test object
dftrain1000x <- df1000x[index,]
dftest1000x <- df1000x[-index,]

#check the dimensions of the test and training and look at distribution
dim(dftrain1000x)
dim(dftest1000x)

head(dftrain1000x)

##################################################################################################
cv.ctrl <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3)
###BUFFEER=1000meters#######################################################################################

#### xgboost in CARET package ##################################################################
set.seed(888)

# Tune model xgboost in Caret 
xgbCaret_grid_1 <- expand.grid(
  nrounds = 10000,
  max_depth = 20,
  eta = 0.01,
  gamma = 1,
  colsample_bytree = 1,
  min_child_weight = 3,
  subsample = 1)

# train  the model with the best parameters
xgb_train_1 <- train(
  x = as.matrix(dftrain1000[, -1]),
  y = as.numeric(dftrain1000$logTCP),
  trControl = cv.ctrl,
  tuneGrid = xgbCaret_grid_1,
  method = "xgbTree"
)

xgb_train_1$bestTune

#write and/or load the model to a file and to load file next time: 
# save(xgb_train_1, file= "xgb_train_1000.rda")
# load(file = "xgb_train_1000.rda")

#xgbCaret_grid_1$best

xgb_train1000 <- predict(xgb_train_1, newdata = dftrain1000[, -1])
xgb_train1000
xgb_test1000 <- predict(xgb_train_1, newdata = dftest1000[, -1])
xgb_test1000
####Error TRAINING##############################################################################
#GBM
MAE_XGBtrain1000 <- MAE(xgb_train1000, dftrain1000$logTCP)
MAE_XGBtrain1000
MAE_XGBtest1000 <- MAE(xgb_test1000, dftest1000$logTCP)
MAE_XGBtest1000

# RMSE(predicted, original)#######################################
#GBM
RMSE_XGBtrain1000 <- RMSE(xgb_train1000, dftrain1000$logTCP)
RMSE_XGBtrain1000
RMSE_XGBtest1000 <- RMSE(xgb_test1000, dftest1000$logTCP)
RMSE_XGBtest1000

# R2(predicted, original)#######################################
#GBM
R2_XGBtrain1000 <- R2(xgb_train1000, dftrain1000$logTCP)
R2_XGBtrain1000
R2_XGBtest1000 <- R2(xgb_test1000, dftest1000$logTCP)
R2_XGBtest1000

#SEE: https://stats.stackexchange.com/questions/171043/how-to-tune-hyperparameters-of-xgboost-trees

##### xgboost in xgboost package ##################################################################

#put training data explanatory variables in matrix
train_x = data.matrix(dftrain1000[, -1])
train_y = dftrain1000[,1]

test_x = data.matrix(dftest1000[, -1])
test_y = dftest1000[, 1]

xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

xgbc = xgboost(data = xgb_train, max.depth = 21, nrounds = 10000, trControl = cv.ctrl, eta = 0.01,
               gamma = 1, colsample_bytree = 1, min_child_weight = 3, subsample = 1, verbose = F)
print(xgbc)

xgbc$bestTune

pred_y_train = predict(xgbc, xgb_train)
pred_y_test = predict(xgbc, xgb_test)

mse0 = mean((train_y - pred_y_train)^2)
mae0 = caret::MAE(train_y, pred_y_train)
rmse0 = caret::RMSE(train_y, pred_y_train)
R20 = caret::R2(train_y, pred_y_train)

cat("MSE0: "= mse0, "MAE0: "= mae0, " RMSE0: "= rmse0, " R20 "= R20)

mse = mean((test_y - pred_y_test)^2)
mae = caret::MAE(test_y, pred_y_test)
rmse = caret::RMSE(test_y, pred_y_test)
R2 = caret::R2(test_y, pred_y_test)

cat("MSE: "= mse, "MAE: "= mae, " RMSE: "= rmse, " R2 "= R2)

# x = 1:length(test_y)
# plot(x, test_y, col = "red", type = "l")
# lines(x, pred_y, col = "blue", type = "l")
# legend(x = 1, y = 38,  legend = c("original test_y", "predicted test_y"), 
#        col = c("red", "blue"), box.lty = 1, cex = 0.8, lty = c(1, 1))

###BUFFEER=500 meters#######################################################################################

# Read in df 
df500x <- read.csv(file = 'data/EV500_HH.csv')

# Take a look at df structure and summary
str(df500x)
summary(df500x)
dim(df500x)

# Count number of rows with TCP level == 0
colSums(df500x == 0)

# Find annoying NA values in df
colSums(is.na(df500x))
colnames(df500x)

#Remove ID column #1 and other unneeded columns
df500x[c(1, 2, 3, 4)] <- NULL
colnames(df)
dim(df500x)

# Remove any NA
df1000x[df1000x == "?"] <- NA 
df500x <- df500x[!(df500x$Age %in% c(NA)),]
df500x <- df500x[!(df500x$Drainage %in% c(NA)),]
df500x <- df500x[!(df500x$HSG %in% c(NA)),]
df500x <- df500x[!(df500x$LU2005 %in% c(NA)),]
df500x <- df500x[!(df500x$LU1990 %in% c(NA)),]
df500x <- df500x[!(df500x$LU1975 %in% c(NA)),]
df500x <- df500x[!(df500x$LU1960 %in% c(NA)),]
df500x <- df500x[!(df500x$LU1945 %in% c(NA)),]
df500x <- df500x[!(df500x$Precip2005 %in% c(NA)),]
df500x <- df500x[!(df500x$Precip1990 %in% c(NA)),]
df500x <- df500x[!(df500x$Precip1975 %in% c(NA)),]
df500x <- df500x[!(df500x$Precip1960 %in% c(NA)),]
df500x <- df500x[!(df500x$Irrig2005 %in% c(NA)),]
df500x <- df500x[!(df500x$Irrig1990 %in% c(NA)),]
df500x <- df500x[!(df500x$Irrig1975 %in% c(NA)),]
df500x <- df500x[!(df500x$Irrig1960 %in% c(NA)),]
df500x <- df500x[!(df500x$NO3_180 %in% c(NA)),]
df500x <- df500x[!(df500x$NO3_400 %in% c(NA)),]
df500x <- df500x[!(df500x$DO5 %in% c(NA)),]
df500x <- df500x[!(df500x$DO2 %in% c(NA)),]

dim(df500x)
any(is.na(df500x))

# Make categorical df into factors
as.data.frame(df500x)

df500x <- transform(
  df500x,
  LU19410=as.factor(LU1945),
  LU1960=as.factor(LU1960),
  LU19710=as.factor(LU1975),
  LU1990=as.factor(LU1990),
  LU20010=as.factor(LU2005),
  HSG=as.factor(HSG),
  Drainage=as.factor(Drainage)
)
dim(df500x)
str(df500x)

# df Partition - split df in train and test set (80/20 split)
set.seed(123)

# Create partition index to split df into training and testing set - next step follows...
index <- createDataPartition(df500x$logTCP, p = .80, list = FALSE)

# Subset that df with index from above into a train and test object
dftrain500x <- df500x[index,]
dftest500x <- df500x[-index,]

#check the dimensions of the test and training and look at distribution
dim(dftrain500x)
dim(dftest500x)

head(dftrain500x)

###BUFFEER=500meters#######################################################################################

#### xgboost in CARET package ##################################################################
set.seed(888)

# Tune model xgboost in Caret 
xgbCaret_grid_5 <- expand.grid(
  nrounds = 10000,
  max_depth = 20,
  eta = 0.01,
  gamma = 1,
  colsample_bytree = 1,
  min_child_weight = 3,
  subsample = 1)

# train  the model with the best parameters
xgb_train_5 <- train(
  x = as.matrix(dftrain500[, -1]),
  y = as.numeric(dftrain500$logTCP),
  trControl = cv.ctrl,
  tuneGrid = xgbCaret_grid_5,
  method = "xgbTree"
)

xgb_train_5$bestTune

#write and/or load the model to a file and to load file next time: 
# save(xgb_train_1, file= "xgb_train_1000.rda")
# load(file = "xgb_train_1000.rda")

#xgbCaret_grid_1$best

xgb_train500 <- predict(xgb_train_5, newdata = dftrain500[, -1])
xgb_train500
xgb_test500 <- predict(xgb_train_5, newdata = dftest500[, -1])
xgb_test500
####Error TRAINING##############################################################################
#xgbTree
MAE_XGBtrain500 <- MAE(xgb_train500, dftrain500x$logTCP)
MAE_XGBtrain500
MAE_XGBtest500 <- MAE(xgb_test500, dftest500x$logTCP)
MAE_XGBtest500

# RMSE(predicted, original)#######################################
#xgbTree
RMSE_XGBtrain500 <- RMSE(xgb_train500, dftrain500$logTCP)
RMSE_XGBtrain500
RMSE_XGBtest500 <- RMSE(xgb_test500, dftest500$logTCP)
RMSE_XGBtest500

# R2(predicted, original)#######################################
#xgbTree
R2_XGBtrain500 <- R2(xgb_train500, dftrain500$logTCP)
R2_XGBtrain500
R2_XGBtest500 <- R2(xgb_test500, dftest500$logTCP)
R2_XGBtest500

#####500 meters xgboost in xgboost package ##################################################################

#put training data explanatory variables in matrix
train_x500 = data.matrix(dftrain500[, -1])
train_y500 = dftrain500[,1]

test_x500 = data.matrix(dftest500[, -1])
test_y500 = dftest500[, 1]

xgb_train500 = xgb.DMatrix(data = train_x, label = train_y)
xgb_test500 = xgb.DMatrix(data = test_x, label = test_y)

xgbc500 = xgboost(data = xgb_train, max.depth = 21, nrounds = 10000, trControl = cv.ctrl, eta = 0.01,
               gamma = 1, colsample_bytree = 1, min_child_weight = 3, subsample = 1, verbose = F)
xgbc500

xgbc500$bestTune

pred_y_train500 = predict(xgbc500, xgb_train500)
pred_y_test500 = predict(xgbc500, xgb_test500)

mse500a = mean((train_y500 - pred_y_train500)^2)
mae500a = caret::MAE(train_y500, pred_y_train500)
rmse500a = caret::RMSE(train_y500, pred_y_train500)
R2500a = caret::R2(train_y500, pred_y_train500)

cat("MSE500a: ", mse500a, "MAE500a: ", mae500a, " RMSE500a: ", rmse500a, " R2500a ", R2500a)

mse500b = mean((test_y - pred_y)^2)
mae500b = caret::MAE(test_y, pred_y_test)
rmse500b = caret::RMSE(test_y, pred_y_test)
R2500b = caret::R2(test_y, pred_y_test)

cat("MSE500b: ", mse500b, "MAE500b: ", mae500b, " RMSE500b: ", rmse500b, " R2500b ", R2500b)

###BUFFEER=1500meters#######################################################################################
# Read in df 
df1500x <- read.csv(file = 'data/EV1500_HH.csv')

# Take a look at df structure and summary
str(df1500x)
summary(df1500x)
dim(df1500x)

# Count number of rows with TCP level == 0
colSums(df1500x == 0)

# Find annoying NA values in df
colSums(is.na(df1500x))
colnames(df1500x)

#Remove ID column #1 and other unneeded columns
df1500x[c(1, 2, 3, 4)] <- NULL
colnames(df1500x)
dim(df1500x)

# Remove any NA
df1500x[df1500x == "?"] <- NA 
df1500x <- df1500x[!(df1500x$Age %in% c(NA)),]
df1500x <- df1500x[!(df1500x$Drainage %in% c(NA)),]
df1500x <- df1500x[!(df1500x$HSG %in% c(NA)),]
df1500x <- df1500x[!(df1500x$LU2005 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$LU1990 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$LU1975 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$LU1960 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$LU1945 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$Precip2005 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$Precip1990 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$Precip1975 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$Precip1960 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$Irrig2005 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$Irrig1990 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$Irrig1975 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$Irrig1960 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$NO3_180 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$NO3_400 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$DO5 %in% c(NA)),]
df1500x <- df1500x[!(df1500x$DO2 %in% c(NA)),]

dim(df1500x)
any(is.na(df1500x))

# Make categorical df into factors
as.data.frame(df1500x)

df1500x <- transform(
  df1500x,
  LU19410=as.factor(LU1945),
  LU1960=as.factor(LU1960),
  LU19710=as.factor(LU1975),
  LU1990=as.factor(LU1990),
  LU20010=as.factor(LU2005),
  HSG=as.factor(HSG),
  Drainage=as.factor(Drainage)
)
dim(df1500x)
str(df1500x)

# df Partition - split df in train and test set (80/20 split)
set.seed(123)

# Create partition index to split df into training and testing set - next step follows...
index <- createDataPartition(df1500x$logTCP, p = .80, list = FALSE)

# Subset that df with index from above into a train and test object
dftrain1500x <- df1500x[index,]
dftest1500x <- df1500x[-index,]

#check the dimensions of the test and training and look at distribution
dim(dftrain1500x)
dim(dftest1500x)

head(dftrain1500x)

#### xgboost in CARET package ##################################################################
set.seed(888)

# Tune model xgboost in Caret 
xgbCaret_grid_15 <- expand.grid(
  nrounds = 10000,
  max_depth = 20,
  eta = 0.01,
  gamma = 1,
  colsample_bytree = 1,
  min_child_weight = 3,
  subsample = 1)

# train  the model with the best parameters
xgb_train_15 <- train(
  x = as.matrix(dftrain1500[, -1]),
  y = as.numeric(dftrain1500$logTCP),
  trControl = cv.ctrl,
  tuneGrid = xgbCaret_grid_15,
  method = "xgbTree"
)

xgb_train_15$bestTune

#write and/or load the model to a file and to load file next time: 
# save(xgb_train_1, file= "xgb_train_1000.rda")
# load(file = "xgb_train_1000.rda")

#xgbCaret_grid_1$best

xgb_train1500 <- predict(xgb_train_15, newdata = dftrain1500[, -1])
xgb_train1500
xgb_test1500 <- predict(xgb_train_15, newdata = dftest1500[, -1])
xgb_test1500
####Error TRAINING##############################################################################
#GBM
MAE_XGBtrain1500 <- MAE(xgb_train1500, dftrain1500$logTCP)
MAE_XGBtrain1500
MAE_XGBtest1500 <- MAE(xgb_test1500, dftest1500$logTCP)
MAE_XGBtest1500

# RMSE(predicted, original)#######################################
#GBM
RMSE_XGBtrain1500 <- RMSE(xgb_train1500, dftrain1500$logTCP)
RMSE_XGBtrain1500
RMSE_XGBtest1500 <- RMSE(xgb_test1500, dftest1500$logTCP)
RMSE_XGBtest1500

# R2(predicted, original)#######################################
#GBM
R2_XGBtrain1500 <- R2(xgb_train1500, dftrain1500$logTCP)
R2_XGBtrain1500
R2_XGBtest1500 <- R2(xgb_test1500, dftest1500$logTCP)
R2_XGBtest1500

#SEE: https://stats.stackexchange.com/questions/171043/how-to-tune-hyperparameters-of-xgboost-trees

##### xgboost in xgboost package ##################################################################

#put training data explanatory variables in matrix
train_x1500 = data.matrix(dftrain1500[, -1])
train_y1500 = dftrain1500[,1]

test_x1500 = data.matrix(dftest1500[, -1])
test_y1500 = dftest1500[, 1]

xgb_train1500 = xgb.DMatrix(data = train_x, label = train_y)
xgb_test1500 = xgb.DMatrix(data = test_x, label = test_y)

xgbc1500 = xgboost(data = xgb_train, max.depth = 21, nrounds = 10000, trControl = cv.ctrl, eta = 0.01,
               gamma = 1, colsample_bytree = 1, min_child_weight = 3, subsample = 1, verbose = F)
print(xgbc1500)

xgbc1500$bestTune

pred_y_train1500 = predict(xgbc1500, xgb_train1500)
pred_y_test1500 = predict(xgbc1500, xgb_test1500)

mse1500a = mean((train_y1500 - pred_y_train1500)^2)
mae1500a = caret::MAE(train_y1500, pred_y_train1500)
rmse1500a = caret::RMSE(train_y1500, pred_y_train1500)
R21500a = caret::R2(train_y1500, pred_y_train1500)

cat("MSE1500a: ", mse1500a, "MAE1500a: ", mae1500a, " RMSE1500a: ", rmse1500a, " R21500a ", R21500a)

mse1500b = mean((test_y1500 - pred_y1500)^2)
mae1500b = caret::MAE(test_y1500, pred_y_test1500)
rmse1500b = caret::RMSE(test_y1500, pred_y_test1500)
R21500b = caret::R2(test_y1500, pred_y_test1500)

cat("MSE1500b: ", mse1500b, "MAE1500b: ", mae1500b, " RMSE1500b: ", rmse1500b, " R21500b ", R21500b)
