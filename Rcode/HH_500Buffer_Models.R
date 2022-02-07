#EV500Mac#### Machine Learning Models for TCP Prediction using the caret package ####

#Load library
library(caret)
library(data.table)
library(ggplot2)
library(randomForest)
library(ranger)
library(dplyr)
library(tidyverse)

# Read in df 
setwd("~/Box/Ch2/R")
df500 <- read.csv(file = 'data/EV500_HH.csv')

# Take a look at df structure and summary
str(df500)
summary(df500)
dim(df500)

# Count number of rows with TCP level == 0
colSums(df500 == 0)

# Find annoying NA values in df
colSums(is.na(df500))
colnames(df500)

#Remove ID column #1
df500[c(1, 2, 3, 4)] <- NULL
colnames(df500)
dim(df500)

# Remove any NA
df500[df500 == "?"] <- NA 
df500 <- df500[!(df500$Age %in% c(NA)),]
df500 <- df500[!(df500$Drainage %in% c(NA)),]
df500 <- df500[!(df500$HSG %in% c(NA)),]
df500 <- df500[!(df500$LU2005 %in% c(NA)),]
df500 <- df500[!(df500$LU1990 %in% c(NA)),]
df500 <- df500[!(df500$LU1975 %in% c(NA)),]
df500 <- df500[!(df500$LU1960 %in% c(NA)),]
df500 <- df500[!(df500$LU1945 %in% c(NA)),]
df500 <- df500[!(df500$Precip2005 %in% c(NA)),]
df500 <- df500[!(df500$Precip1990 %in% c(NA)),]
df500 <- df500[!(df500$Precip1975 %in% c(NA)),]
df500 <- df500[!(df500$Precip1960 %in% c(NA)),]
df500 <- df500[!(df500$Irrig2005 %in% c(NA)),]
df500 <- df500[!(df500$Irrig1990 %in% c(NA)),]
df500 <- df500[!(df500$Irrig1975 %in% c(NA)),]
df500 <- df500[!(df500$Irrig1960 %in% c(NA)),]
df500 <- df500[!(df500$NO3_180 %in% c(NA)),]
df500 <- df500[!(df500$NO3_400 %in% c(NA)),]
df500 <- df500[!(df500$DO5 %in% c(NA)),]
df500 <- df500[!(df500$DO2 %in% c(NA)),]

dim(df500)

any(is.na(df500))

# Make categorical df into factors
as.data.frame(df500)
dim(df500)

df500 <- transform(
  df500,
  LU1945=as.factor(LU1945),
  LU1960=as.factor(LU1960),
  LU1975=as.factor(LU1975),
  LU1990=as.factor(LU1990),
  LU2005=as.factor(LU2005),
  HSG=as.factor(HSG),
  Drainage=as.factor(Drainage)
)
dim(df500)
str(df500)

# df Partition - split df in train and test set (80/20 split)
set.seed(123)

# Create partition index to split df into training and testing set - next step follows...
index <- createDataPartition(df500$logTCP, p = .80, list = FALSE)

# Subset that df with index from above into a train and test object
dftrain500 <- df500[index,]
dftest500 <- df500[-index,]

#check the dimensions of the test and training and look at distribution
dim(dftrain500)
dim(dftest500)

# Create a plot of training and testing (copy first) just to make sure their distributions are similar
hist_dftrain <- copy(dftrain500)
hist_dftest <- copy(dftest500)

hist_dftrain$set <- 'training'
hist_dftest$set <- 'testing'

dfgg <- rbind(hist_dftrain, hist_dftest)

gg <- ggplot(dfgg, aes(x = logTCP, fill = set)) + 
  geom_histogram(bins = 100) + 
  xlim(-3,2) + geom_density(alpha = 0.1)
gg
##################################################################################################
trainctrl <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3)
################################################################################################

#### CART MODEL ##################################################################
set.seed(111)

# fit the model
rpart_tree <- train(logTCP ~ ., 
                    data=dftrain500, 
                    method="rpart", 
                    tuneLength = 50, 
                    #metric = "Accuracy",
                    trControl = trainctrl)
rpart_tree

rpart_train500 <- predict(rpart_tree, newdata=dftrain500)
rpart_train500
rpart_test500 <- predict(rpart_tree, newdata=dftest500)
rpart_test500

set.seed(112)
# fit the model
rpart2_tree500 <- train(logTCP ~ ., 
                    data=dftrain500, 
                    method="rpart2", 
                    tuneLength = 50, 
                    #metric = "Accuracy",
                    trControl = trainctrl)
rpart2_tree500

rpart2_train500 <- predict(rpart2_tree, newdata=dftrain500)
rpart2_train500
rpart2_test500 <- predict(rpart2_tree, newdata=dftest500)
rpart2_test500

#### RANDOM FOREST MODEL with randomForest ##################################################################
#Train by default, the number of decision trees in the forest is 500 and the number of features used as potential candidates for each split is 3.
rF <- randomForest(
  logTCP ~ .,
  data = dftrain500, 
  ntree = 1001, 
  mtry = 10
)

rF_train500 <- predict(rF, newdata = dftrain500)
rF_test500 <- predict(rF, newdata = dftest500)

rF_train500 <- predict(rF, newdata = dftrain500)
rF_test500 <- predict(rF, newdata = dftest500)

R2train <- 1 - (sum((dftrain500$logTCP-rF_train500)^2)/sum((dftrain500$logTCP-mean(dftrain500$logTCP))^2))
R2train 

R2test <- 1 - (sum((dftest500$logTCP-rF_test500)^2)/sum((dftest500$logTCP-mean(dftest500$logTCP))^2))
R2test 

length(dftest500$logTCP)
length(rF_test500)

# rf2 <- tuneRF(
#   x          = dftrain500[,-1],
#   y          = dftrain500[,1],
#   ntreeTry   = 500,
# )

varImp(rF)
library(dplyr)
VI <- varImp(rF)
head(VI)

# this part just creates the data.frame for the plot part
library(dplyr)
VI <- as.data.frame(VI)
VI$varnames <- rownames(VI) # row names to column
rownames(VI) <- NULL  
head(VI)

# this is the plot part, be sure to use reorder with the correct measure name
# imp_gg_VI <- ggplot(VI, aes(x=reorder(varnames, Overall), y=Overall)) + 
#   geom_point() +
#   geom_segment(aes(x=varnames,xend=varnames,y=0,yend=Overall)) +
#   scale_color_discrete(name="Variable Group") +
#   ylab("Mean Decrease Gini") +
#   xlab("Variable Name") +
#   coord_flip()
# imp_gg_VI 

#### RANDOM FOREST MODEL with CARET ##################################################################
set.seed(222)

#create a grid with different hyperparmeters
rftunegrid500 <- expand.grid(mtry=c(8,12))

#Now train with above hyperparameters
rf_tree500 <- train(logTCP ~ ., method="rf", data = dftrain500, tuneGrid = rftunegrid500, trControl = trainctrl)
rf_tree500

varImp(rf_tree500)

#Now train with the best hyperparameters
rf_train500 <- predict(rf_tree500, newdata = dftrain500)
rf_train500

rf_test500 <- predict(rf_tree500, newdata = dftest500)
rf_test500

plot(rf_train500)

#also try method = ranger#################################
set.seed(333)
library(ranger)

rangertunegrid <- expand.grid(mtry = 1:3)

rangerGrid <- rangertunegrid$bestTune

#bestTune

ranger_tree500 <- train(logTCP ~ ., method="ranger", data = dftrain500, trControl = trainctrl, tuneGrid = rangerGrid, verbose=FALSE)
ranger_tree500

ranger_train500 <- predict(ranger_tree500, newdata = dftrain500)
ranger_train500

ranger_test500 <- predict(ranger_tree500, newdata = dftest500)
ranger_test500

#### GRADIENT BOOSTED MODEL ##################################################################
set.seed(444)

# gbm_tree_auto500 <- train(logTCP ~ .,
#                        data=dftrain500,
#                        method="gbm",
#                        distribution="gaussian",
#                        trControl = trainctrl,
#                        verbose = FALSE)
# 
# #Tuning Gradient Boosted Model
# myGrid <- expand.grid(n.trees = c(500),
#                       interaction.depth = c(7, 10),
#                       shrinkage = c(0.1, 0.2),
#                       n.minobsinnode = c(7, 10))
# 
# gbm_tree_tune500 <- train(logTCP ~ .,
#                        data=dftrain500,
#                        method="gbm",
#                        distribution="gaussian",
#                        trControl = trainctrl, 
#                        tuneGrid = myGrid, 
#                        verbose=FALSE)
# gbm_tree_tune500

bestgrid <- expand.grid(n.trees = 500,
                      interaction.depth = 10,
                      shrinkage = 0.1,
                      n.minobsinnode = 10)

gbm_tree500 <- train(logTCP ~ .,
                  data=dftrain500,
                  method="gbm",
                  distribution="gaussian",
                  trControl = trainctrl,
                  tuneGrid = bestgrid,
                  verbose = FALSE)
gbm_tree500

gbm_train500 <- predict(gbm_tree500, newdata = dftrain500)
gbm_test500 <- predict(gbm_tree500, newdata = dftest500)

####Error TRAINING##############################################################################
MAE_CARTtrain500 <- MAE(rpart_train500, dftrain500$logTCP)
MAE_CARTtest500 <- MAE(rpart_test500, dftest500$logTCP)

#rF randomForest
MAE_rFtrain500 <- MAE(rF_train500, dftrain500$logTCP)
MAE_rFtest500 <- MAE(rF_test500, dftest500$logTCP)
MAE_rFtrain500
MAE_rFtest500

#RF Caret
MAE_RFtrain500 <- MAE(rf_train500, dftrain500$logTCP)
MAE_RFtest500 <- MAE(rf_test500, dftest500$logTCP)

#Ranger
MAE_RANGERtrain500 <- MAE(ranger_train500, dftrain500$logTCP)
MAE_RANGERtest500 <- MAE(ranger_test500, dftest500$logTCP)

#GBM
MAE_GBMtrain500 <- MAE(gbm_train500, dftrain500$logTCP)
MAE_GBMtest500 <- MAE(gbm_test500, dftest500$logTCP)

# RMSE(predicted, original)#######################################
#CART
RMSE_CARTtrain500 <- RMSE(rpart_train500, dftrain500$logTCP)
RMSE_CARTtest500 <- RMSE(rpart_test500, dftest500$logTCP)

#rF randomForest
RMSE_rFtrain500 <- RMSE(rF_train500, dftrain500$logTCP)
RMSE_rFtest500 <- RMSE(rF_test500, dftest500$logTCP)
RMSE_rFtrain500
RMSE_rFtest500

#RF Caret
RMSE_RFtrain500 <- RMSE(rf_train500, dftrain500$logTCP)
RMSE_RFtest500 <- RMSE(rf_test500, dftest500$logTCP)

#Ranger
RMSE_RANGERtrain500 <- RMSE(ranger_train500, dftrain500$logTCP)
RMSE_RANGERtest500 <- RMSE(ranger_test500, dftest500$logTCP)

#GBM
RMSE_GBMtrain500 <- RMSE(gbm_train500, dftrain500$logTCP)
RMSE_GBMtest500 <- RMSE(gbm_test500, dftest500$logTCP)

# R2(predicted, original)#######################################
R2_CARTtrain500 <- R2(rpart_train500, dftrain500$logTCP)
R2_CARTtest500 <- R2(rpart_test500, dftest500$logTCP)

#rF randomForest
R2_rFtrain500 <- R2(rF_train500, dftrain500$logTCP, use = ifelse(na.rm, "complete.obs", "everything"))
R2_rFtest500 <- R2(rF_test500, dftest500$logTCP, use = ifelse(na.rm, "complete.obs", "everything"))

postResample(pred = rF_train500, obs = dftrain500$logTCP)

#RF Caret
R2_RFtrain500 <- R2(rf_train500, dftrain500$logTCP)
R2_RFtest500 <- R2(rf_test500, dftest500$logTCP)

#Ranger
R2_RANGERtrain500 <- R2(ranger_train500, dftrain500$logTCP)
R2_RANGERtest500 <- R2(ranger_test500, dftest500$logTCP)

#GBM
R2_GBMtrain500 <- R2(gbm_train500, dftrain500$logTCP)
R2_GBMtest500 <- R2(gbm_test500, dftest500$logTCP)

# # Error diagnostics for all models
# #error_AllModels <- data.frame(mae_p1_cart, rmse_p1_cart, r2_p1_cart))

a500 <- c('CART' , MAE_CARTtrain500, RMSE_CARTtrain500, R2_CARTtrain500, MAE_CARTtest500, RMSE_CARTtest500, R2_CARTtest500)
b500 <- c('rF', MAE_rFtrain500, RMSE_rFtrain500, R2_rFtrain500, MAE_rFtest500, RMSE_rFtest500, R2_rFtest500)
c500 <- c('RF caret', MAE_RFtrain500, RMSE_RFtrain500, R2_RFtrain500, MAE_RFtest500, RMSE_RFtest500, R2_RFtest500)
d500 <- c('Ranger', MAE_RANGERtrain500, RMSE_RANGERtrain500, R2_RANGERtrain500, MAE_RANGERtest500, RMSE_RANGERtest500, R2_RANGERtest500)
e500 <- c('GBM', MAE_GBMtrain500, RMSE_GBMtrain500, R2_GBMtrain500, MAE_GBMtest500, RMSE_GBMtest500, R2_GBMtest500)

error_all500 <- rbind(a500,b500,c500,d500, e500)
error_all500 <- as.data.table(error_all500)
class(error_all500)

names(error_all500) <- c('Model', 'MAE Train', 'RMSE Train', 'R2 Train', 'MAE Test', 'RMSE Test', 'R2 Test')
error_all500
print(error_all500)

write.csv(error_all500, file = "model_error_caret500.csv", row.names=FALSE)
