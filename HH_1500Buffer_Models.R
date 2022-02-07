#1500BUFFER#### Machine Learning Models for TCP Prediction ####

##Load library
library(caret)
library(data.table)
library(ggplot2)
library(randomForest)
library(ranger)
library(dplyr)
library(tidyverse)

# Read in df 
setwd("C:/Users/Hope Hauptman/Box/Ch2/R")
df1500 <- read.csv(file = 'data/EV1500_HH.csv')

# Take a look at df structure and summary
str(df1500)
summary(df1500)
dim(df1500)

# Count number of rows with TCP level == 0
colSums(df1500 == 0)

# Find annoying NA values in df
colSums(is.na(df1500))
colnames(df1500)

#Remove ID column #1 and other uneeded columns
df1500[c(1, 2, 3, 4)] <- NULL
colnames(df1500)
dim(df1500)

# Remove any NA
df1500[df1500 == "?"] <- NA
df1500 <- df1500[!(df1500$Age %in% c(NA)),]
df1500 <- df1500[!(df1500$Drainage %in% c(NA)),]
df1500 <- df1500[!(df1500$HSG %in% c(NA)),]
df1500 <- df1500[!(df1500$LU2005 %in% c(NA)),]
df1500 <- df1500[!(df1500$LU1990 %in% c(NA)),]
df1500 <- df1500[!(df1500$LU1975 %in% c(NA)),]
df1500 <- df1500[!(df1500$LU1960 %in% c(NA)),]
df1500 <- df1500[!(df1500$LU1945 %in% c(NA)),]
df1500 <- df1500[!(df1500$Precip2005 %in% c(NA)),]
df1500 <- df1500[!(df1500$Precip1990 %in% c(NA)),]
df1500 <- df1500[!(df1500$Precip1975 %in% c(NA)),]
df1500 <- df1500[!(df1500$Precip1960 %in% c(NA)),]
df1500 <- df1500[!(df1500$Irrig2005 %in% c(NA)),]
df1500 <- df1500[!(df1500$Irrig1990 %in% c(NA)),]
df1500 <- df1500[!(df1500$Irrig1975 %in% c(NA)),]
df1500 <- df1500[!(df1500$Irrig1960 %in% c(NA)),]
df1500 <- df1500[!(df1500$NO3_180 %in% c(NA)),]
df1500 <- df1500[!(df1500$NO3_400 %in% c(NA)),]
df1500 <- df1500[!(df1500$DO5 %in% c(NA)),]
df1500 <- df1500[!(df1500$DO2 %in% c(NA)),]

dim(df1500)

any(is.na(df1500))

# Make categorical df into factors
as.data.frame(df1500)

df1500 <- transform(
  df1500,
  LU1945=as.factor(LU1945),
  LU1960=as.factor(LU1960),
  LU1975=as.factor(LU1975),
  LU1990=as.factor(LU1990),
  LU2005=as.factor(LU2005),
  HSG=as.factor(HSG),
  Drainage=as.factor(Drainage)
)
dim(df1500)
str(df1500)

# df Partition - split df in train and test set (80/20 split)
set.seed(123)

# Create partition index to split df into training and testing set - next step follows...
index <- createDataPartition(df1500$logTCP, p = .80, list = FALSE)

# Subset that df with index from above into a train and test object
dftrain1500 <- df1500[index,]
dftest1500 <- df1500[-index,]

#check the dimensions of the test and training and look at distribution
dim(dftrain1500)
dim(dftest1500)

# Create a plot of training and testing (copy first) just to make sure their distributions are similar
hist_dftrain <- copy(dftrain1500)
hist_dftest <- copy(dftest1500)

hist_dftrain$set <- 'training'
hist_dftest$set <- 'testing'

dfgg <- rbind(hist_dftrain, hist_dftest)

ggplot(dfgg, aes(x = logTCP, fill = set)) + 
  geom_histogram(bins = 100) + 
  xlim(-3,2) + geom_density(alpha = 0.1)

##################################################################################################
trainctrl <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3)
################################################################################################

#### CART MODEL ##################################################################
set.seed(111)

# fit the model
rpart_tree1500 <- train(logTCP ~ ., 
                    data=dftrain1500, 
                    method="rpart", 
                    tuneLength = 50, 
                    #metric = "Accuracy",
                    trControl = trainctrl)
rpart_tree1500

rpart_train1500 <- predict(rpart_tree, newdata=dftrain1500)
rpart_train1500
rpart_test1500 <- predict(rpart_tree, newdata=dftest1500)
rpart_test1500

set.seed(112)
# fit the model
rpart2_tree1500 <- train(logTCP ~ ., 
                     data=dftrain1500, 
                     method="rpart2", 
                     tuneLength = 50, 
                     #metric = "Accuracy",
                     trControl = trainctrl)
rpart2_tree1500

rpart2_train1500 <- predict(rpart2_tree1500, newdata=dftrain1500)
rpart2_train1500
rpart2_test1500 <- predict(rpart2_tree1500, newdata=dftest1500)
rpart2_test1500

#### RANDOM FOREST MODEL with randomForest ##################################################################
#Train by default, the number of decision trees in the forest is 500 and the number of features used as potential candidates for each split is 3.
rF1500 <- randomForest(
  logTCP ~ .,
  data = dftrain1500, 
  ntree = 1001, 
  mtry = 10
)

rF_train1500 <- predict(rF1500, newdata = dftrain1500)
rF_test1500 <- predict(rF1500, newdata = dftest1500)

R2train <- 1 - (sum((dftrain1500$logTCP-rF_train1500)^2)/sum((dftrain1500$logTCP-mean(dftrain1500$logTCP))^2))
R2train 

R2test <- 1 - (sum((dftest1500$logTCP-rF_test1500)^2)/sum((dftest1500$logTCP-mean(dftest1500$logTCP))^2))
R2test 

length(dftest1500$logTCP)
length(rF_test1500)

# rf2 <- tuneRF(
#   x          = dftrain500[,-1],
#   y          = dftrain500[,1],
#   ntreeTry   = 500,
# )

varImp(rF1500)
library(dplyr)
VI1500 <- varImp(rF1500)
head(VI1500)

# this part just creates the data.frame for the plot part
library(dplyr)
VI1500 <- as.data.frame(VI1500)
VI1500$varnames <- rownames(VI1500) # row names to column
rownames(VI1500) <- NULL  
head(VI1500)

# this is the plot part, be sure to use reorder with the correct measure name
imp_gg_VI1500 <- ggplot(VI1500, aes(x=reorder(varnames, Overall), y=Overall)) + 
  geom_point() +
  geom_segment(aes(x=varnames,xend=varnames,y=0,yend=Overall)) +
  scale_color_discrete(name="Variable Group") +
  ylab("Mean Decrease Gini") +
  xlab("Variable Name") +
  coord_flip()
imp_gg_VI1500 

# varImpPlot(rF1500, sort = TRUE, main = "Variable Importance Plot" , 
#            labels = rep(c("1975 Land Use",
#                           "2005 Land Use",
#                           "1960 Land Use",
#                           "1945 Land Use",
#                           "Soil drainage class",
#                           "1990 Land Use", 
#                           "Hydrologic soil group",
#                           "Nitrates 400",
#                           "Nitrates 180",
#                           "Groundwater Age",
#                           "Irrigation 1960",
#                           "Dissolved Oxygen 2",
#                           "Dissolved Oxygen 0.5",
#                           "Precipitation 2005")))

#### RANDOM FOREST MODEL with CARET ##################################################################
set.seed(222)

#create a grid with different hyperparmeters
rftunegrid1500 <- expand.grid(mtry=c(8,12))

#Now train with above hyperparameters
rf_tree1500 <- train(logTCP ~ ., method="rf", data = dftrain1500, tuneGrid = rftunegrid1500, trControl = trainctrl)
rf_tree1500

varImp(rf_tree1500)

#Now train with the best hyperparameters
rf_train1500 <- predict(rf_tree1500, newdata = dftrain1500)
rf_train1500

rf_test1500 <- predict(rf_tree1500, newdata = dftest1500)
rf_test1500

#caret function for error stats: 
stats_train1500 <- caret::postResample(rf_tree1500, newdata = dftest1500)
stats_test1500 <- caret::postResample(rf_tree1500, newdata = dftest1500)
#also try method = ranger#################################
set.seed(333)
library(ranger)

rangertunegrid <- expand.grid(mtry = 1:3)

rangerGrid <- rangertunegrid$bestTune

#bestTune

ranger_tree1500 <- train(logTCP ~ ., method="ranger", data = dftrain1500, trControl = trainctrl, tuneGrid = rangerGrid, verbose=FALSE)
ranger_tree1500

ranger_train1500 <- predict(ranger_tree1500, newdata = dftrain1500)
ranger_train1500

ranger_test1500 <- predict(ranger_tree1500, newdata = dftest1500)
ranger_test1500

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

bestgrid1500 <- expand.grid(n.trees = 500,
                        interaction.depth = 10,
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

gbm_tree1500 <- train(logTCP ~ .,
                     data=dftrain1500,
                     method="gbm",
                     distribution="gaussian",
                     trControl = trainctrl,
                     tuneGrid = bestgrid1500,
                     verbose = FALSE)
gbm_tree1500

gbm_train1500 <- predict(gbm_tree1500, newdata = dftrain1500)
gbm_test1500 <- predict(gbm_tree1500, newdata = dftest1500)

####Error TRAINING##############################################################################
MAE_CARTtrain1500 <- MAE(rpart_train1500, dftrain1500$logTCP)
MAE_CARTtest1500 <- MAE(rpart_test1500, dftest1500$logTCP)
MAE_CARTtrain1500

#rF randomForest
MAE_rFtrain1500 <- MAE(rF_train1500, dftrain1500$logTCP)
MAE_rFtest1500 <- MAE(rF_test1500, dftest1500$logTCP)
MAE_rFtest1500

#RF caret
MAE_RFtrain1500 <- MAE(rf_train1500, dftrain1500$logTCP)
MAE_RFtest1500 <- MAE(rf_test1500, dftest1500$logTCP)
MAE_RFtrain1500

#Ranger
MAE_RANGERtrain1500 <- MAE(ranger_train1500, dftrain1500$logTCP)
MAE_RANGERtest1500 <- MAE(ranger_test1500, dftest1500$logTCP)

#GBM
MAE_GBMtrain1500 <- MAE(gbm_train1500, dftrain1500$logTCP)
MAE_GBMtest1500 <- MAE(gbm_test1500, dftest1500$logTCP)

# RMSE(predicted, original)#######################################
#CART
RMSE_CARTtrain1500 <- RMSE(rpart_train1500, dftrain1500$logTCP)
RMSE_CARTtest1500 <- RMSE(rpart_test1500, dftest1500$logTCP)
RMSE_CARTtrain1500

#rF randomForest
RMSE_rFtrain1500 <- RMSE(rF_train1500, dftrain1500$logTCP)
RMSE_rFtest1500 <- RMSE(rF_test1500, dftest1500$logTCP)
RMSE_rFtest1500

#RF caret
RMSE_RFtrain1500 <- RMSE(rf_train1500, dftrain1500$logTCP)
RMSE_RFtest1500 <- RMSE(rf_test1500, dftest1500$logTCP)

#Ranger
RMSE_RANGERtrain1500 <- RMSE(ranger_train1500, dftrain1500$logTCP)
RMSE_RANGERtest1500 <- RMSE(ranger_test1500, dftest1500$logTCP)

#GBM
RMSE_GBMtrain1500 <- RMSE(gbm_train1500, dftrain1500$logTCP)
RMSE_GBMtest1500 <- RMSE(gbm_test1500, dftest1500$logTCP)

# R2(predicted, original)#######################################
R2_CARTtrain1500 <- R2(rpart_train1500, dftrain1500$logTCP)
R2_CARTtest1500 <- R2(rpart_test1500, dftest1500$logTCP)

#rF randomForest
R2_rFtrain1500 <- R2(rF_train1500, dftrain1500$logTCP)
R2_rFtest1500 <- R2(rF_test1500, dftest1500$logTCP)

postResample(rpart_test1500, dftest1500$logTCP)

R2_rFTrain <- 1 - sum((rF_train1500 - dftrain1500$logTCP)^2)/sum((dftrain1500$logTCP-mean(dftrain1500$logTCP))^2)
R2_rFTrain

R2_rFTest <- 1 - (sum((dftest1500$logTCP-rF_test1500)^2)/sum((dftest1500$logTCP-mean(dftest1500$logTCP))^2))
summary(R2_rFTest)

#RF caret
R2_RFtrain1500 <- R2(rf_train1500, dftrain1500$logTCP)
R2_RFtest1500 <- R2(rf_test1500, dftest1500$logTCP)

#Ranger
R2_RANGERtrain1500 <- R2(ranger_train1500, dftrain1500$logTCP)
R2_RANGERtest1500 <- R2(ranger_test1500, dftest1500$logTCP)

#GBM
R2_GBMtrain1500 <- R2(gbm_train1500, dftrain1500$logTCP)
R2_GBMtest1500 <- R2(gbm_test1500, dftest1500$logTCP)



# # Error diagnostics for all models
# #error_AllModels <- data.frame(mae_p1_cart, rmse_p1_cart, r2_p1_cart))

a1500 <- c('CART', MAE_CARTtrain1500, RMSE_CARTtrain1500, R2_CARTtrain1500, MAE_CARTtest1500, RMSE_CARTtest1500, R2_CARTtest1500)
b1500 <- c('rF', MAE_rFtrain1500, RMSE_rFtrain1500, R2_rFtrain1500, MAE_rFtest1500, RMSE_rFtest1500, R2_rFtest1500)
c1500 <- c('RF caret', MAE_RFtrain1500, RMSE_RFtrain1500, R2_RFtrain1500, MAE_RFtest1500, RMSE_RFtest1500, R2_RFtest1500)
d1500 <- c('Ranger', MAE_RANGERtrain1500, RMSE_RANGERtrain1500, R2_RANGERtrain1500, MAE_RANGERtest1500, RMSE_RANGERtest1500, R2_RANGERtest1500)
e1500 <- c('GBM', MAE_GBMtrain1500, RMSE_GBMtrain1500, R2_GBMtrain1500, MAE_GBMtest1500, RMSE_GBMtest1500, R2_GBMtest1500)

error_all1500 <- rbind(a1500,b1500,c1500,d1500)
error_all1500 <- as.data.table(error_all1500)
class(error_all1500)

names(error_all1500) <- c('Model', 'MAE Train', 'RMSE Train', 'R2 Train', 'MAE Test', 'RMSE Test', 'R2 Test')
error_all1500
print(error_all1500)

write.csv(error_all1500, file = "model_error_caret1500.csv", row.names=FALSE)


