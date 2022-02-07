#EV1000Mac#### Machine Learning Models for TCP Prediction using the caret package ####

#Load library
library(caret)
library(data.table)
library(ggplot2)
library(randomForest)
library(ranger)
library(dplyr)
library(tidyverse)
library(gbm)

# Read in df 
setwd("C:/Users/Hope Hauptman/Box/Ch2/R/data")
df1000 <- read.csv(file = 'EV1000_HH.csv')
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

write.csv(dftrain1000, 'C:/Users/Hope Hauptman/Box/Ch2/R/data/dftrain1000.csv')
write.csv(dftest1000, 'C:/Users/Hope Hauptman/Box/Ch2/R/data/dftest1000.csv')

#check the dimensions of the test and training and look at distribution
dim(dftrain1000)
dim(dftest1000)

# Create a plot of training and testing (copy first) just to make sure their distributions are similar
hist_dftrain1000 <- copy(dftrain1000)
hist_dftest1000 <- copy(dftest1000)

hist_dftrain1000$set <- 'training'
hist_dftest1000$set <- 'testing'

dfgg1000 <- rbind(hist_dftrain1000, hist_dftest1000)
df <- dfgg1000 %>% mutate(TCP = exp(logTCP)) %>% select(TCP, set)

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

##################################################################################################
trainctrl <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3)
################################################################################################

####1 CART MODEL ##################################################################
set.seed(111)

# fit the model
rpart_tree1000 <- train(logTCP ~ ., 
                    data=dftrain1000, 
                    method="rpart", 
                    tuneLength = 100, 
                    trControl = trainctrl)
rpart_tree1000

rpart_train1000 <- predict(rpart_tree1000, newdata=dftrain1000)
rpart_train1000
rpart_test1000 <- predict(rpart_tree1000, newdata=dftest1000)
rpart_test1000

set.seed(112)
# fit the model
rpart2_tree1000 <- train(logTCP ~ ., 
                     data=dftrain1000, 
                     method="rpart2", 
                     tuneLength = 100,
                     trControl = trainctrl)
rpart2_tree1000

rpart2_train1000 <- predict(rpart2_tree1000, newdata=dftrain1000)
rpart2_train1000
rpart2_test1000 <- predict(rpart2_tree1000, newdata=dftest1000)
rpart2_test1000

####2 RANDOM FOREST MODEL with randomForest ##################################################################

#Train by default, the number of decision trees in the forest is 1000 and the number of features used as potential candidates for each split is 3.
rF1000 <- randomForest(
  logTCP ~ .,
  data = dftrain1000, 
  ntree = 1001, 
  mtry = 10
)

rF_train1000 <- predict(rF1000, newdata = dftrain1000)
rF_train1000

rF_test1000 <- predict(rF1000, newdata = dftest1000)
rF_test1000

rF_train1000 <- predict(rF1000, newdata = dftrain1000)
rF_test1000 <- predict(rF1000, newdata = dftest1000)

R2train <- 1 - (sum((dftrain1000$logTCP-rF_train1000)^2)/sum((dftrain1000$logTCP-mean(dftrain1000$logTCP))^2))
R2train 

R2test <- 1 - (sum((dftest1000$logTCP-rF_test1000)^2)/sum((dftest1000$logTCP-mean(dftest1000$logTCP))^2))
R2test 

length(dftest1500$logTCP)
length(rF_test1500)

varImp(rF1000)
library(dplyr)
VI1000 <- varImp(rF1000)
head(VI1000)

# this part just creates the data.frame for the plot part
library(dplyr)
VI1000 <- as.data.frame(VI1000)
VI1000$varnames <- rownames(VI1000) # row names to column
rownames(VI1000) <- NULL  
head(VI1000)

# this is the plot part, be sure to use reorder with the correct measure name
# imp_gg_VI1000 <- ggplot(VI1000, aes(x=reorder(varnames, Overall), y=Overall)) + 
#   geom_point() +
#   geom_segment(aes(x=varnames,xend=varnames,y=0,yend=Overall)) +
#   scale_color_discrete(name="Variable Group") +
#   ylab("Mean Decrease Gini") +
#   xlab("Variable Name") +
#   coord_flip()
# imp_gg_VI1000

save(rF1000, file="rF1000.RData", overwrite=T)

####3 RANDOM FOREST MODEL with CARET ##################################################################
set.seed(222)

#create a grid with different hyperparmeters
rftunegrid1000 <- expand.grid(mtry=c(8,12))

#Now train with above hyperparameters
rfcaret_1000 <- train(logTCP ~ ., method="rf", data = dftrain1000, tuneGrid = rftunegrid1000, trControl = trainctrl)
rfcaret_1000

varImp(rfcaret_tree1000)

#Now train with the best hyperparameters
rfcaret_train1000 <- predict(rfcaret_1000, newdata = dftrain1000)
rfcaret_train1000

rfcaret_test1000 <- predict(rfcaret_1000, newdata = dftest1000)
rfcaret_test1000

write.csv(rfcaret_test1000, "C:/Users/Hope Hauptman/Box/Ch2/R/data/rfcaret_test1000.csv")
write.csv(rfcaret_train1000, "C:/Users/Hope Hauptman/Box/Ch2/R/data/rfcaret_train1000.csv")

rfcaret_test1000
plot(exp(rfcaret_test1000), exp(dftest1000$logTCP))
obs_pred <- data.frame(pred = exp(rfcaret_test1000), obs = exp(dftest1000$logTCP))

obs_predPlot <- ggplot(obs_pred, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")

obs_predPlot

#save model to file
save(rfcaret_1000, file="rf_caret1000.Rdata")

load(file = "C:/Users/Hope Hauptman/Box/Ch2/R/data/rf_caret1000.Rdata")

library(dplyr)
caretVarImp <- varImp(rfcaret_1000)
plot(caretVarImp)

imp <- do.call(rbind.data.frame, caretVarImp)
imp

library(writexl)

write_xlsx(caretVarImp,"C:/Users/Hope Hauptman/Box/Ch2/R/data/caretVarImp.xlsx")


#4also try method = ranger#################################
set.seed(333)
library(ranger)

rangertunegrid1000 <- expand.grid(mtry = 1:3)

rangerGrid1000 <- rangertunegrid1000$bestTune

#bestTune

ranger_tree1000 <- train(logTCP ~ ., method="ranger", data = dftrain1000, trControl = trainctrl, tuneGrid = rangerGrid1000, verbose=FALSE)
ranger_tree1000

ranger_train1000 <- predict(ranger_tree1000, newdata = dftrain1000)
ranger_train1000

ranger_test1000 <- predict(ranger_tree1000, newdata = dftest1000)
ranger_test1000

####5 GRADIENT BOOSTED MODEL ##################################################################
set.seed(444)

# gbm_tree_auto1000 <- train(logTCP ~ .,
#                        data=dftrain1000,
#                        method="gbm",
#                        distribution="gaussian",
#                        trControl = trainctrl,
#                        verbose = FALSE)
# 
# #Tuning Gradient Boosted Model
# myGrid <- expand.grid(n.trees = c(1000),
#                       interaction.depth = c(7, 10),
#                       shrinkage = c(0.1, 0.2),
#                       n.minobsinnode = c(7, 10))
# 
# gbm_tree_tune1000 <- train(logTCP ~ .,
#                        data=dftrain1000,
#                        method="gbm",
#                        distribution="gaussian",
#                        trControl = trainctrl, 
#                        tuneGrid = myGrid, 
#                        verbose=FALSE)
# gbm_tree_tune1000

bestgrid1000 <- expand.grid(n.trees = 1000,
                        interaction.depth = 10,
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

gbm_tree1000 <- train(logTCP ~ .,
                     data=dftrain1000,
                     method="gbm",
                     distribution="gaussian",
                     trControl = trainctrl,
                     tuneGrid = bestgrid1000,
                     verbose = FALSE)
gbm_tree1000

gbm_train1000 <- predict(gbm_tree1000, newdata = dftrain1000)
gbm_test1000 <- predict(gbm_tree1000, newdata = dftest1000)

####Error TRAINING##############################################################################
MAE_CARTtrain1000 <- MAE(rpart_train1000, dftrain1000$logTCP)
MAE_CARTtest1000 <- MAE(rpart_test1000, dftest1000$logTCP)

#randomForest
MAE_rFtrain1000 <- MAE(rF_train1000, dftrain1000$logTCP)
MAE_rFtest1000 <- MAE(rF_test1000, dftest1000$logTCP)
MAE_rFtest1000

#RF caret
MAE_RFtrain1000 <- MAE(rfcaret_train1000, dftrain1000$logTCP)
MAE_RFtest1000 <- MAE(rfcaret_test1000, dftest1000$logTCP)

#Ranger
MAE_RANGERtrain1000 <- MAE(ranger_train1000, dftrain1000$logTCP)
MAE_RANGERtest1000 <- MAE(ranger_test1000, dftest1000$logTCP)

#GBM
MAE_GBMtrain1000 <- MAE(gbm_train1000, dftrain1000$logTCP)
MAE_GBMtest1000 <- MAE(gbm_test1000, dftest1000$logTCP)

# RMSE(predicted, original)#######################################
#CART
RMSE_CARTtrain1000 <- RMSE(rpart_train1000, dftrain1000$logTCP)
RMSE_CARTtest1000 <- RMSE(rpart_test1000, dftest1000$logTCP)

#randomForest
RMSE_rFtrain1000 <- RMSE(rF_train1000, dftrain1000$logTCP)
RMSE_rFtest1000 <- RMSE(rF_test1000, dftest1000$logTCP)
RMSE_rFtest1000

#RF caret
RMSE_RFtrain1000 <- RMSE(rfcaret_train1000, dftrain1000$logTCP)
RMSE_RFtest1000 <- RMSE(rfcaret_test1000, dftest1000$logTCP)

#Ranger
RMSE_RANGERtrain1000 <- RMSE(ranger_train1000, dftrain1000$logTCP)
RMSE_RANGERtest1000 <- RMSE(ranger_test1000, dftest1000$logTCP)

#GBM
RMSE_GBMtrain1000 <- RMSE(gbm_train1000, dftrain1000$logTCP)
RMSE_GBMtest1000 <- RMSE(gbm_test1000, dftest1000$logTCP)

# R2(predicted, original)#######################################
R2_CARTtrain1000 <- R2(rpart_train1000, dftrain1000$logTCP)
R2_CARTtest1000 <- R2(rpart_test1000, dftest1000$logTCP)

#randomForest
R2_rFtrain1000 <- R2(rF_train1000, dftrain1000$logTCP)
R2_rFtest1000 <- R2(rF_test1000, dftest1000$logTCP)

#RF caret
R2_RFtrain1000 <- R2(rfcaret_train1000, dftrain1000$logTCP)
R2_RFtest1000 <- R2(rfcaret_test1000, dftest1000$logTCP)

#Ranger
R2_RANGERtrain1000 <- R2(ranger_train1000, dftrain1000$logTCP)
R2_RANGERtest1000 <- R2(ranger_test1000, dftest1000$logTCP)

#GBM
R2_GBMtrain1000 <- R2(gbm_train1000, dftrain1000$logTCP)
R2_GBMtest1000 <- R2(gbm_test1000, dftest1000$logTCP)

# # Error diagnostics for all models
# #error_AllModels <- data.frame(mae_p1_cart, rmse_p1_cart, r2_p1_cart))

a1000 <- c('CART', MAE_CARTtrain1000, RMSE_CARTtrain1000, R2_CARTtrain1000, MAE_CARTtest1000, RMSE_CARTtest1000, R2_CARTtest1000)
b1000 <- c('rF', MAE_rFtrain1000, RMSE_rFtrain1000, R2_rFtrain1000, MAE_rFtest1000, RMSE_rFtest1000, R2_rFtest1000)
c1000 <- c('RFcaret', MAE_RFtrain1000, RMSE_RFtrain1000, R2_RFtrain1000, MAE_RFtest1000, RMSE_RFtest1000, R2_RFtest1000)
d1000 <- c('Ranger', MAE_RANGERtrain1000, RMSE_RANGERtrain1000, R2_RANGERtrain1000, MAE_RANGERtest1000, RMSE_RANGERtest1000, R2_RANGERtest1000)
e1000 <- c('GBM', MAE_GBMtrain1000, RMSE_GBMtrain1000, R2_GBMtrain1000, MAE_GBMtest1000, RMSE_GBMtest1000, R2_GBMtest1000)
f1000 <- c('xgbTree')
g1000 <- c('xgboost')

error_all1000 <- rbind(a1000,b1000,c1000,d1000,e1000)
error_all1000 <- as.data.table(error_all1000)
class(error_all1000)

names(error_all1000) <- c('Model', 'MAE Train', 'RMSE Train', 'R2 Train', 'MAE Test', 'RMSE Test', 'R2 Test')
error_all1000
print(error_all1000)

write.csv(error_all1000, file = "model_error_caret1000.csv", row.names=FALSE)

############################BOOSTED MODELS#################################################
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
df1000x <- read.csv(file = 'data/EV1000_HH.csv')

# Take a look at df structure and summary
str(df1000x)
summary(df1000x)
dim(df1000x)

# Count number of rows with TCP level == 0
colSums(df1000x == 0)

# Find annoying NA values in df
colSums(is.na(df1000x))
colnames(df1000x)

#Remove ID column #1 and other unneeded columns
df1000x[c(1, 2, 3, 4)] <- NULL
colnames(df1000x)
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
  LU1945=as.factor(LU1945),
  LU1960=as.factor(LU1960),
  LU1975=as.factor(LU1975),
  LU1990=as.factor(LU1990),
  LU2005=as.factor(LU2005),
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
as.matrix(dftrain1000x)
as.matrix(dftest1000x)
head(dftrain1000x)

##################################################################################################
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1, number = 3, 
                        #summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        allowParallel=T)
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
  x = as.matrix(dftrain1000x[, -1]),
  y = as.numeric(dftrain1000x$logTCP),
  trControl = cv.ctrl,
  tuneGrid = xgbCaret_grid_1,
  method = "xgbTree"
)

xgb_train_1$bestTune

#write and/or load the model to a file and to load file next time: 
# save(xgb_train_1, file= "xgb_train_1000.rda")
# load(file = "xgb_train_1000.rda")

#xgbCaret_grid_1$best

xgb_train1000 <- predict(xgb_train_1, newdata = dftrain1000x[, -1])
xgb_train1000
xgb_test1000 <- predict(xgb_train_1, newdata = dftest1000x[, -1])
xgb_test1000
####Error TRAINING##############################################################################
#GBM
MAE_XGBtrain1000 <- MAE(xgb_train1000, dftrain1000x$logTCP)
MAE_XGBtrain1000
MAE_XGBtest1000 <- MAE(xgb_test1000, dftest1000x$logTCP)
MAE_XGBtest1000

# RMSE(predicted, original)#######################################
#GBM
RMSE_XGBtrain1000 <- RMSE(xgb_train1000, dftrain1000x$logTCP)
RMSE_XGBtrain1000
RMSE_XGBtest1000 <- RMSE(xgb_test1000, dftest1000x$logTCP)
RMSE_XGBtest1000

# R2(predicted, original)#######################################
#GBM
R2_XGBtrain1000 <- R2(xgb_train1000, dftrain1000x$logTCP)
R2_XGBtrain1000
R2_XGBtest1000 <- R2(xgb_test1000, dftest1000x$logTCP)
R2_XGBtest1000

#caret function for error stats: 
res <- caret::postResample(vect1, vect2)