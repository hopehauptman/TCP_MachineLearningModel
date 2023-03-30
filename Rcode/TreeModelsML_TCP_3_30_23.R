#Machine Learning Models for TCP Prediction 3/9/23###MS resubmit###
##models are based on a 1000m circular buffer around well point data
###READ IN LIBRARIES#########################################################################
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
library(terra)
library(readxl)

###READ IN AND WRANGLE DATA##########################################################################
# Read in df 
setwd("C:/Users/bhauptman/Desktop/RcodeTCPmodel")
df1000 <- read.csv(file = 'data/EV1000_HH.csv')
as.data.frame(df1000)
nrow(df1000)
colnames(df1000)

levels(df1000$HSG)

# Read in df 
setwd("C:/Users/bhauptman/Desktop/RcodeTCPmodel")
df1000 <- read.csv(file = 'data/EV1000_HH.csv')
head(df1000)

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
colnames(df1000)
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
  LU1945=as.factor(LU1945),
  LU1960=as.factor(LU1960),
  LU1975=as.factor(LU1975),
  LU1990=as.factor(LU1990),
  LU2005=as.factor(LU2005),
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


write.csv(dftrain1000, 'C:/Users/bhauptman/Desktop/RcodeTCPmodel/data/dftrain1000.csv')
write.csv(dftest1000, 'C:/Users/bhauptman/Desktop/RcodeTCPmodel/data/dftest1000.csv')

#check the dimensions of the test and training and look at distribution
nrow(df1000)
nrow(dftrain1000)
nrow(dftest1000)

dim(dftrain1000)
dim(dftest1000)

# Create a plot of training and testing (copy first) just to make sure their distributions are similar
hist_dftrain1000 <- copy(dftrain1000)
hist_dftest1000 <- copy(dftest1000)

hist_dftrain1000$set <- 'training'
hist_dftest1000$set <- 'testing'

dfgg1000 <- rbind(hist_dftrain1000, hist_dftest1000)
dfgg1000

df <- dfgg1000 %>% mutate(TCP = exp(logTCP)) %>% dplyr::select(TCP, set)

#graph test and training data raw up to 1 ug/L only to zoom in 
ggplot(df, aes(x = TCP, fill = set)) + 
  geom_histogram(bins = 100) + 
  xlim(0,1) + 
  ylim(0, 200) + 
  labs(x="TCP concentration (ul/L)", y="count", set="Data") + 
  theme_bw() +
  theme(legend.title=element_blank()) + 
  scale_fill_discrete(labels=c("Training data (n=10,652; 80%)", "Testing data (n=2,662; 20%)"))

#graph test and training data raw 
ggplot(df, aes(x = TCP, fill = set)) + 
  geom_histogram(bins = 100) + 
  xlim(0,60) + 
  ylim(0, 200) + 
  labs(x="TCP concentration (ug/L)", y="count", set="Data") + 
  theme_bw() +
  theme(legend.title=element_blank()) + 
  scale_fill_discrete(labels=c("Training data (n=10,652; 80%)", "Testing data (n=2,662; 20%)"))

#graph test and training data log
ggplot(dfgg1000, aes(x = logTCP, fill = set)) + 
  geom_histogram(bins = 100) + 
  xlim(-3,2) + geom_density(alpha = 0.1) +
  theme_bw()

###Set cross validation parameters########################################################################################
trainctrl <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3)

####1 CART MODEL ##################################################################

#create train hyperparameter ranges (note 30 is the max value allowed)
tune.gridcart <- expand.grid(maxdepth = seq(1,30,1))

#make reproducible
set.seed(112)
# fit the model
rpart_tree1000 <- train(logTCP ~ ., 
                        data=dftrain1000, 
                        method="rpart2", 
                        tuneGrid =tune.gridcart,
                        metric = "RMSE",
                        trControl = trainctrl)
rpart_tree1000

#Save the model
saveRDS(rpart_tree1000, "rpart_tree.Rds")

# Call the model from file
rpart_tree1000 <- readRDS("rpart_tree.Rds")

rpart_train <- predict(rpart_tree1000, newdata=dftrain1000)
rpart_test <- predict(rpart_tree1000, newdata=dftest1000)

write.csv(rpart_train, "C:/Users/bhauptman/Desktop/RcodeTCPmodel/rpart_train.csv")
write.csv(rpart_test, "C:/Users/bhauptman/Desktop/RcodeTCPmodel/rpart_test.csv")

#bias correct CART model ------------------------------------------------------------
#Bias correction code based on Ransom et al., 2022 Method: Belitz and Stackelberg, 2021. https://doi.org/10.1016/j.envsoft.2021.105006

rpart_train <- as.numeric(unlist(rpart_train))
rpart_test <- as.numeric(unlist(rpart_test))

##The training data (ML and obs) are sorted
sort_train_obs <- sort(dftrain1000$logTCP)
sort_train_predCART <- sort(rpart_train)

# For each value predicted by the model, find the matching observed value (based on sorted vectors)
train_pred_correctedCART <- approx(sort_train_predCART, sort_train_obs, rpart_train, ties = mean)$y

# get the slope of the fitted line between predicted and corrected values
ltf_slopeCART <- coef(lm(rpart_train ~ train_pred_correctedCART))[[2]]

# Apply method to hold-out data
# For a given value of uncorrected hold out predictions, we interpolate using the ordered (x,y) pairs: (PredEst, train_pred_corrected)
holdout_pred_correctedCART <- approx(rpart_train, train_pred_correctedCART, rpart_test, ties = mean)$y
summary(holdout_pred_correctedCART)

# Second step: use QQ_slope to adjust the extreme values that led to nulls (this has no effect on the hold out predictions because none outside range)
holdout_pred_correctedCART <- ifelse(rpart_test < min(rpart_train), min(train_pred_correctedCART) - ltf_slope *(min(rpart_train) - rpart_test), holdout_pred_correctedCART)
holdout_pred_correctedCART <- ifelse(rpart_test > max(rpart_train), max(train_pred_correctedCART) + ltf_slope *(rpart_test - max(rpart_train)), holdout_pred_correctedCART)
summary(holdout_pred_correctedCART)

# calculate bias corrected fit stats for points
## fit statistics for training data (should match reported values in results section)
round(rPearson(train_pred_correctedCART, dftrain1000$logTCP)^2,2)
round(RMSE(train_pred_correctedCART, dftrain1000$logTCP),2)
#
## fit statistics for hold-out data  (should match reported values in results section)
round(rPearson(holdout_pred_correctedCART, dftest1000$logTCP)^2,2)
round(RMSE(holdout_pred_correctedCART, dftest1000$logTCP),2)

# calculate fit stats for CDF corrected values
cdf_quantsCART <- seq(0,1,.01)
#
train_quantCART <- quantile(train_pred_correctedCART, probs = cdf_quantsCART, type = 5) # type = 5 is Hazen formula preferred by Helsel and Hirsch/Hydro preferred
holdout_quantCART <- quantile(holdout_pred_correctedCART, probs = cdf_quantsCART, type = 5)
#
train_quant_obsCART <- quantile(dftrain1000$logTCP, probs = cdf_quantsCART, type = 5) 
holdout_quant_obsCART <- quantile(dftest1000$logTCP, probs = cdf_quantsCART, type = 5)
#
#training EDM corrected
# should be 0, serves as internal check
round(RMSE(train_quantCART, train_quant_obsCART),2)
#
#holdout EDM corrected
round(RMSE(holdout_quantCART, holdout_quant_obsCART),2)

####2 RANDOM FOREST ##################################################################

#create a grid with different hyperparameters
rftunegrid1000 <- expand.grid(mtry = c(16))

#make reproducible
set.seed(222)
#Now train with above hyperparameters
rfcaret_1000 <- train(logTCP ~ ., method="rf", 
                      data = dftrain1000, 
                      tuneGrid = rftunegrid1000, 
                      metric='RMSE',
                      ntree=1000,
                      trControl = trainctrl)
rfcaret_1000

#save file
saveRDS(rfcaret_1000, "rfcaret_1000.Rds")

#Call model from file
rfcaret_1000 <- readRDS("rfcaret_1000.Rds")

#apply model to holdout data
rfcaret_train1000 <- predict(rfcaret_1000, newdata=dftrain1000)
rfcaret_test1000 <- predict(rfcaret_1000, newdata=dftest1000)

write.csv(rfcaret_test1000, "C:/Users/bhauptman/Desktop/RcodeTCPmodel/data/rfcaret_test1000.csv")
write.csv(rfcaret_train1000, "C:/Users/bhauptman/Desktop/RcodeTCPmodel/data/rfcaret_train1000.csv")


#bias correct the RF model ------------------------------------------------------------

##The training data (ML and obs) are sorted
rfcaret_train1000 <- as.numeric(unlist(rfcaret_train1000))
rfcaret_test1000 <- as.numeric(unlist(rfcaret_test1000))

sort_train_obs <- sort(dftrain1000$logTCP)
sort_train_predRF <- sort(rfcaret_train1000)

# For each value predicted by the model, find the matching observed value (based on sorted vectors)
train_pred_correctedRF <- approx(sort_train_predRF, sort_train_obs, rfcaret_train1000, ties = mean)$y

# get the slope of the fitted line between predicted and corrected values
ltf_slopeRF <- coef(lm(rfcaret_train1000 ~ train_pred_correctedRF))[[2]]

# Apply method to hold-out data
# For a given value of uncorrected hold out predictions, we interpolate using the ordered (x,y) pairs: (PredEst, train_pred_corrected)
holdout_pred_correctedRF <- approx(rfcaret_train1000, train_pred_correctedRF, rfcaret_test1000, ties = mean)$y
summary(holdout_pred_correctedRF)

# Second step: use QQ_slope to adjust the extreme values that led to nulls (this has no effect on the hold out predictions because none outside range)
holdout_pred_correctedRF <- ifelse(rfcaret_test1000 < min(rfcaret_train1000), min(train_pred_correctedRF) - ltf_slope *(min(rfcaret_train1000) - rfcaret_test1000), holdout_pred_correctedRF)
holdout_pred_correctedRF <- ifelse(rfcaret_test1000 > max(rfcaret_train1000), max(train_pred_correctedRF) + ltf_slope *(rfcaret_test1000 - max(rfcaret_train1000)), holdout_pred_correctedRF)
summary(holdout_pred_correctedRF)
length(holdout_pred_correctedRF)
typeof(train_pred_correctedRF)

summary(train_pred_correctedRF)

#values reported in Table
# calculate bias corrected fit stats for points
# fit statistics for training data (should match reported values in results section)
round(rPearson(train_pred_correctedRF, dftrain1000$logTCP)^2,2)
round(RMSE(train_pred_correctedRF, dftrain1000$logTCP),2)
#
# fit statistics for hold-out data  (should match reported values in results section)
round(rPearson(holdout_pred_correctedRF, dftest1000$logTCP)^2,2)
round(RMSE(holdout_pred_correctedRF, dftest1000$logTCP),2)

# calculate fit stats for CDF corrected values
cdf_quantsRF <- seq(0,1,.01)
#
train_quantRF <- quantile(train_pred_correctedRF, probs = cdf_quantsRF, type = 5) # type = 5 is Hazen formula preferred by Helsel and Hirsch/Hydro preferred
holdout_quantRF <- quantile(holdout_pred_correctedRF, probs = cdf_quantsRF, type = 5)
#
train_quant_obsRF <- quantile(dftrain1000$logTCP, probs = cdf_quantsRF, type = 5) 
holdout_quant_obsRF <- quantile(dftest1000$logTCP, probs = cdf_quantsRF, type = 5)
#
#training EDM corrected
# should be 0, serves as internal check
round(RMSE(train_quantRF, train_quant_obsRF),2)
#
#holdout EDM corrected
#should match Table 3 mtry 14 model
round(RMSE(holdout_quantRF, holdout_quant_obsRF),2)

#------------------------------------------------------------
plot(exp(rfcaret_test1000), exp(dftest1000$logTCP))
obs_pred <- data.frame(pred = exp(rfcaret_test1000), obs = exp(dftest1000$logTCP))

# plot (not bias corrected)
obs_predPlot <- ggplot(obs_pred, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")

obs_predPlot
#still need to put in this info below
# plot (with the Bias corrected)
obs_predPlotBiasCorrect <- ggplot(obs_pred, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")

obs_predPlotBiasCorrect

#Since this is the best model we will also look at a variable importance table#########################
library("readxl")
library(ggplot2)
library(data.table)

varImp(rfcaret_1000)
ggplot2::ggplot(varImp(rfcaret_1000), top = 20)

caretVarImp <- varImp(rfcaret_1000)
caretVarImp

imp <- caretVarImp$importance %>% 
  rownames_to_column('variable') %>%
  as.data.table()
imp

as.factor(imp$variable)
as.factor(imp$Overall)
imp$variable

write_xlsx(imp,"C:/Users/bhauptman/Desktop/RcodeTCPmodel/imp.xlsx")

#go into imp and manually change to top 20 values

#Top 25 only ?? not sure how 
my_data <- read_excel("imp20maj.xlsx")
as.data.frame(my_data)

plot <- ggplot(data=my_data, aes(x = reorder(variable, Overall), y = Overall)) + geom_bar(stat='identity')+
  coord_flip() + xlab("Variable Name") + ylab("Importance") + theme_bw() 
plot


#PDP Single Variable plots###############################
library(gridExtra)
#model names
rfcaret_1000$ptype

pdp1 <- pdp::partial(rfcaret_1000, pred.var = "Precip2005")
plot.pdp1 <- ggplot2::autoplot(pdp1, contour = TRUE, ylab="logTCP ug/L", xlab='Precipitation 2005')
plot.pdp1

pdp2 <- pdp::partial(rfcaret_1000, pred.var = "Precip1990")
plot.pdp2 <- ggplot2::autoplot(pdp2, contour = TRUE, ylab="logTCP ug/L", xlab='Precipitation 1990')
plot.pdp2

pdp3 <- pdp::partial(rfcaret_1000, pred.var = "Precip1960")
plot.pdp3 <- ggplot2::autoplot(pdp3, contour = TRUE, ylab="logTCP ug/L", xlab='Precipitation 1960')
plot.pdp3

pdp4 <- pdp::partial(rfcaret_1000, pred.var = "Precip1975")
plot.pdp4 <- ggplot2::autoplot(pdp4, contour = TRUE, ylab="logTCP ug/L", xlab='Precipitation 1975')
plot.pdp4

pdp5 <- pdp::partial(rfcaret_1000, pred.var = "DO2")
plot.pdp5 <- ggplot2::autoplot(pdp5, contour = TRUE, ylab="logTCP ug/L", xlab='Dissolved oxygen proability < 2ug/L')
plot.pdp5

pdp6 <- pdp::partial(rfcaret_1000, pred.var = "DO5")
plot.pdp6 <- ggplot2::autoplot(pdp6, contour = TRUE, ylab="logTCP ug/L", xlab='Dissolved oxygen proability < 0.5ug/L')
plot.pdp6

pdp7 <- pdp::partial(rfcaret_1000, pred.var = "NO3_180")
plot.pdp7 <- ggplot2::autoplot(pdp7, contour = TRUE, ylab="logTCP ug/L", xlab='Nitrate concentration 180 feet')
plot.pdp7

pdp8 <- pdp::partial(rfcaret_1000, pred.var = "Irrig1960")
plot.pdp8 <- ggplot2::autoplot(pdp8, contour = TRUE, ylab="logTCP ug/L", xlab='1960 Irrigation nitrate levels')
plot.pdp8

pdp9 <- pdp::partial(rfcaret_1000, pred.var = "NO3_400")
plot.pdp9 <- ggplot2::autoplot(pdp9, contour = TRUE, ylab="logTCP ug/L", xlab='Nitrate concentration 400 feet')
plot.pdp9

pdp10 <- pdp::partial(rfcaret_1000, pred.var = "Irrig1975")
plot.pdp10 <- ggplot2::autoplot(pdp10, contour = TRUE, ylab="logTCP ug/L", xlab='1975 Irrigation nitrate levels')
plot.pdp10

pdp11 <- pdp::partial(rfcaret_1000, pred.var = "Irrig2005")
plot.pdp11 <- ggplot2::autoplot(pdp11, contour = TRUE, ylab="logTCP ug/L", xlab='2005 Irrigation nitrate levels')
plot.pdp11

pdp12 <- pdp::partial(rfcaret_1000, pred.var = "Irrig1990")
plot.pdp12 <- ggplot2::autoplot(pdp12, contour = TRUE, ylab="logTCP ug/L", xlab='1990 Irrigation nitrate levels')
plot.pdp12

pdp13 <- pdp::partial(rfcaret_1000, pred.var = "Age")
plot.pdp13 <- ggplot2::autoplot(pdp13, contour = TRUE, ylab="logTCP ug/L", xlab='Age')
plot.pdp13

pdp14 <- pdp::partial(rfcaret_1000, pred.var = "LU1945")
plot.pdp14 <- ggplot2::autoplot(pdp14, contour = TRUE, ylab="logTCP ug/L", xlab='Land Use 1945')
plot.pdp14

pdp15 <- pdp::partial(rfcaret_1000, pred.var = "HSG")
plot.pdp15 <- ggplot2::autoplot(pdp15, contour = TRUE, ylab="logTCP ug/L", xlab='Hydrologic soil group (HSG)')
plot.pdp15

pdp16 <- pdp::partial(rfcaret_1000, pred.var = "Drainage")
plot.pdp16 <- ggplot2::autoplot(pdp16, contour = TRUE, ylab="logTCP ug/L", xlab='Drainage Class')
plot.pdp16

pdp17 <- pdp::partial(rfcaret_1000, pred.var = "LU1960")
plot.pdp17 <- ggplot2::autoplot(pdp17, contour = TRUE, ylab="logTCP ug/L", xlab='Land Use 1960')
plot.pdp17

pdp18 <- pdp::partial(rfcaret_1000, pred.var = "LU1975")
plot.pdp18 <- ggplot2::autoplot(pdp18, contour = TRUE, ylab="logTCP ug/L", xlab='Land Use 1975')
plot.pdp18

pdp19 <- pdp::partial(rfcaret_1000, pred.var = "LU1990")
plot.pdp19 <- ggplot2::autoplot(pdp19, contour = TRUE, ylab="logTCP ug/L", xlab='Land Use 1990')
plot.pdp19

pdp20 <- pdp::partial(rfcaret_1000, pred.var = "LU2005")
plot.pdp20 <- ggplot2::autoplot(pdp20, contour = TRUE, ylab="logTCP ug/L", xlab='Land Use 2005')
plot.pdp20


grid.arrange(plot.pdp1, plot.pdp2, plot.pdp3, plot.pdp4, plot.pdp5, plot.pdp6, plot.pdp7, plot.pdp8, 
             plot.pdp9, plot.pdp10, plot.pdp11, plot.pdp12, plot.pdp13, plot.pdp14, plot.pdp15, plot.pdp16, 
             plot.pdp17, plot.pdp18,  plot.pdp19,  plot.pdp20)
             ncol=4)

#######3 BRT#########################################################################################
#a. 'gbm' in caret
##note other values for expand.grid were used duing v training/hyperparmeter testing (see supplement for ranges used) 

gbmGrid <-  expand.grid(interaction.depth = 12, 
                        n.trees = 3000,
                        shrinkage = 0.05,
                        n.minobsinnode = 10)

set.seed(777)
gbmFit <- train(logTCP ~ ., method = "gbm", 
                data = dftrain1000, 
                trControl = trainctrl, 
                tuneGrid = gbmGrid, 
                metric='RMSE', 
                verbose = FALSE)
str(dftrain1000)

gbm_train <- predict(gbmFit, newdata = dftrain1000)
gbm_test <- predict(gbmFit, newdata = dftest1000)

#save file
saveRDS(gbmFit, "gbmFit.Rds")

# to read later 
gbmFit <- readRDS("gbmFit.Rds")

#bias correct BRT model, ------------------------------------------------------------

# The training data (ML and obs) are sorted

##The training data (ML and obs) are sorted
gbm_train <- as.numeric(unlist(gbm_train))
gbm_test <- as.numeric(unlist(gbm_test))

sort_train_obs <- sort(dftrain1000$logTCP)
sort_train_predBRT <- sort(gbm_train)

# For each value predicted by the model, find the matching observed value (based on sorted vectors)
train_pred_correctedBRT <- approx(sort_train_predBRT, sort_train_obs, gbm_train, ties = mean)$y

# Get the slope of the fitted line between predicted and corrected values
ltf_slopeBRT <- coef(lm(gbm_train ~ train_pred_correctedBRT))[[2]]

# Apply method to hold-out data
# For a given value of uncorrected hold out predictions, we interpolate using the ordered (x,y) pairs: (PredEst, train_pred_corrected)
holdout_pred_correctedBRT <- approx(gbm_train, train_pred_correctedBRT, gbm_test, ties = mean)$y
summary(holdout_pred_correctedBRT)

# Second step: use QQ_slope to adjust the extreme values that led to nulls (this has no effect on the hold out predictions because none outside range)
holdout_pred_correctedBRT <- ifelse(gbm_test < min(gbm_train), min(train_pred_correctedBRT) - ltf_slopeBRT *(min(gbm_train) - gbm_test), holdout_pred_correctedBRT)
holdout_pred_correctedBRT <- ifelse(gbm_test > max(gbm_train), max(train_pred_correctedBRT) + ltf_slopeBRT *(gbm_test - max(gbm_train)), holdout_pred_correctedBRT)
summary(holdout_pred_correctedBRT)

#values reported in Table 3 for mtry 20 model
# calculate bias corrected fit stats for points
# fit statistics for training data (should match reported values in results section)
round(rPearson(train_pred_correctedBRT, dftrain1000$logTCP)^2,2)
round(RMSE(train_pred_correctedBRT, dftrain1000$logTCP),2)
#
# fit statistics for hold-out data  (should match reported values in results section)
round(rPearson(holdout_pred_correctedBRT, dftest1000$logTCP)^2,2)
round(RMSE(holdout_pred_correctedBRT, dftest1000$logTCP),2)

# calculate fit stats for CDF corrected values
cdf_quantsBRT <- seq(0,1,.01)
#
train_quantBRT <- quantile(train_pred_correctedBRT, probs = cdf_quantsBRT, type = 5) # type = 5 is Hazen formula preferred by Helsel and Hirsch/Hydro preferred
holdout_quantBRT <- quantile(holdout_pred_correctedBRT, probs = cdf_quantsBRT, type = 5)
#
train_quant_obsBRT <- quantile(dftrain1000$logTCP, probs = cdf_quantsBRT, type = 5) 
holdout_quant_obsBRT <- quantile(dftest1000$logTCP, probs = cdf_quantsBRT, type = 5)
#
#training EDM corrected
# should be 0, serves as internal check
round(RMSE(train_quantBRT, train_quant_obsBRT),2)
#
#holdout EDM corrected
#should match Table 3 mtry 14 model
round(RMSE(holdout_quantBRT, holdout_quant_obsBRT),2)

#Error states for all models##########################################
## MAE(predicted, original)
#CART
MAE_CARTtrain <- MAE(rpart_train, dftrain1000$logTCP)
MAE_CARTtrain
MAE_CARTtrainCorrected <- MAE(train_pred_correctedCART, dftrain1000$logTCP)
MAE_CARTtrainCorrected
MAE_CARTtest <- MAE(rpart_test, dftest1000$logTCP)
MAE_CARTtest
MAE_CARTtestCorrected <- MAE(holdout_pred_correctedCART, dftest1000$logTCP)
MAE_CARTtestCorrected

#RF
MAE_RFtrain <- MAE(rfcaret_train1000, dftrain1000$logTCP)
MAE_RFtrain
MAE_RFtrainCorrected <- MAE(train_pred_correctedRF, dftrain1000$logTCP)
MAE_RFtrainCorrected
MAE_RFtest <- MAE(rfcaret_test1000, dftest1000$logTCP)
MAE_RFtest
MAE_RFtestCorrected <- MAE(holdout_pred_correctedRF, dftest1000$logTCP)
MAE_RFtestCorrected

#BRT
MAE_GBMtrain <- MAE(gbm_train, dftrain1000$logTCP)
MAE_GBMtrain
MAE_GBMtrainCorrected <- MAE(train_pred_correctedBRT, dftrain1000$logTCP)
MAE_GBMtrainCorrected
MAE_GBMtest <- MAE(gbm_test, dftest1000$logTCP)
MAE_GBMtest
MAE_GBMtestCorrected <- MAE(holdout_pred_correctedBRT, dftest1000$logTCP)
MAE_GBMtestCorrected 

##RMSE(predicted, original)
#CART
RMSE_CARTtrain <- RMSE(rpart_train, dftrain1000$logTCP)
RMSE_CARTtrain
RMSE_CARTtrainCorrected <- RMSE(train_pred_correctedCART, dftrain1000$logTCP)
RMSE_CARTtrainCorrected
RMSE_CARTtest <- RMSE(rpart_test, dftest1000$logTCP)
RMSE_CARTtest
RMSE_CARTtestCorrected <- RMSE(holdout_pred_correctedCART, dftest1000$logTCP)
RMSE_CARTtestCorrected

#RF
RMSE_RFtrain <- RMSE(rfcaret_train1000, dftrain1000$logTCP)
RMSE_RFtrain
RMSE_RFtrainCorrected <- RMSE(train_pred_correctedRF, dftrain1000$logTCP)
RMSE_RFtrainCorrected
RMSE_RFtest <- RMSE(rfcaret_test1000, dftest1000$logTCP)
RMSE_RFtest
RMSE_RFtestCorrected <- RMSE(holdout_pred_correctedRF, dftest1000$logTCP)
RMSE_RFtestCorrected

#BRT
RMSE_GBMtrain <- RMSE(gbm_train, dftrain1000$logTCP)
RMSE_GBMtrain
RMSE_GBMtrainCorrected <- RMSE(train_pred_correctedBRT, dftrain1000$logTCP)
RMSE_GBMtrainCorrected
RMSE_GBMtest <- RMSE(gbm_test, dftest1000$logTCP)
RMSE_GBMtest
RMSE_GBMtestCorrected <- RMSE(holdout_pred_correctedBRT, dftest1000$logTCP)
RMSE_GBMtestCorrected

##R2(predicted, original)
#CART
R2_CARTtrain <- R2(rpart_train, dftrain1000$logTCP)
R2_CARTtrain
R2_CARTtrainCorrected <- R2(train_pred_correctedCART, dftrain1000$logTCP)
R2_CARTtrainCorrected
R2_CARTtest <- R2(rpart_test, dftest1000$logTCP)
R2_CARTtest 
R2_CARTtestCorrected <- R2(holdout_pred_correctedCART, dftest1000$logTCP)
R2_CARTtestCorrected 

#RF
R2_RFtrain <- R2(rfcaret_train1000, dftrain1000$logTCP)
R2_RFtrain
R2_RFtrainCorrected <- R2(train_pred_correctedRF, dftrain1000$logTCP)
R2_RFtrainCorrected
R2_RFtest <- R2(rfcaret_test1000, dftest1000$logTCP)
R2_RFtest
R2_RFtestCorrected <- R2(holdout_pred_correctedRF, dftest1000$logTCP)
R2_RFtestCorrected

#BRT
R2_GBMtrain <- R2(gbm_train, dftrain1000$logTCP)
R2_GBMtrain
R2_GBMtrainCorrected <- R2(train_pred_correctedBRT, dftrain1000$logTCP)
R2_GBMtrainCorrected
R2_GBMtest <- R2(gbm_test, dftest1000$logTCP)
R2_GBMtest
R2_GBMtestCorrected <- R2(holdout_pred_correctedBRT, dftest1000$logTCP)
R2_GBMtestCorrected

#Combine all error stats in a dataframe
a1000 <- c('CART', MAE_CARTtrain, MAE_CARTtrainCorrected, RMSE_CARTtrain, RMSE_CARTtrainCorrected, R2_CARTtrain, R2_CARTtrainCorrected, 
           MAE_CARTtest, MAE_CARTtestCorrected, RMSE_CARTtest, RMSE_CARTtestCorrected, R2_CARTtest, R2_CARTtestCorrected)

b1000 <- c('RF', MAE_RFtrain, MAE_RFtrainCorrected, RMSE_RFtrain, RMSE_RFtrainCorrected, R2_RFtrain, R2_RFtrainCorrected, 
           MAE_RFtest, MAE_RFtestCorrected, RMSE_RFtest, RMSE_RFtestCorrected, R2_RFtest, R2_RFtestCorrected)

c1000 <- c('BRT', MAE_GBMtrain, MAE_GBMtrainCorrected, RMSE_GBMtrain, RMSE_GBMtrainCorrected, R2_GBMtrain, R2_GBMtrainCorrected, 
           MAE_GBMtest, MAE_GBMtestCorrected, RMSE_GBMtest, RMSE_GBMtestCorrected, R2_GBMtest, R2_GBMtestCorrected)

error_all1000 <- rbind(a1000,b1000,c1000)
error_all1000 <- as.data.table(error_all1000)

names(error_all1000) <- c('Model', 'MAE Train', 'MAE_RFtrain Corrected', 
                          'RMSE Train', 'RMSE Train Corrected', 
                          'R2 Train', 'R2 Train Corrected',
                          'MAE Test', 'MAE Test Corrected',
                          'RMSE Test', 'RMSE Test Corrected',
                          'R2 Test', 'R2 Test Corrected')
error_all1000

write.csv(error_all1000, file = "error_all1000.csv", row.names=FALSE)

#######################################################################################################################################################
#######################################################################################################################################################

#b. xgboost in xgboost package
traincontrol <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#put training data explanatory variables in matrix
train_x = data.matrix(dftrain1000[, -1])
train_y = dftrain1000[,1]

test_x = data.matrix(dftest1000[, -1])
test_y = dftest1000[, 1]

xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

xgbc = xgboost(data = xgb_train, max.depth = 21, nrounds = 10000, trControl = traincontrol, eta = 0.01,
               gamma = 1, colsample_bytree = 1, min_child_weight = 3, subsample = 1, verbose = F)
print(xgbc)

xgbc$bestTune

pred_y_train = predict(xgbc, xgb_train)
pred_y = predict(xgbc, xgb_test)

#Error analysis xgboost train
mse0 = mean((train_y - pred_y_train)^2)
mae0 = caret::MAE(train_y, pred_y_train)
rmse0 = caret::RMSE(train_y, pred_y_train)
R20 = caret::R2(train_y, pred_y_train)

cat("MSE: ", mse0, "MAE: ", mae0, " RMSE: ", rmse0, " R2 ", R20)

#Error analysis test
mse = mean((test_y - pred_y)^2)
mae = caret::MAE(test_y, pred_y)
rmse = caret::RMSE(test_y, pred_y)
R2 = caret::R2(test_y, pred_y)

cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse, " R2 ", R2)
