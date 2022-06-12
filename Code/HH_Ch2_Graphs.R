###############Graphs and figures for Ch 2 TCP predictive model##################################

setwd("C:/Users/bhauptman/Box/Ch2/Manuscipt and figures")

#load libraries
library(data.table)
library(ggplot2)

#########################Plot for Mass balance##########################################
MBdata <- read.csv(file = "MassBalanceGraph.csv") 

colors <- c("0.17% TCP by mass"= "steelblue", "7% TCP by mass" = "darkred")

MBplot <- ggplot(MBdata , aes(x = year)) + 
                   geom_line(aes(y=TCP.17, color="0.17% TCP by mass"), size=1.2) +
                   geom_line(aes(y=TCP7, color="7% TCP by mass"), size=1.2) +
                   labs(x="Year", y="log10 TCP Mass in kg", color="Legend") +
  scale_y_continuous(breaks=seq(0,8,1))+
  theme_bw()

MBplot

#########################VaribaleImportance plt using caret RF model######################
library("readxl")
library(ggplot2)
library(data.table)

my_data <- read_excel("C:/Users/bhauptman/Box/Ch2/R/data/Caret_VarImpPlot.xlsx")

as.data.frame(my_data)

as.factor(my_data$Importance)
as.factor(my_data$Variable)

plot <- ggplot(data=my_data, aes(x = reorder(Variable, Importance), y = Importance))+coord_flip()+geom_bar(stat = "identity")+
  xlab("Variable Name") + ylab("Relative Importance") + theme_bw()
plot

#####################################Land Use Median TCP graph#############################
library(ggplot2)
library(data.table)
library(dplyr)

LU_Data <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/LandUseEV1000.csv")
LU_Data

as.data.table(LU_Data)
str(LU_Data)

as.factor(LU_Data$MajLU2005Num)
as.factor(LU_Data$MajLU2005Name)
as.factor(LU_Data$MajLU1990Num)
as.factor(LU_Data$MajLU1990Name)
as.factor(LU_Data$MajLU1975Num)
as.factor(LU_Data$MajLU1975Name)
as.factor(LU_Data$MajLU1960Num)
as.factor(LU_Data$MajLU1960Name)
as.factor(LU_Data$MajLU1945Num)
as.factor(LU_Data$MajLU1945Name)

Med2005 <- LU_Data %>% group_by(name=MajLU2005Name) %>% summarise(medianTCP2005 = median(MeanTCP_ppb))
Med2005

Med1990 <- LU_Data %>% group_by(name=MajLU1990Name) %>% summarise(medianTCP1990 = median(MeanTCP_ppb))
Med1990

Med1975 <-LU_Data %>% group_by(name=MajLU1975Name) %>% summarise(medianTCP1975 = median(MeanTCP_ppb))
Med1975

Med1960 <-LU_Data %>% group_by(name=MajLU1960Name) %>% summarise(medianTCP1960 = median(MeanTCP_ppb))
Med1960

Med1945 <-LU_Data %>% group_by(name = MajLU1945Name) %>% summarise(medianTCP1945 = median(MeanTCP_ppb))
Med1945

df <- merge(Med2005, Med1990, by = "name") %>% merge(Med1975) %>% merge(Med1960) %>% merge(Med1945)
as.data.table(df)

setcolorder(df, c(1,6,5,4,3,2))
df

p <- ggplot(data=df, aes(x=name, y=medianTCP2005)) + 
  ylab("Median TCP Concentration (ppb)")+
  theme_bw()+
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle=45, vjust=1, hjust=1))
p

#get data in long form to graph with stupid ggplot
library(reshape2)
longdf <- melt(data = df, 
               id.vars = "name", 
               variable.name = "Year", 
               value.name = "TCP"
)
longdf 

#now plot long form in ggplot
p1 <- ggplot(data=longdf , aes(x=name, y=TCP)) + 
  ylab("Median TCP Concentration (ppb)")+
  geom_bar(aes(fill = Year), position = "dodge", stat = "identity") + 
  theme(axis.text.x = element_text(angle=45, vjust=1, hjust=1), axis.title.x=element_blank())+
  theme(legend.title = element_blank())
p1

##############obs versus predicted graphs#################################################
#read in dftrain and dftest data sets 
dftrain1000 <- read.csv(file = "C:/Users/bhauptman/Box/Ch2/R/data/dftrain1000.csv")
dftest1000 <- read.csv(file = "C:/Users/bhauptman/Box/Ch2/R/data/dftest1000.csv")

##################Caret randomForest#######################################
#read in data, put in data frame and select relevant column
rfcaret_train <- read.csv(file = "C:/Users/bhauptman/Box/Ch2/R/data/rfcaret_train1000.csv")
ncol(rfcaret_train)
as.data.frame(rfcaret_train)
rfcaret_train <- rfcaret_train[,2]
rfcaret_train

rfcaret_test <- read.csv(file = "C:/Users/bhauptman/Box/Ch2/R/data/rfcaret_test1000.csv")
ncol(rfcaret_test)
as.data.frame(rfcaret_test)
rfcaret_test <- rfcaret_test[,2]
rfcaret_test
  
obs_predTrain <- data.frame(pred = exp(rfcaret_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTrain <- data.frame(pred = rfcaret_train, obs = dftrain1000$logTCP)
#adding columns which will be used as factors
obs_pred_logTrain["TrainTest"]='train'
obs_pred_logTrain["model"]='Random Forest'

obs_predTest <- data.frame(pred = exp(rfcaret_test), obs = exp(dftest1000$logTCP))
obs_pred_logTest <- data.frame(pred = rfcaret_test, obs = dftest1000$logTCP)
#adding columns which will be used as factors
obs_pred_logTest["TrainTest"]='test'
obs_pred_logTest["model"]='Random Forest'

#combine into data frame for graphing 
df_rfCaret <- rbind(obs_pred_logTrain, obs_pred_logTest)

####this graph is log of the data for both x and y axis for TRAINING data
obs_predPlotTrainLog <- ggplot(obs_pred_logTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: Caret Random Forest    Data: Training")
obs_predPlotTrainLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainLog.png")

#this graph is log of the data for both x and y axis for TESTING data
obs_predPlotTestLog <- ggplot(obs_pred_logTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log 10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: Caret Random Forest    Data: Testing")
obs_predPlotTestLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestLog.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predTrain <- ggplot(obs_predTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: Caret Random Forest    Data: Training")
obs_predlogTrain

obs_predTest <- ggplot(obs_predTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: Caret Random Forest    Data: Testing")
obs_predTest

##################CART################################################
#construct additional obs and predicted graphs for other models
rpart_train <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/rpart_train1000.csv")
as.data.frame(rpart_train)
rpart_train <- rpart_train[,2]

rpart_test <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/rpart_test1000.csv")
as.data.frame(rpart_test)
rpart_test <- rpart_test[,2]

obs_predTrainCart <- data.frame(pred = exp(rpart_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTrainCart <- data.frame(pred = rpart_train, obs = dftrain1000$logTCP)

obs_predTestCart <- data.frame(pred = exp(rpart_test), obs = exp(dftest1000$logTCP))
obs_pred_logTestCart <- data.frame(pred = rpart_test, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrainCart["TrainTest"]='train'
obs_pred_logTrainCart["model"]='CART'

obs_pred_logTestCart["TrainTest"]='test'
obs_pred_logTestCart["model"]='CART'

#combine into one data frame for graphing 
df_CART <- rbind(obs_pred_logTrainCart, obs_pred_logTestCart)

####this graph is log of the data for both x and y axis for TRAINING data
obs_predPlotTrainCartLog <- ggplot(obs_pred_logTrainCart, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: CART    Data: Training")
obs_predPlotTrainCartLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainCartLog.png")

obs_predPlotTestCartLog <- ggplot(obs_pred_logTestCart, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: CART    Data: Testing")
obs_predPlotTestCartLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestCartLog.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predPlotTrainCart <- ggplot(obs_predTrainCart, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: CART    Data: Training")
obs_predPlotTrainCart

obs_predPlotTestCart <- ggplot(obs_predTestCart, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: CART    Data: Testing")
obs_predPlotTestCart

##################randomForest################################################
rF_train <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/rF_train1000.csv")
as.data.frame(rF_train)
rF_train <- rF_train[,2]

rF_test <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/rF_test1000.csv")
as.data.frame(rF_test)
rF_test <- rF_test[,2]

obs_predTrainRF <- data.frame(pred = exp(rF_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTrainRF <- data.frame(pred = rF_train, obs = dftrain1000$logTCP)
                                
obs_predTestRF <- data.frame(pred = exp(rF_test), obs = exp(dftest1000$logTCP))
obs_predlogTestRF <- data.frame(pred = rF_test, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrainRF["TrainTest"]='train'
obs_pred_logTrainRF["model"]='randomForest'

obs_predlogTestRF["TrainTest"]='test'
obs_predlogTestRF["model"]='randomForest'

#combine into one data frame for graphing 
df_rF <- rbind(obs_pred_logTrainRF, obs_predlogTestRF)

####this graph is log of the data for both x and y axis for TRAINING data
obs_predPlotTrainRFLog <- ggplot(obs_pred_logTrainRF, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: randomForest    Data: Training")
obs_predPlotTrainRFLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainRFLog.png")

obs_predPlotTestRFLog <- ggplot(obs_predlogTestRF, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: randomForest    Data: Testing")
obs_predPlotTestRFLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestRFLog.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predPlotTrainRF <- ggplot(obs_predTrainRF, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: randomForest    Data: Training")
obs_predPlotTrainRF

obs_predPlotTestRF <- ggplot(obs_predTestRF, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: randomForest    Data: Testing")
obs_predPlotTestRF

##################ranger################################################
ranger_train <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/ranger_train1000.csv")
as.data.frame(ranger_train)
ranger_train <- ranger_train[,2]

ranger_test <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/ranger_test1000.csv")
as.data.frame(ranger_test)
ranger_test <- ranger_test[,2]

obs_predTrainRanger <- data.frame(pred = exp(ranger_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTrainRanger <- data.frame(pred = ranger_train, obs = dftrain1000$logTCP)
                              
obs_predTestRanger <- data.frame(pred = exp(ranger_test), obs = exp(dftest1000$logTCP))
obs_pred_logTestRanger <- data.frame(pred = ranger_test, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrainRanger["TrainTest"]='train'
obs_pred_logTrainRanger["model"]='Ranger'

obs_pred_logTestRanger["TrainTest"]='test'
obs_pred_logTestRanger["model"]='Ranger'

#combine into one data frame for graphing 
df_ranger <- rbind(obs_pred_logTrainRanger, obs_pred_logTestRanger)

####this graph is log of the data for both x and y axis for TRAINING data
obs_predPlotTrainRangerLog <- ggplot(obs_pred_logTrainRanger, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: Ranger    Data: Training")
obs_predPlotTrainRangerLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainRangerLog.png")

#this graph is log of the data for both x and y axis for TESTING data
obs_predPlotTestRangerLog <- ggplot(obs_pred_logTestRanger, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: Ranger    Data: Testing")
obs_predPlotTestRangerLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestRangerLog.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predPlotTrainRanger <- ggplot(obs_predTrainRanger, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: Ranger    Data: Training")
obs_predPlotTrainRanger
####this graph is untransformed data for both x and y axis for TESTING data
obs_predPlotTestRanger <- ggplot(obs_predTestRanger, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: Ranger    Data: Testing")
obs_predPlotTestRanger

##################gbm################################################
gbm_train <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/gbm_train1000.csv")
as.data.frame(gbm_train)
gbm_train <- gbm_train[,2]

gbm_test <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/gbm_test1000.csv")
as.data.frame(gbm_test)
gbm_test <- gbm_test[,2]

obs_predTraingbm <- data.frame(pred = exp(gbm_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTraingbm <- data.frame(pred = gbm_train, obs = dftrain1000$logTCP)
                                  
obs_predTestgbm <- data.frame(pred = exp(gbm_test), obs = exp(dftest1000$logTCP))
obs_pred_logTestgbm <- data.frame(pred = gbm_test, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTraingbm["TrainTest"]='train'
obs_pred_logTraingbm["model"]='gbm'

obs_pred_logTestgbm["TrainTest"]='test'
obs_pred_logTestgbm["model"]='gbm'

#combine into one data frame for graphing 
df_gbm <- rbind(obs_pred_logTraingbm, obs_pred_logTestgbm)

####this graph is log of the data for both x and y axis for TRAINING data 
obs_predPlotTraingbmLOG <- ggplot(obs_pred_logTraingbm, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: gbm    Data: Training")
obs_predPlotTraingbmLOG
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTraingbmLOG.png")

#this graph is log of the data for both x and y axis for TESTING data 
obs_predPlotTestgbmLOG <- ggplot(obs_pred_logTestgbm, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: gbm    Data: Testing")
obs_predPlotTestgbmLOG
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestgbmLOG.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predPlotTraingbm <- ggplot(obs_predTraingbm, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: gbm    Data: Training")
obs_predPlotTraingbm
#this graph is untransformed data for both x and y axis for Teting data
obs_predPlotTestgbm <- ggplot(obs_predTestgbm, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: gbm    Data: Testing")
obs_predPlotTestgbm

##################xgb################################################
xgb_train <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/xgb_train1000.csv")
as.data.frame(xgb_train)
xgb_train <- xgb_train[,2]

xgb_test <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/xgb_test1000.csv")
as.data.frame(xgb_test)
xgb_test <- xgb_test[,2]

obs_predTrainxgb <- data.frame(pred = exp(xgb_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTrainxgb <- data.frame(pred = xgb_train, obs = dftrain1000$logTCP)
                                  
obs_predTestxgb <- data.frame(pred = exp(xgb_test), obs = exp(dftest1000$logTCP))
obs_pred_logTestxgb <- data.frame(pred = xgb_test, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrainxgb["TrainTest"]='train'
obs_pred_logTrainxgb["model"]='xgb'

obs_pred_logTestxgb["TrainTest"]='test'
obs_pred_logTestxgb["model"]='xgb'

#combine into one data frame for graphing 
df_xgb <- rbind(obs_pred_logTrainxgb, obs_pred_logTestxgb)

####this graph is log of the data for both x and y axis for TRAINING data 
obs_predPlotTrainxgbLOG <- ggplot(obs_pred_logTrainxgb, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: xgb    Data: Training")
obs_predPlotTrainxgbLOG
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainxgbLOG.png")

#this graph is log of the data for both x and y axis for TESTING data 
obs_predPlotTestxgbLOG <- ggplot(obs_pred_logTestxgb, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: xgb    Data: Testing")
obs_predPlotTestxgbLOG
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestxgbLOG.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predPlotTrainxgb <- ggplot(obs_predTrainxgb, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: xgb    Data: Training")
obs_predPlotTrainxgb
#this graph is log of the data for both x and y axis for TESTING data 
obs_predPlotTestxgbLOG <- ggplot(obs_predTestxgb, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: xgb    Data: Testing")
obs_predPlotTestxgb
  
##################xgboost################################################
xboost_train <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/pred_y_train.csv")
as.data.frame(xboost_train)
xboost_train <- xboost_train[,2]

xboost_test <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/pred_y_test.csv")
as.data.frame(xboost_test)
xboost_test <- xboost_test[,2]

obs_predTrainxboost <- data.frame(pred = exp(xboost_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTrainxboost <- data.frame(pred = xboost_train, obs = dftrain1000$logTCP)
                                  
obs_predTestxboost <- data.frame(pred = exp(xboost_test), obs = exp(dftest1000$logTCP))
obs_pred_logTestxboost <- data.frame(pred = xboost_test, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrainxboost["TrainTest"]='train'
obs_pred_logTrainxboost["model"]='Gradient Boosted'

obs_pred_logTestxboost["TrainTest"]='test'
obs_pred_logTestxboost["model"]='Gradient Boosted'

#combine into one data frame for graphing 
df_xboost <- rbind(obs_pred_logTrainxboost, obs_pred_logTestxboost)

####this graph is log of the data for both x and y axis for TRAINING data 
obs_predPlotTrainxboostLOG <- ggplot(obs_pred_logTrainxboost, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: xboost    Data: Training")
obs_predPlotTrainxboostLOG
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainxboostLOG.png")

obs_predPlotTestxboostLOG <- ggplot(obs_pred_logTestxboost, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: xboost    Data: Testing")
obs_predPlotTestxboostLOG
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestxboostLOG.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predPlotTrainxboost <- ggplot(obs_predTrainxboost, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: xboost    Data: Training")
obs_predPlotTrainxboost

obs_predPlotTestxboost <- ggplot(obs_predTestxboost, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: xboost    Data: Testing")
obs_predPlotTestxboost

###########################COMBINING GRAPHS USING rbind and FACET GRID#######################################
df_all <- rbind(df_CART, df_rfCaret, df_xboost)
as.data.frame(df_all)

df_all$model <- as.factor(df_all$model)
df_all$TrainTest <- as.factor(df_all$TrainTest)

#subset train and test data
subTrain <- subset(df_all, TrainTest == "train")

subTest <- subset(df_all, TrainTest == 'test')

# #copy data  
# subTrain1 <- subTrain
# levels(subTrain1$model)
# subTrain1$model <- factor(subTrain1$model, levels= "rfCaret", "CART", "gbm", "randomForest", "Ranger", "xboost", "xgb")
# 
# #copy data
# subTest1 <- subTest
# subTest1$model <- factor(subTest1$model, levels="CART", "rfCaret", "randomForest", "Ranger", "gbm", "xgb","xboost")

# plot by facet wrap for training and testing
ptrain <- ggplot(subTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Data: Training")
ptraingrid <-ptrain + facet_wrap(~model)
ptraingrid
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/ptraingrid.png")

ptest <- ggplot(subTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Data: Testing")
ptestgrid <- ptest + facet_wrap(~model)
ptestgrid
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/ptestgrid.png")

######################################################################################################################
###ERROR for 3 models

# Read in df 
library(readxl)
library(ggplot2)
library(tidyr)

setwd("C:/Users/bhauptman/Box/Ch2/R")
error3 <- read_excel('data/ggplot_error3.xlsx')
error3 <- transform(
  error3,
  Metric =as.factor(Metric),
  Model =as.factor(Model),
  Name =as.factor(Name),
  Buffer=as.factor(Buffer)
)
head(error3)

levels(error3$Metric)

#Set facet wrap labels
my_labeller <- as_labeller(c( MAE = 'MAE', RMSE = 'RMSE', `R2` = 'R^2'), default = label_parsed)
my_labeller

# Line plot with multiple groups
ggplot(data=error3, aes(x=Buffer, y=Value, group=Model)) +  
  theme_bw() +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model)) +
  labs(y = "Value", x = 'Buffer Radius (m)') +
  facet_wrap(~ Metric, nrow = 1, scales = "free_y", 
             labeller=my_labeller)

# Reorder line plot so R2 comes last
library(tidyverse)
library(dplyr)

error3 %>% mutate(across(Metric, factor, levels=c('MAE','RMSE','R2'))) %>%
  ggplot(aes(x=Buffer, y=Value, group=Model))+
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model)) +
  theme_bw() +
  labs(y = "Value", x = 'Buffer radius (m)') +
  facet_wrap(~ Metric, nrow = 3, scales = "free_y", labeller=my_labeller)

#default level order
levels(error3$Model)
#change the order of labels
error3$Model <- factor(error3$Model, levels = c("RandomForest", "xgboost", "CART"))

#add superscript for R^2
error3 %>% mutate(across(Metric, factor, levels=c("MAE" , "RMSE", "R2"), labels = c('MAE', 'RMSE', 'R2'))) %>%
  ggplot(aes(x=Buffer, y=Value, group=Model))+
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model)) +
  theme_bw() +
  labs(y = "Value", x = 'Buffer radius (m)') +
  facet_grid(~ Metric, labeller=my_labeller) +
  labs(color='Model name') 


#change order of panels: https://stackoverflow.com/questions/15116081/controlling-order-of-facet-grid-facet-wrap-in-ggplot2










###ERROR for 6 graphs

# Read in df 
library(readxl)
library(ggplot2)
library(tidyr)

setwd("C:/Users/bhauptman/Box/Ch2/R")
error <- read_excel('data/ggplot_error.xlsx')
error <- transform(
  error,
  Metric =as.factor(Metric),
  Model =as.factor(Model),
  Name =as.factor(Name),
  Buffer=as.factor(Buffer)
)
head(error)

levels(error$Metric)

#Set facet wrap labels
my_labeller <- as_labeller(c( MAE = 'MAE', RMSE = 'RMSE', `R^2` = 'R^2'), default = label_parsed)
my_labeller

# Line plot with multiple groups
ggplot(data=error, aes(x=Buffer, y=Value, group=Model)) +  
  theme_bw() +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model)) +
  labs(y = "Value", x = 'Buffer radius (m)') +
  facet_wrap(~ Metric, nrow = 3, scales = "free_y", 
             labeller=my_labeller)

# Reorder line plot so R2 comes last
library(tidyverse)
library(dplyr)
  
error %>% mutate(across(Metric, factor, levels=c('MAE','RMSE','R2'))) %>%
  ggplot(aes(x=Buffer, y=Value, group=Model))+
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model)) +
  theme_bw() +
  labs(y = "Value", x = 'Buffer radius (m)') +
  facet_wrap(~ Metric, nrow = 3, scales = "free_y", labeller=my_labeller)

#default level order
levels(error$Model)
#change the order of labels
error$Model <- factor(error$Model, levels = c("rf", "Ranger", "xgboost", "xgbTree", "GBM", "rpart"))

#add superscript for R^2
error %>% mutate(across(Metric, factor, levels=c("MAE" , "RMSE", "R2"), labels = c('MAE', 'RMSE', 'R2'))) %>%
  ggplot(aes(x=Buffer, y=Value, group=Model))+
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model)) +
  theme_bw() +
  labs(y = "Value", x = 'Buffer radius (m)') +
  facet_grid(~ Metric, labeller=my_labeller) +
  labs(color='Model name') 


#change order of panels: https://stackoverflow.com/questions/15116081/controlling-order-of-facet-grid-facet-wrap-in-ggplot2
