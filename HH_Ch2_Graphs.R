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
                   labs(x="Year", y="log10 TCP concentration in kg", color="Legend") +
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

##################randomForest#######################################
#read in data, put in data frame and select relevant column
rfcaret_train <- read.csv(file = "C:/Users/bhauptman/Box/Ch2/R/data/rfcaret_train1000.csv")
ncol(rfcaret_train)
as.data.frame(rfcaret_train)
rfcaret_train <- rfcaret_train[,2]

rfcaret_test <- read.csv(file = "C:/Users/bhauptman/Box/Ch2/R/data/rfcaret_test1000.csv")
ncol(rfcaret_test)
as.data.frame(rfcaret_test)
rfcaret_test <- rfcaret_test[,2]
  
obs_predTrain <- data.frame(pred = exp(rfcaret_train), obs = exp(dftrain1000$logTCP))
obs_predTrain
obs_pred_logTrain <- data.frame(pred = rfcaret_train, obs = dftrain1000$logTCP)

obs_predTest <- data.frame(pred = exp(rfcaret_test), obs = exp(dftest1000$logTCP))
obs_predlogTest <- data.frame(pred = rfcaret_test, obs = dftest1000$logTCP)

#this graph is Regular data (untransformed) 
obs_predPlotTrain <- ggplot(obs_predTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predPlotTrain

obs_predPlotTest <- ggplot(obs_predlogTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predPlotTest

#this graph is log of one or either depending on code
obs_predlogTrain <- ggplot(obs_predTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predlogTrain

obs_predlogTest <- ggplot(obs_predlogTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predlogTest

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
obs_predlogTestCart <- data.frame(pred = rpart_test, obs = dftest1000$logTCP)

#this graph is Regular data (untransformed) 
obs_predPlotTrainCart <- ggplot(obs_predTrainCart, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predPlotTrainCart

obs_predPlotTestCart <- ggplot(obs_predlogTestCart, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
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

#this graph is Regular data (untransformed) 
obs_predPlotTrainRF <- ggplot(obs_predTrainRF, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predPlotTrainRF

obs_predPlotTestRF <- ggplot(obs_predlogTestRF, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
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
obs_predlogTestRanger <- data.frame(pred = ranger_test, obs = dftest1000$logTCP)

#this graph is Regular data (untransformed) 
obs_predPlotTrainRanger <- ggplot(obs_predTrainRanger, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predPlotTrainRanger

obs_predPlotTestRanger <- ggplot(obs_predlogTestRanger, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
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
obs_predlogTestgbm <- data.frame(pred = gbm_test, obs = dftest1000$logTCP)

#this graph is Regular data (untransformed) 
obs_predPlotTraingbm <- ggplot(obs_predTraingbm, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predPlotTraingbm

obs_predPlotTestgbm <- ggplot(obs_predlogTestgbm, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
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
obs_predlogTestxgb <- data.frame(pred = xgb_test, obs = dftest1000$logTCP)

#this graph is Regular data (untransformed) 
obs_predPlotTrainxgb <- ggplot(obs_predTrainxgb, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predPlotTrainxgb

obs_predPlotTestxgb <- ggplot(obs_predlogTestxgb, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predPlotTestxgb

##################xboost################################################
xboost_train <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/pred_y_train.csv")
as.data.frame(xboost_train)
xboost_train <- xboost_train[,2]

xboost_test <- read.csv("C:/Users/bhauptman/Box/Ch2/R/data/pred_y_test.csv")
as.data.frame(xboost_test)
xboost_test <- xboost_test[,2]

obs_predTrainxboost <- data.frame(pred = exp(xboost_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTrainxboost <- data.frame(pred = xboost_train, obs = dftrain1000$logTCP)
                                  
obs_predTestxboost <- data.frame(pred = exp(xboost_test), obs = exp(dftest1000$logTCP))
obs_predlogTestxboost <- data.frame(pred = xboost_test, obs = dftest1000$logTCP)

#this graph is Regular data (untransformed) 
obs_predPlotTrainRangexboost <- ggplot(obs_predTrainxboost, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predPlotTrainxboost

obs_predPlotTestxboost <- ggplot(obs_predlogTestRanger, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)")
obs_predPlotTestxboost
