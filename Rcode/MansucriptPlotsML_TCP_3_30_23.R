###############Graphs and figures for Ch 2 TCP predictive model##################################

setwd("C:/Users/bhauptman/Desktop/RcodeTCPmodel")

#load libraries
library(data.table)
library(ggplot2)

#########################Plot for Mass balance##########################################
MBdata <- read.csv(file = "data/MassBalanceGraph.csv") 

colors <- c("0.17% TCP by mass"= "steelblue", "7% TCP by mass" = "darkred")

MBplot <- ggplot(MBdata , aes(x = year)) + 
                   geom_line(aes(y=TCP.17, color="0.17% TCP by mass"), size=1.2) +
                   geom_line(aes(y=TCP7, color="7% TCP by mass"), size=1.2) +
                   labs(x="Year", y="log10 TCP Mass in kg", color="Legend") +
  scale_y_continuous(breaks=seq(0,8,1))+
  theme_bw()

MBplot

#########################Variable Importance plot using caret RF model######################
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

#go into imp and manually chage to top 20 values

#Top 25 only ?? not sure how 
my_data <- read_excel("imp20maj.xlsx")
as.data.frame(my_data)

plot <- ggplot(data=my_data, aes(x = reorder(variable, Overall), y = Overall)) + geom_bar(stat='identity')+
  coord_flip() + xlab("Variable Name") + ylab("Importance") + theme_bw() 
plot

#####################################Land Use Median TCP graph#############################
library(ggplot2)
library(data.table)
library(dplyr)

LU_Data <- read.csv("data/LandUseEV1000.csv")
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


##############obs versus predicted graphs#######################################################################################
## Note: must run models and bias predictions before making the plots below

#read in dftrain and dftest data sets 
dftrain1000 <- read.csv(file = "data/dftrain1000.csv")
dftest1000 <- read.csv(file = "data/dftest1000.csv")

######################################################################
#                 Random Forest                                      #
######################################################################
#NOT BIASED CORRECTED#######################
#read in data, put in data frame and select relevant column
rfcaret_train <- read.csv(file = "data/rfcaret_train1000.csv")
ncol(rfcaret_train)
as.data.frame(rfcaret_train)
rfcaret_train <- rfcaret_train[,2]
rfcaret_train

rfcaret_test <- read.csv(file = "data/rfcaret_test1000.csv")
ncol(rfcaret_test)
as.data.frame(rfcaret_test)
rfcaret_test <- rfcaret_test[,2]
rfcaret_test
  
obs_predTrain <- data.frame(pred = exp(rfcaret_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTrain <- data.frame(pred = rfcaret_train, obs = dftrain1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrain["TrainTest"]='train'
obs_pred_logTrain["model"]='RF'

obs_predTest <- data.frame(pred = exp(rfcaret_test), obs = exp(dftest1000$logTCP))
obs_pred_logTest <- data.frame(pred = rfcaret_test, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTest["TrainTest"]='test'
obs_pred_logTest["model"]='RF'

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
obs_predTrain

obs_predTest <- ggplot(obs_predTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: Caret Random Forest    Data: Testing")
obs_predTest

# BIASED CORRECTED#######################
obs_predTrain <- data.frame(pred = exp(train_pred_correctedRF), obs = exp(dftrain1000$logTCP))
obs_pred_logTrain <- data.frame(pred = train_pred_correctedRF, obs = dftrain1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrain["TrainTest"]='train'
obs_pred_logTrain["model"]='RF'

obs_predTest <- data.frame(pred = exp(holdout_pred_correctedRF), obs = exp(dftest1000$logTCP))
obs_pred_logTest <- data.frame(pred = holdout_pred_correctedRF, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTest["TrainTest"]='test'
obs_pred_logTest["model"]='RF'

#combine into data frame for graphing 
df_rfCaret <- rbind(obs_pred_logTrain, obs_pred_logTest)

####this graph is log of the data for both x and y axis for TRAINING data
obs_predPlotTrainLog <- ggplot(obs_pred_logTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: Random Forest    Data: Training")
obs_predPlotTrainLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainLog.png")

#this graph is log of the data for both x and y axis for TESTING data
obs_predPlotTestLog <- ggplot(obs_pred_logTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log 10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: Random Forest    Data: Testing")
obs_predPlotTestLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestLog.png")

####this graph is untransformed data for both x and y axis for Training data
obs_predTrain <- ggplot(obs_predTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: Random Forest    Data: Training")
obs_predTrain

#this graph is untransformed data for both x and y axis for Testing data
obs_predTest <- ggplot(obs_predTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: Caret Random Forest    Data: Testing")
obs_predTest

######################################################################
#                 CART (rart2)                                       # 
######################################################################
#NOT BIASED CORRECTED


#construct additional obs and predicted graphs for other models
rpart_train <- read.csv("data/rpart_train1000.csv")
as.data.frame(rpart_train)
rpart_train <- rpart_train[,2]

rpart_test <- read.csv("data/rpart_test1000.csv")
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

#NOT BIASED CORRECTED#######################
#read in data, put in data frame and select relevant column
obs_predTrain <- data.frame(pred = exp(rpart_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTrain <- data.frame(pred = rpart_train, obs = dftrain1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrain["TrainTest"]='train'
obs_pred_logTrain["model"]='Random Forest'

obs_predTest <- data.frame(pred = exp(rpart_test), obs = exp(dftest1000$logTCP))
obs_pred_logTest <- data.frame(pred = rpart_test, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTest["TrainTest"]='test'
obs_pred_logTest["model"]='CART'

#combine into data frame for graphing 
df_CART <- rbind(obs_pred_logTrain, obs_pred_logTest)

####this graph is log of the data for both x and y axis for TRAINING data
obs_predPlotTrainLog <- ggplot(obs_pred_logTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: CART    Data: Training")
obs_predPlotTrainLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainLog.png")

#this graph is log of the data for both x and y axis for TESTING data
obs_predPlotTestLog <- ggplot(obs_pred_logTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log 10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: CART    Data: Testing")
obs_predPlotTestLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestLog.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predTrain <- ggplot(obs_predTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: Caret Random Forest    Data: Training")
obs_predTrain

obs_predTest <- ggplot(obs_predTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: Caret Random Forest    Data: Testing")
obs_predTest

#BIASED CORRECTED#######################
obs_predTrain <- data.frame(pred = exp(train_pred_correctedCART), obs = exp(dftrain1000$logTCP))
obs_pred_logTrain <- data.frame(pred = train_pred_correctedCART, obs = dftrain1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrain["TrainTest"]='train'
obs_pred_logTrain["model"]='CART'

obs_predTest <- data.frame(pred = exp(holdout_pred_correctedCART), obs = exp(dftest1000$logTCP))
obs_pred_logTest <- data.frame(pred = holdout_pred_correctedCART, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTest["TrainTest"]='test'
obs_pred_logTest["model"]='CART'

#combine into data frame for graphing 
df_CART <- rbind(obs_pred_logTrain, obs_pred_logTest)

####this graph is log of the data for both x and y axis for TRAINING data
obs_predPlotTrainLog <- ggplot(obs_pred_logTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: CART    Data: Training")
obs_predPlotTrainLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainLog.png")

#this graph is log of the data for both x and y axis for TESTING data
obs_predPlotTestLog <- ggplot(obs_pred_logTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log 10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: CART    Data: Testing")
obs_predPlotTestLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestLog.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predTrain <- ggplot(obs_predTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: CART    Data: Training")
obs_predTrain

obs_predTest <- ggplot(obs_predTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: CART    Data: Testing")
obs_predTest


######################################################################
#                  BRT (gbm)                                         #
######################################################################
#NOT BIASED CORRECTED

#NOT BIASED CORRECTED
gbm_train <- read.csv("data/gbm_train1000.csv")
as.data.frame(gbm_train)
gbm_train <- gbm_train[,2]

gbm_test <- read.csv("data/gbm_test1000.csv")
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

#NOT BIASED CORRECTED#######################
#read in data, put in data frame and select relevant column

obs_predTrain <- data.frame(pred = exp(gbm_train), obs = exp(dftrain1000$logTCP))
obs_pred_logTrain <- data.frame(pred = gbm_train, obs = dftrain1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrain["TrainTest"]='train'
obs_pred_logTrain["model"]='BRT'

obs_predTest <- data.frame(pred = exp(gbm_test), obs = exp(dftest1000$logTCP))
obs_pred_logTest <- data.frame(pred = gbm_test, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTest["TrainTest"]='test'
obs_pred_logTest["model"]='BRT'

#combine into data frame for graphing 
df_BRT <- rbind(obs_pred_logTrain, obs_pred_logTest)

####this graph is log of the data for both x and y axis for TRAINING data
obs_predPlotTrainLog <- ggplot(obs_pred_logTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: BRT    Data: Training")
obs_predPlotTrainLog
#ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainLog.png")

#this graph is log of the data for both x and y axis for TESTING data
obs_predPlotTestLog <- ggplot(obs_pred_logTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log 10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: BRT    Data: Testing")
obs_predPlotTestLog
#ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestLog.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predTrain <- ggplot(obs_predTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: BRT    Data: Training")
obs_predTrain

obs_predTest <- ggplot(obs_predTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: BRT    Data: Testing")
obs_predTest


#BIASED CORRECTED#######################
obs_predTrain <- data.frame(pred = exp(train_pred_correctedBRT), obs = exp(dftrain1000$logTCP))
obs_pred_logTrain <- data.frame(pred = train_pred_correctedBRT, obs = dftrain1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTrain["TrainTest"]='train'
obs_pred_logTrain["model"]='BRT'

obs_predTest <- data.frame(pred = exp(holdout_pred_correctedBRT), obs = exp(dftest1000$logTCP))
obs_pred_logTest <- data.frame(pred = holdout_pred_correctedBRT, obs = dftest1000$logTCP)

#adding columns which will be used as factors
obs_pred_logTest["TrainTest"]='test'
obs_pred_logTest["model"]='BRT'

#combine into data frame for graphing 
df_rfCaret <- rbind(obs_pred_logTrain, obs_pred_logTest)

####this graph is log of the data for both x and y axis for TRAINING data
obs_predPlotTrainLog <- ggplot(obs_pred_logTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: BRT    Data: Training")
obs_predPlotTrainLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTrainLog.png")

#this graph is log of the data for both x and y axis for TESTING data
obs_predPlotTestLog <- ggplot(obs_pred_logTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted log 10 TCP concentration (ppb)") + ylab("Observed log10 TCP concentration (ppb)") + ggtitle("Model: BRT    Data: Testing")
obs_predPlotTestLog
ggsave("C:/Users/bhauptman/Box/Ch2/R/plots/obs_predPlotTestLog.png")

####this graph is untransformed data for both x and y axis for TRAINING data
obs_predTrain <- ggplot(obs_predTrain, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: BRT    Data: Training")
obs_predTrain

obs_predTest <- ggplot(obs_predTest, aes(x=pred, y=obs)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+ theme_bw() +
  xlab("Predicted TCP concentration (ppb)") + ylab("Observed TCP concentration (ppb)") + ggtitle("Model: BRT    Data: Testing")
obs_predTest


#####################################################################
####COMBINING train test and biased unbiased together########
df_all <- rbind(df_rfCaret, df_BRT)
as.data.frame(df_all)

df_all$model <- as.factor(df_all$model)
df_all$TrainTest <- as.factor(df_all$TrainTest)

#subset train and test data
subTrain <- subset(df_all, TrainTest == "train")

subTest <- subset(df_all, TrainTest == 'test')

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


####COMBINING GRAPHS USING rbind and FACET GRID########
df_all <- rbind(df_CART, df_rfCaret, df_xboost)
as.data.frame(df_all)

df_all$model <- as.factor(df_all$model)
df_all$TrainTest <- as.factor(df_all$TrainTest)

#subset train and test data
subTrain <- subset(df_all, TrainTest == "train")

subTest <- subset(df_all, TrainTest == 'test')

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

###ERROR for 3 models###########################################################################################################

# Read in df 
library(readxl)
library(ggplot2)
library(tidyr)

error3 <- read_excel('data/ggplot_error3.xlsx')
error3 <- transform(
  error3,
  Metric =as.factor(Metric),
  Model =as.factor(Model),
  Name =as.factor(Name),
  Buffer=as.factor(Buffer)
)
error3

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
  labs(y = "Value", x = 'Buffer radius (m)') +
  facet_wrap(~ Metric, nrow = 3, scales = "free_y", labeller=my_labeller)

#default level order
levels(error3$Model)
#change the order of labels
error3$Model <- factor(error3$Model, levels = c("RF", "BRT", "CART"))

#change the label names


#add superscript for R^2
error3 %>% mutate(across(Metric, factor, levels=c("MAE" , "RMSE", "R2"), labels = c('MAE', 'RMSE', 'R2'))) %>%
  ggplot(aes(x=Buffer, y=Value, group=Model))+
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model)) +
  labs(y = "Value", x = 'Buffer radius (m)') +
  facet_grid(~ Metric, labeller=my_labeller) +
  labs(color='Model name') 

#change order of panels: https://stackoverflow.com/questions/15116081/controlling-order-of-facet-grid-facet-wrap-in-ggplot2