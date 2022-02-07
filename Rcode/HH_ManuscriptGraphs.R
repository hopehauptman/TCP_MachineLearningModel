###############Graphs and figures for Ch 2 TCP predictive model##################################

setwd("C:/Users/Hope Hauptman/Box/Ch2/Manuscipt and figures")

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

my_data <- read_excel("C:/Users/Hope Hauptman/Box/Ch2/R/data/Caret_VarImpPlot.xlsx")

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

LU_Data <- read.csv("C:/Users/Hope Hauptman/Box/Ch2/R/data/LandUseEV1000.csv")
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
load(file = "C:/Users/Hope Hauptman/Box/Ch2/R/data/rf_caret1000.Rdata")
read.csv(file = "C:/Users/Hope Hauptman/Box/Ch2/R/data/dftrain1000.csv")
read.csv(file = "C:/Users/Hope Hauptman/Box/Ch2/R/data/dftrain1000.csv")
read.csv(file = "C:/Users/Hope Hauptman/Box/Ch2/R/data/rfcaret_train1000.csv")
read.csv(file = "C:/Users/Hope Hauptman/Box/Ch2/R/data/rfcaret_test1000.csv")

obs_predTrain <- data.frame(pred = exp(rfcaret_train1000), obs = exp(dftrain1000$logTCP))
obs_pred_logTrain <- data.frame(pred = rfcaret_train1000), obs = dftrain1000$logTCP)

obs_predTest <- data.frame(pred = exp(rfcaret_test1000), obs = exp(dftest1000$logTCP))
obs_predlogTest <- data.frame(pred = rfcaret_test1000), obs = dftest1000$logTCP)

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

























