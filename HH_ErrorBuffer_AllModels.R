# Read in df 
library(readxl)
library(ggplot2)
library(tidyr)

setwd("C:/Users/Hope Hauptman/Box/Ch2/R")
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
my_labeller <- as_labeller(c( MAE = 'MAE', RMSE = 'RMSE', `R^2` = 'R^2'),default = label_parsed))
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
