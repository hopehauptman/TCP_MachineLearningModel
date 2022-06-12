## Final raster projection of all models clean script. 4/5/2022

# load needed packages
library(sf)
library(sp)
library(rgeos)
library(rgdal)
library(PROJ)  
library(raster)
library(purrr)
library(caret)
library(xgboost)
library(tmaptools)
library(tidyverse)
library(caret)
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

# set working directory
setwd("C:/Users/bhauptman/Box/Ch2/R")

# create a new folder in Outputs with today's date, to store new Outputs
dir.create(paste0("Outputs/", Sys.Date()))

############Stacking rasters###################################################################

# import all raster tiff files from ArcGIS pro as a list. Help here: https://stackoverflow.com/questions/52746936/how-to-efficiently-import-multiple-raster-tif-files-into-r
rastlist <- list.files(path = "C:/Users/bhauptman/Box/Ch2/R/data/tif_shp", pattern='.tif$', all.files=TRUE, full.names=T)

#Import all tiff files into raster package using lapply
allrasters <- lapply(rastlist, raster)

# take at look at some raster info and see if extents are the same
summary(allrasters)
compareRaster(allrasters) #rasters are not the same so extents will have to be adjusted

# read shape file of StudyArea outline from ARC GISpro
StudyArea <- readOGR(dsn = "C:/Users/bhauptman/Box/Ch2/R/data/tif_shp", layer = "StudyArea")

# Check the crs (coordinate reference system) of the study area
crs(StudyArea)

# check that all rasters have the same crs, origins and extent
lapply(allrasters, crs)
compareCRS(allrasters[[1]], allrasters[[20]])

lapply(allrasters, origin)

extent(StudyArea)
lapply(allrasters, extent)

# assign the study area to an object
myextent <- extent(StudyArea)

#Perform transformation of study area to match raster layer - in this case raster 1 was selected arbitrarily
#crop_allrastersSA <- lapply(allrasters, FUN=crop, myextent)
resample_allrasters10 <- lapply(allrasters, FUN=resample, allrasters[[10]])

writeRaster(resample_allrasters10, filename="resample_allrasters10.tif", options="INTERLEAVE=BAND", overwrite=TRUE)

#check to see if that works
compareRaster(resample_allrasters10)
lapply(resample_allrasters10, extent)
lapply(resample_allrasters10, origin)

##Stacking the rasters##
Rstack <- stack(resample_allrasters10)
Rstack
plot(Rstack)

#########reorder names in raster stack according to order of names in models########################
#load a model from file
load("C:/Users/bhauptman/Box/Ch2/R/rf_tree1000.Rdata")

#Show order of names in model
names(Rstack) 
str(rf_tree1000)

#Isolate the file final model which contains the layer names form the model
s <- rf_tree1000$finalModel

#Now within the final model file there is another file with names - isolating that 
mod_name <- s$xNames
str(mod_name)

write(mod_name, "mod_name.txt")

#load(name, "name.txt")

# Changing the names in the raster stack to match those in the model
Rstack <- Rstack[[mod_name]]

# write raster stack to disk
writeRaster(Rstack, filename="Rstack.tif", overwrite=TRUE)

#load raster to eliminate the need for starting from this line: 
RstackRenamed <- terra::rast("Rstack.tif")
names(RstackRenamed) <- mod_name
names(RstackRenamed)

# write raster stack to disk
r <- terra::writeRaster(RstackRenamed, filename="RstackRenamed.tif", overwrite=TRUE)

###### Start here  to predict model directly onto map ##########################################################################

r <- brick("RstackRenamed.tif")
names(r) #just to check that order is right

# read shape file of StudyArea outline from ARC GISpro
StudyArea <- readOGR(dsn = "C:/Users/bhauptman/Box/Ch2/R/data/tif_shp", layer = "StudyArea")

#################Predicting onto raster stack surface using different models########################################
library(raster)

## 1. rf model ##

#load a model from file
load("C:/Users/bhauptman/Box/Ch2/R/rf_tree1000.Rdata")

#test: predict directly onto raster stack using raster package
r_pred <- raster::predict(model = rf_tree1000, object = r)

# crop to the study area
r_pred_mask <- raster::mask(r_pred, StudyArea)

# back transform raster values to remove log
rf_trans = 10^(r_pred_mask)
plot(rf_trans)

# save the prediction map
writeRaster(rf_trans, "rf_raster.tif", overwrite=TRUE)

# save the prediction map
writeRaster(rf_trans, paste0("Outputs/", Sys.Date(), "rf_raster.tif", overwrite=TRUE))

#to call from file
#mystack = stack("multilayer.tif")
######################
## 2. rfcaret model ## This is the "best model according to R2 values

rfCaret <- readRDS("FinalModels/rfcaret_1000.Rds")

#test: predict directly onto raster stack using raster package
r_pred_CARET <- raster::predict(model = rfCaret, object = r)

# crop to the study area
r_pred_CARET <- mask(r_pred_CARET, StudyArea)

# back transform raster values to remove log
rfcaret_trans = 10^(r_pred_CARET)
plot(rfcaret_trans)

# save the prediction map
writeRaster(rfcaret_trans, "rfcaret.tif", overwrite=TRUE)

# save the prediction map
writeRaster(rfcaret_trans, paste0("Outputs/", Sys.Date(), "rfCaret.tif", overwrite=TRUE))

######################
## 3. xgb_train_1 model ##

#load model from file
xgb <- readRDS("FinalModels/xgb_train_1.Rds")

#test: predict directly onto raster stack using raster package
memory.limit(13200000000)
r_pred_xgb <- raster::predict(model = xgb, object = r)

# crop to the study area
r_pred_xgb <- mask(r_pred_xgb, StudyArea)

# back transform raster values to remove log
xgb_trans = 10^(r_pred_xgb)
plot(xgb_trans)

# save the prediction map
writeRaster(xgb_trans, "xgb.tif", overwrite=TRUE)

# save the prediction map
writeRaster(xgb_trans, paste0("Outputs/", Sys.Date(), "xgb.tif", overwrite=TRUE))

######################
## 4. cart model ##

#load model from file
cart <- readRDS("FinalModels/rpart_tree1000.Rds")
str(cart)

#test: predict directly onto raster stack using raster package
r_cart <- raster::predict(model = cart, object = r)

# crop to the study area
r_pred_cart <- mask(r_cart, StudyArea)

# back transform raster values to remove log
cart_trans = 10^(r_pred_cart)
plot(cart_trans)

# save the prediction map
writeRaster(cart_trans, "cart_trans.tif", overwrite=TRUE)

# save the prediction map
writeRaster(cart_trans, paste0("Outputs/", Sys.Date(), "cart_trans.tif", overwrite=TRUE))

######################
## 5. ranger model ##

#load model from file
ranger <- readRDS("FinalModels/ranger_tree1000.Rds")

#test: predict directly onto raster stack using raster package
r_ranger <- raster::predict(model = ranger, object = r)

# crop to the study area
r_pred_ranger <- mask(r_ranger, StudyArea)

# back transform raster values to remove log
ranger_trans = 10^(r_pred_ranger)
plot(ranger_trans)

# save the prediction map
writeRaster(ranger_trans, "ranger_trans.tif", overwrite=TRUE)

# save the prediction map
writeRaster(ranger_trans, paste0("Outputs/", Sys.Date(), "ranger.tif", overwrite=TRUE))

######################
## 6. gbm model ##

#load model from file
gbm <- readRDS("FinalModels/gbm_tree1000.Rds")

#test: predict directly onto raster stack using raster package
r_gbm <- raster::predict(model = gbm, object = r)

# crop to the study area
r_pred_gbm <- mask(r_gbm, StudyArea)

# back transform raster values to remove log
gbm_trans = 10^(r_pred_gbm)
plot(gbm_trans)

# save the prediction map
writeRaster(gbm_trans, "gbm_trans.tif", overwrite=TRUE)

# save the prediction map
writeRaster(gbm_trans, paste0("Outputs/", Sys.Date(), "gbm.tif", overwrite=TRUE))

######################
## 7. xboost model ##
xgboost <- readRDS("FinalModels/xgbc.Rds")

#test: predict directly onto raster stack using raster package
r_xgboost <- raster::predict(model = xgboost, object = r)

# crop to the study area
r_pred_xgboost <- mask(r_xgboost, StudyArea)

# back transform raster values to remove log
xgboost_trans = 10^(r_pred_xgboost)
plot(xgboost_trans)

# save the prediction map
writeRaster(xgboost_trans, "xgboost_trans.tif", overwrite=TRUE)

# save the prediction map
writeRaster(xgboost_trans, paste0("Outputs/", Sys.Date(), "xgboost_trans.tif", overwrite=TRUE))

###############################################################################################################
#reading in the models from file
cart <- readRDS("FinalModels/rpart_tree1000.Rds")
randomForest <- readRDS("FinalModels/rF1000.Rds")
rfCaret <- readRDS("FinalModels/rfcaret_1000.Rds")
ranger <- readRDS("FinalModels/ranger_tree1000.Rds")
gbm <- readRDS("FinalModels/gbm_tree1000.Rds")
xgb <- readRDS("FinalModels/xgb_train_1.Rds")
xgboost <- readRDS("FinalModels/xgbc.Rds")

###############################################################################################################
#PROBABLY CAN DELETE THIS LATER#
##### alternative to match extents and stack rasters###########################################################
# ##Creating a function to make a polygon of each raster's extent and then applying the function##
# fxn<-function(ras){
#   bb<-bbox(ras)
#   bbpoly<-bb_poly(bb)
#   st_crs(bbpoly)<-crs(ras)
#   return(as_Spatial(bbpoly))
# }
# 
# # Apply the function to the rasters
# ext<-lapply(allrasters, FUN=fxn)
# 
# ##Aggregating and dissolving all extents to get the full extent of all rasters##
# full.ext<-aggregate(do.call(bind, ext), dissolve=TRUE)
# 
# ##Creating a blank raster with the full extent, the desired final projection, and the desired resolution##
# blank<-raster(ext=extent(full.ext), nrow=allrasters[[10]]@nrows, ncol=allrasters[[10]]@ncols, crs=allrasters[[10]]@crs)
# 
# ##Resampling all rasters in the list to the desired extent and resolution##
# memory.limit(13200000000)
# rastostack <-lapply(allrasters, FUN = resample, blank)
# 
# ##Stacking the rasters##
# Ras <- stack(rastostack)
# 
# #View the stack
# plot(Ras)
###########################################################
