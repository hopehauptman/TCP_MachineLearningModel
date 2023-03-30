#https://gis.stackexchange.com/questions/326910/generating-prediction-raster-from-random-forest-model-using-r

#Install libraries
library(sf)
library(sp)
library(rgdal)
library(raster)
library(purrr)
library(rgeos)
library(tmaptools)
library(caret)
library(xgboost)
library(data.table)
library(terra)

########Combine rasters in study area into the same stack##################
setwd("C:/Users/bhauptman/Desktop/RcodeTCPmodel")

#first import all files in a single folder as a list Help here: https://stackoverflow.com/questions/52746936/how-to-efficiently-import-multiple-raster-tif-files-into-r
rastlist <- list.files(path = "data/tif_shp", pattern='.tif$', all.files=TRUE, full.names=T)

#import all tiff files into raster using lapply, to call single raster element: allrasters[[1]]
allrasters <- lapply(rastlist, FUN=raster)
summary(allrasters)

# #read shape file of StudyArea outline from ARC GISpro
StudyArea <- readOGR(dsn = "data/tif_shp", layer = "StudyArea")
# 
# #Perform transformation of study area to match raster layer - in this case raster 1 was selected arbitrarily
StudyArea <- spTransform(StudyArea, crs(allrasters[[1]]))

#match the extent of all rasters to an arbitrary raster in stack
extent(allrasters[[1]])

#apply crop function to all items in list -  or maybe lapply(allrasters, FUN=mask, StudyArea)
lapply(allrasters, FUN=crop, allrasters[[3]])

#Check the crs (coordinate reference system)
crs(allrasters[[1]])

#need to change one raster crs to be the same as another
#allrasters[[2]] <- projectRaster(allrasters[[2]], crs=crs(allrasters[[3]]))
compareCRS(allrasters[[1]], allrasters[[14]])

##Creating a function to make a polygon of each raster's extent##
fxn<-function(ras){
  bb<-bbox(ras)
  bbpoly<-bb_poly(bb)
  st_crs(bbpoly)<-crs(ras)
  return(as_Spatial(bbpoly))
}

ext<-lapply(allrasters, FUN=fxn)

##Aggregating and dissolving all extents to get the full extent of all rasters##
full.ext<-aggregate(do.call(bind, ext), dissolve=TRUE)

##Creating a blank raster with the full extent, the desired final projection, and the desired resolution##
blank<-raster(ext=extent(full.ext), nrow=allrasters[[1]]@nrows, ncol=allrasters[[1]]@ncols, crs=allrasters[[1]]@crs)

##Resampling all rasters in the list to the desired extent and resolution##
rastostack <-lapply(allrasters, FUN = resample, blank)

##Stacking the rasters##
Ras <- raster::stack(rastostack)
Ras

# #saving the new stack to a file: writeRaster(yourStackObject, filename="multilayer.tif", options="INTERLEAVE=BAND", overwrite=TRUE)
writeRaster(Ras, filename="Ras.tif",  options="INTERLEAVE=BAND", overwrite=TRUE)

#load model from file
#rfcaret_1000_old <- readRDS("C:/Users/bhauptman/Desktop/rfcaret_1000.Rds")

Ras <- raster("C:/Users/bhauptman/Desktop/RcodeTCPmodel/Ras1.tif")

#rename layers in raster stack to match the names in the models
name <- c("Age", "Drainage", "HSG", "LU2005", "LU1990", "LU1975", "LU1960", "LU1945", "NO3_180", "NO3_400", "DO5", "DO2", "Precip2005", "Precip1990", "Precip1975", "Precip1960", "Irrig2005", "Irrig1990", "Irrig1975", "Irrig1960")

#apply those names to the raster
Ras1 <- Ras[[name]]

#check layer name and order
Ras1@layers

#####Predict models onto raster stacks and bias correct predictions using EDM method 
#RF#################################################################################
r <- predict(Ras1, rfcaret_1000)
ras <- Ras[[1]]
values(ras) <- exp(r)
plot(exp(r))
rval <- values(r)

#saving the new stack to a file: writeRaster(yourStackObject, filename="multilayer.tif", options="INTERLEAVE=BAND", overwrite=TRUE)
writeRaster(ras, filename="ras.tif",  options="INTERLEAVE=BAND", overwrite=TRUE)

#now apply the bias correct
# bias correct the vector based on corrected training data (map values treated same as hold out for bias correction)
map_qq <- approx(rfcaret_train1000, train_pred_correctedRF, rval, ties = mean)$y

# Second step: use QQ_slope to adjust the extreme values that led to nulls (if any)
map_qq <- ifelse(rval < min(rfcaret_train1000), min(train_pred_correctedRF) - ltf_slope *(min(rfcaret_train1000) - rval), map_qq)
map_qq <- ifelse(rval > max(rfcaret_train1000), max(train_pred_correctedRF) + ltf_slope *(rval - max(rfcaret_train1000)), map_qq)

# put the corrected predictions back into raster format
R <- Ras[[1]] # get any raster from the dataset to use as a template for the predictions 

values(R) <- exp(map_qq) # assign the predicted values to it

R<- mask(R,sum(Ras)) # sum will make a raster with NA where any raster in the stack has NA

plot(R) # needs to be masked to avoid filled in areas where there were NA

#saving the new stack to a file: writeRaster(yourStackObject, filename="multilayer.tif", options="INTERLEAVE=BAND", overwrite=TRUE)
writeRaster(R, filename="R.tif",  options="INTERLEAVE=BAND", overwrite=TRUE)

#CART###############################################################################
c <- predict(Ras1, rpart_tree1000)
plot(exp(c))
cval <- values(c)

#now apply the bias correct
# bias correct the vector based on corrected training data (map values treated same as hold out for bias correction)
map_qqc <- approx(rpart_train, train_pred_correctedCART, cval, ties = mean)$y

# Second step: use QQ_slope to adjust the extreme values that led to nulls (if any)
map_qqc <- ifelse(cval < min(rpart_test), min(train_pred_correctedCART) - ltf_slopeCART *(min(rpart_train) - ccal), map_qqc)
map_qqc <- ifelse(cval > max(rpart_test), max(train_pred_correctedCART) + ltf_slopeCART *(cval - max(rpart_train)), map_qqc)

# put the corrected predictions back into raster format
C <- Ras[[1]] #get any raster from the dataset to use as a template for the predictions 

values(C) <- exp(map_qqc) # assign the predicted values to it

C <- mask(C,sum(Ras)) # sum will make a raster with NA where any raster in the stack has NA

plot(C) # needs to be masked to avoid filled in areas where there were NA
writeRaster(C, filename="C.tif",  options="INTERLEAVE=BAND", overwrite=TRUE)

#BRT#################################################################################
g <- predict(Ras1, gbmFit)
plot(exp(g))
gval <- values(g)

#now apply the bias correct
# bias correct the vector based on corrected training data map values treated same as hold out for bias correction
map_qqg <- approx(gbm_train, train_pred_correctedBRT, gval, ties = mean)$y

#Second step: use QQ_slope to adjust the extreme values that led to nulls (if any)
map_qqg <- ifelse(gval < min(gbm_train), min(train_pred_correctedBRT) - ltf_slopeBRT *(min(gbm_train) - gval), map_qqg)
map_qqg <- ifelse(gval > max(gbm_train), max(train_pred_correctedBRT) + ltf_slopeBRT *(gval - max(gbm_train)), map_qqg)

# put the corrected predictions back into raster format
G <- Ras[[1]] # get any raster from the dataset to use as a template for the predictions 

values(G) <- exp(map_qqg) # assign the predicted values to it

G <- mask(G,sum(Ras)) # sum will make a raster with NA where any raster in the stack has NA

plot(G) # needs to be masked to avoid filled in areas where there were NA

writeRaster(G, filename="G.tif",  options="INTERLEAVE=BAND", overwrite=TRUE)

typeof(G)


