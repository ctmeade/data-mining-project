#Libraries
library(caret)
library(mice)
library(dplyr)

#Import Data
headers <- c("timesPregnant", "plasmaGlucose", "diastolicPressure", "tricepThickness", "serumInsulin", "bmi", "pedigreeFunction", "age", "diabetes")

data <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"),
                 header = FALSE,
                 col.names = headers)

#Make the coded classification column more readable
data$diabetes <- as.factor(ifelse(data$diabetes == 0, "notDiabetic", "Diabetic"))

#####EXPLORATION

###Dealing with missing values

#Data set is not encoded with NA's
#But there are 0's where biologocially impossible
#This must be how NAs were encoded
#Varibles with missing data include plasamaGlucose, diastolicPressure, tricepThickness, serumInsulin, bmi

#Replace missing values with NA

for (i in 2:6){
  for (n in 1:nrow(data)){
    if (data[n, i] == 0){
      data[n, i] <- NA
    }
  }
}

#How many observations have missing values?

md.pattern(data)

#Only 392 complete observations
#That means almost half of ovservations missing
#Can't drop that many observations
#Must impute missing values

#Multivariate Imputation by Chained Equations 

tempdata <- mice(data, m = 1, method = 'pmm', seed = 131)
data <- complete(tempdata, 1)


densityplot(tempdata)

#Look at distrubution of imputed data (magenta) vs original (blue)
#Similar, so we're ok

###Feature Selection

#Need to make sure variablbes aren't correlated
correlationMatrix <- cor(data[,1:8])

#Cutoff at .7
highCorrelation <- findCorrelation(correlationMatrix, cutoff=0.7)
print(highCorrelation)
#Dont need to eliminate variable

#Standardization
#Center and Scale data
data[, 1:8] <- scale(data[, 1:8], center = TRUE, scale = TRUE)

#Construct 10fold CV method
tenFoldCV <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 10,
                          classProbs=TRUE,
                          summaryFunction=twoClassSummary)



#Partition data in training and test sets

train <- sample.int(0.7*nrow(data), )



x.data <- data[,1:8]
y.data <- as.factor(data[,9])

#Models

rf <- train(x.data,
            y.data,
            method = "rf",
            metric = "ROC",
            trControl = tenFoldCV)

rf


svm <- caret::train(x.data,
                    y.data,
                    method = "svmRadial",
                    metric = "ROC",
                    trControl = tenFoldCV,
                    tuneLength = 10)

svm

svmExpandedGrid <- expand.grid(sigma = c(.07, .09, .11, .13, .15, .17, .19),
                               C = c(.1, .15, .2, .25, .3, .35, .4))

svm2 <- caret::train(x.data,
                     y.data,
                     method = "svmRadial",
                     metric = "ROC",
                     trControl = tenFoldCV,
                     tuneGrid = svmExpandedGrid)

svm2

mlpnn <- caret::train(x.data,
                      y.data,
                      method = "mlpML",
                      metric= "ROC",
                      trControl=tenFoldCV)

mlpnn

