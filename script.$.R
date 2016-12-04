#Libraries
library(caret)
library(mice)

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

#Multivariate Imputation by Predictive mean matching 

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



###Partition data in training and test sets

sampleSize <- floor(.7 * nrow(data))

set.seed(131)
trainIndices <- sample(seq_len(nrow(data)), size = sampleSize)

x.train <- data[trainIndices, 1:8]
y.train <- data[trainIndices, 9]

x.test <- data[-trainIndices, 1:8]
y.test <- data[-trainIndices, 9]

#Models

#Random Forest
set.seed(130)
rf <- train(x.train,
            y.train,
            method = "rf",
            metric = "ROC",
            trControl = tenFoldCV)

rf


#Support vector machine with radial bias kernel function
set.seed(131)
svm <- caret::train(x.train,
                    y.train,
                    method = "svmRadial",
                    metric = "ROC",
                    trControl = tenFoldCV,
                    tuneLength = 6)

svm

#Tune SVM models by expanding search grid around optimal sigma and C returned by the previous model

svmExpandedGrid <- expand.grid(sigma = c(.07, .1, .13),
                               C = c(.4, .45, .5, .55, .6))

set.seed(131)
svm2 <- caret::train(x.train,
                     y.train,
                     method = "svmRadial",
                     metric = "ROC",
                     trControl = tenFoldCV,
                     tuneGrid = svmExpandedGrid)

svm2

#Multi-layer perceptron neural network
set.seed(131)
mlpnn <- caret::train(x.data,
                      y.data,
                      method = "mlpML",
                      metric= "ROC",
                      trControl=tenFoldCV)

mlpnn

#RF test accuracy

rf.predict <- predict(rf$finalModel, x.test)
rf.test.accuracy <- mean(rf.predict == y.test)
rf.test.accuracy

#SVM Test accuracy
svm.predict <- predict(svm2$finalModel, x.test)
svm.test.accuracy <- mean(svm.predict == y.test)
svm.test.accuracy

#MLPNN Test Accuracy
mlpnn.predict <- predict(mlpnn$finalModel, x.test)
mlpnn.predict <- as.data.frame(mlpnn.predict)
mlpnn.predict$prediction <- ifelse(mlpnn.predict$V1 >= .5, "Diabetic", "notDiabetic")
mlpnn.test.accuracy <- mean(mlpnn.predict$prediction == y.test)
mlpnn.test.accuracy
