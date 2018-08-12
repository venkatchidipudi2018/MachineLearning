#importing dataset
dataset = read.csv("Data.csv")
#View(dataset)


#Taking care of missing values
dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                          dataset$Salary)



#Encoding categorial data
dataset$City = factor(dataset$City, levels = c('Bangalore', 'Hyderabad', 'Chennai'),
                      labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased, levels = c('Yes', 'No'),
                      labels = c(1, 0))


#Splitting the dataset into training dataset and test dataset
#install.packages('caTools') --> already installed
library(caTools)
set.seed(123)
split = dataset.split(dataset$Purchased, SplitRatio = 0.8)
training_dataset = subset(dataset, split == TRUE)
test_dataset = subset(dataset, split == FALSE)


#Feature Scaling
training_dataset[,2:3] = scale(training_dataset[,2:3])
test_dataset[,2:3] = scale(test_dataset[,2:3])