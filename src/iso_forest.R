library(solitude)
library(readr)
library(DALEX)
library(tidyverse)
library(caret)
library(foreign)
library(randomForest)
library(umap)
library(plotly)

outlier_data <- read.arff('data/Annthyroid_withoutdupl_norm_05_v10.arff')
index1 <- unlist(createDataPartition(as.factor(outlier_data$outlier), p = 0.1))

outlier_data_train <- outlier_data[-index1, ]
outlier_data_val <- outlier_data[index1, ]

outlier_data_train_X <- outlier_data_train %>% select(-c("id", "outlier"))
outlier_data_train_Y <- outlier_data_train$outlier

outlier_data_val_X <- outlier_data_val %>% select(-c("id", "outlier"))
outlier_data_val_Y <- outlier_data_val$outlier

# scramble the columns
uniform_sampling <- function(df, fraction = 1){
    
    dummy_data <- NULL
    
    for(i in 1:ncol(df)){
        
        temp <- sample(x = df[, i], 
                       size = round(fraction * nrow(df)), 
                       replace = T)
        
        dummy_data[[names(df)[i]]] <- temp
    }
    
    dummy_data_out <- do.call(data.frame, dummy_data)
    
    # Set targets
    df$target <- 1
    
    # Append data
    dummy_data_out <- merge(df, 
                            dummy_data_out,
                            by = names(df %>% select(-target)),
                            all = T)
    
    # Recode the target
    dummy_data_out$target[is.na(dummy_data_out$target)] <- 0
    
    return(dummy_data_out)
    
}

Synth_data <- uniform_sampling(outlier_data_train_X, fraction = 1)

table(Synth_data$target)

Synth_data$target <- ifelse(Synth_data$target == '1', 'real', 'synth')
Synth_data$target <- as.factor(Synth_data$target)

# UMAP plot of original data
config1 <- umap.defaults
config1$verbose <- T
UMAP0 <- umap(as.matrix(outlier_data_train_X), 
              config1, 
              method = 'naive')

# Plot
p <- plot_ly(data = data.frame(UMAP0$layout),
             x = ~X1,
             y = ~X2,
             color = ~outlier_data_train_Y,
             type = 'scatter',
             mode = 'markers',
             colors = 'Paired')

p



# Umap plot
UMAP <- umap(as.matrix(Synth_data %>% select(-target)), config1, method = 'naive')

# Plot
p <- plot_ly(data = data.frame(UMAP$layout),
             x = ~X1,
             y = ~X2,
             color = ~Synth_data$target,
             type = 'scatter',
             mode = 'markers',
             colors = 'Paired')

p



# Run randomforest on synthetic data
fitControl0 <- trainControl(method = "repeatedcv",
                            number = 5,
                            repeats = 1,
                            classProbs = TRUE,
                            summaryFunction = prSummary,
                            verboseIter = TRUE,
                            returnResamp = 'all',
                            savePredictions = 'all') 


rfFit0 <- caret::train(target ~., 
                       data = Synth_data, 
                       method = "rf", 
                       trControl = fitControl0,
                       tuneLength = 5,
                       metric = "AUC")


rf <- randomForest(target ~., 
                   data = Synth_data,
                   proximity = T)

# Get confusion matrix
pred <- predict(rfFit0, newdata = outlier_data_val_X)
levels(pred) <- c('no', 'yes')

confusionMatrix(pred, outlier_data_val_Y, mode = 'everything', positive = 'yes')