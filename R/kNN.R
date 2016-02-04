# ----------------------------------------------------------------------
# Simple kNN classifier
# ----------------------------------------------------------------------
#' kNN classifier 
#' 
#' Classify the input with a k nearest neighbors classifier.
#'
#' @param data A data frame or a matrix where rows are observations and columns are features. If \code{type} is "train" this is training dataset, and if it is "predict" it is test dataset.
#' @param trueClasses A vector with labels for each row in \code{data} if \code{type} is "train", and with labels for each row in \code{memory} if \code{type} is "predict".
#' @param memory A data frame or a matrix where rows are observations and columns are features. If \code{type} is "train" this argument is not needed, and if it is "predict" it is a training dataset.
#' @param k Number of neighbors that the classifier should use. It has to be an odd number.
#' @param p Distance metric the classifier should use, the value can be either 1, 2 or Inf. 
#' @param type Whether the goal is to train the classifier or predict classes of new observations based on past ones. The value can be either "train" or "predict".
#' @return A list with following elements: predictedClasses, prob, accuracy and errorCount.
#' @export
#' @import assertthat 
#' @examples
#' # create artificial dataset
#' inputsTest   <- matrix(rnorm(200), ncol=2)
#' inputsTrain  <- matrix(rnorm(200), ncol=2)
#' classesTrain <- c(rep(0, 50), rep(1, 50))
#' # get the kNN predictions for the test set
#' kNN(inputsTest, classesTrain, inputsTrain, k=15, p=2, type="predict")

kNN <- function(data, trueClasses, memory=NULL, 
                k=1, p=2, type="train") {
    
    # test the inputs
    library(assertthat)
    not_empty(X); not_empty(y); 
    if (type == "train") {
        assert_that(nrow(X) == length(y))
    }
    is.string(type); assert_that(type %in% c("train", "predict"))
    is.count(k); 
    assert_that(p %in% c(1, 2, Inf))
    if (type == "predict") {
        assert_that(not_empty(memory) & 
                    ncol(memory) == ncol(X) & 
                    nrow(memory) == length(y))
    }

    # Compute the distance between each point and all others 
    noObs <- nrow(X)
    
    # if we are making predictions on the test set based on the memory, 
    # we compute distances between each test observation and observations
    # in our memory
    if (type == "train") {
        distMatrix <- matrix(NA, noObs, noObs)
        for (obs in 1:noObs) {
            
            # getting the probe for the current observation
            probe <- as.numeric(X[obs,])
            probeExpanded <- matrix(probe, nrow = noObs, ncol = 2, 
                                    byrow = TRUE)

            # computing distances between the probe and exemplars in the
            # training X
            if (p %in% c(1,2)) {
                distMatrix[obs, ] <- (rowSums((abs(X - 
                                      probeExpanded))^p) )^(1/p)
            } else if (p==Inf) {
                distMatrix[obs, ] <- apply(abs(X - probeExpanded), 1, max)
            }  
        }
    } else if (type == "predict") {
        noMemory <- nrow(memory)
        distMatrix <- matrix(NA, noObs, noMemory)
        for (obs in 1:noObs) {
           
            # getting the probe for the current observation
            probe <- as.numeric(X[obs,])
            probeExpanded <- matrix(probe, nrow = noMemory, ncol = 2, 
                                    byrow = TRUE)

            # computing distances between the probe and exemplars in the memory
            if (p %in% c(1,2)) {
                distMatrix[obs, ] <- (rowSums((abs(memory - 
                                      probeExpanded))^p) )^(1/p)
            } else if (p==Inf) {
                distMatrix[obs, ] <- apply(abs(memory - probeExpanded), 1, max)
            }  
        }
    }
    
    # Sort the distances in increasing numerical order and pick the first 
    # k elements
    neighbors <- apply(distMatrix, 1, order) 

    # Compute and return the most frequent class in the k nearest neighbors
    prob <- rep(NA, noObs)
    for (obs in 1:noObs) {
        prob[obs] <- mean(y[neighbors[1:k, obs]])
    }

    # predicted label
    predictedClasses <- ifelse(prob > 0.5, 1, 0)

    # examine the performance, available only if training
    if (type == "train") {
        errorCount <- table(predictedClasses, y)
        accuracy <- mean(predictedClasses == y)
    } else if (type == "predict") {
        errorCount <- NA
        accuracy <- NA
    }

    # return the results
    return(list(predictedClasses = predictedClasses, 
                prob = prob,
                accuracy = accuracy,
                errorCount = errorCount))
}

