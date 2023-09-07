#załadowanie potrzebnych pakietów
library(dplyr)
library(DataExplorer)
library(rpart.plot)   #classification tree
library(rpart)   #classification tree
library(ipred)
library(randomForest)
library(adabag)
library(ggplot2)
#załadowanie danych
df_titanic <- read.csv('C:/Data Science/Python/Raporty/Titanic Ensemble/df_titanic_R.csv')

#przypisanie typów zmiennych
factor_kol <- c('survived', 'pclass', 'sex', 'sibsp', 'parch', 'embarked')
num_kol <- c('age', 'fare')
df_titanic <- df_titanic %>% mutate_at(factor_kol, factor) %>% mutate_at(num_kol, as.numeric)    #(dplyr package)
glimpse(df_titanic)   #sprawdzenie poprawności typów zmiennych

t(introduce(df_titanic))               
plot_intro(df_titanic)                
apply(df_titanic,2,function(x){length(which(is.na(x)))})   #sprawdzenie czy na pewno nie ma wartości brakujących



#---------------------Pętla symulacji----------------------

n <- dim(df_titanic)[1]
M <- 100
#wektory dokładności ogólnej
acc_vec_dec_tree <- c()
acc_vec_bagged_tree_25 <- c()
acc_vec_bagged_tree_150 <- c()
acc_vec_rf_25 <- c()
acc_vec_rf_150 <- c()
acc_vec_boos_25_Bre <- c()
acc_vec_boos_25_Fre <- c()
acc_vec_boos_150_Bre <- c()
acc_vec_boos_150_Fre <- c()

#wektory dokładności dla kategorii survived=0
acc_vec_dec_tree_surv0 <- c()
acc_vec_bagged_tree_25_surv0 <- c()
acc_vec_bagged_tree_150_surv0 <- c()
acc_vec_rf_25_surv0 <- c()
acc_vec_rf_150_surv0 <- c()
acc_vec_boos_25_Bre_surv0 <- c()
acc_vec_boos_25_Fre_surv0 <- c()
acc_vec_boos_150_Bre_surv0 <- c()
acc_vec_boos_150_Fre_surv0 <- c()

#wektory dokładności dla kategorii survived=1
acc_vec_dec_tree_surv1 <- c()
acc_vec_bagged_tree_25_surv1 <- c()
acc_vec_bagged_tree_150_surv1 <- c()
acc_vec_rf_25_surv1 <- c()
acc_vec_rf_150_surv1 <- c()
acc_vec_boos_25_Bre_surv1 <- c()
acc_vec_boos_25_Fre_surv1 <- c()
acc_vec_boos_150_Bre_surv1 <- c()
acc_vec_boos_150_Fre_surv1 <- c()

#wektory czasów obliczeniowych dla każdej metody
acc_vec_dec_tree_time <- 0
acc_vec_bagged_tree_25_time <- 0
acc_vec_bagged_tree_150_time <- 0
acc_vec_rf_25_time <- 0
acc_vec_rf_150_time <- 0
acc_vec_boos_25_Bre_time <- 0
acc_vec_boos_25_Fre_time <- 0
acc_vec_boos_150_Bre_time <- 0
acc_vec_boos_150_Fre_time <- 0

for (i in 1:M){
  learning.set.index <- sample(1:n,0.7*n)
  
  learning.set <- df_titanic[learning.set.index,]
  test.set     <- df_titanic[-learning.set.index,]
  etykietki.rzecz <- test.set$survived
  toc <- Sys.time()
  acc_vec_dec_tree_time <- acc_vec_dec_tree_time+(toc-tic)
  n.test <- dim(test.set)[1]
  
  
  tic <- Sys.time()
  model.dec_tree <- rpart(survived~., data = learning.set, method = 'class', control = (minsplit=2))
  
  etykietki.prog <- predict(model.dec_tree, test.set, type= "class")
  (wynik.tablica <- table(etykietki.prog,etykietki.rzecz))
  accuracy <- (sum(diag(wynik.tablica))) / n.test
  accuracy_surv0 <- wynik.tablica[1,1]/sum(wynik.tablica[,1])
  accuracy_surv1 <- wynik.tablica[2,2]/sum(wynik.tablica[,2])
  
  acc_vec_dec_tree <- c(acc_vec_dec_tree, accuracy)
  acc_vec_dec_tree_surv0 <- c(acc_vec_dec_tree_surv0, accuracy_surv0)
  acc_vec_dec_tree_surv1 <- c(acc_vec_dec_tree_surv1, accuracy_surv1)
  toc <- Sys.time()
  acc_vec_dec_tree_time <- acc_vec_dec_tree_time+(toc-tic)
  
  #dwa modele baggeed trees
  tic <- Sys.time()
  model.bagged_tree <- adabag::bagging(survived~., data = learning.set, nbagg=25,
                               control = rpart.control(minsplit=2) )
  etykietki.prog <- predict(model.bagged_tree, test.set, type='class')
  (wynik.tablica <- table(etykietki.prog$class,etykietki.rzecz))
  accuracy <- (sum(diag(wynik.tablica))) / n.test
  accuracy_surv0 <- wynik.tablica[1,1]/sum(wynik.tablica[,1])
  accuracy_surv1 <- wynik.tablica[2,2]/sum(wynik.tablica[,2])
  
  acc_vec_bagged_tree_25 <- c(acc_vec_bagged_tree_25, accuracy)
  acc_vec_bagged_tree_25_surv0 <- c(acc_vec_bagged_tree_25_surv0, accuracy_surv0)
  acc_vec_bagged_tree_25_surv1 <- c(acc_vec_bagged_tree_25_surv1, accuracy_surv1)
  toc <- Sys.time()
  acc_vec_bagged_tree_25_time <- acc_vec_bagged_tree_25_time+(toc-tic)
  
  
  tic <- Sys.time()
  model.bagged_tree <- adabag::bagging(survived~., data = learning.set, nbagg=150,
                               control = rpart.control(minsplit=2) )
  etykietki.prog <- predict(model.bagged_tree, test.set, type='class')
  (wynik.tablica <- table(etykietki.prog$class,etykietki.rzecz))
  accuracy <- (sum(diag(wynik.tablica))) / n.test
  accuracy_surv0 <- wynik.tablica[1,1]/sum(wynik.tablica[,1])
  accuracy_surv1 <- wynik.tablica[2,2]/sum(wynik.tablica[,2])
  
  acc_vec_bagged_tree_150 <- c(acc_vec_bagged_tree_150, accuracy)
  acc_vec_bagged_tree_150_surv0 <- c(acc_vec_bagged_tree_150_surv0, accuracy_surv0)
  acc_vec_bagged_tree_150_surv1 <- c(acc_vec_bagged_tree_150_surv1, accuracy_surv1)
  toc <- Sys.time()
  acc_vec_bagged_tree_150_time <- acc_vec_bagged_tree_150_time+(toc-tic)
  
  
  #dwa modele random forest
  tic <- Sys.time()
  model.rf <- randomForest(survived ~ .,
                           data = learning.set,
                           method = "rf", 
                           metric = "Accuracy", # which metric should be optimized for 
                           mtry=3,
                           # options to be passed to randomForest
                           ntree = 25,
                           nodesize=5) 
  etykietki.prog <- predict(model.rf, test.set, type='class')
  (wynik.tablica <- table(etykietki.prog,etykietki.rzecz))
  accuracy <- (sum(diag(wynik.tablica))) / n.test
  accuracy_surv0 <- wynik.tablica[1,1]/sum(wynik.tablica[,1])
  accuracy_surv1 <- wynik.tablica[2,2]/sum(wynik.tablica[,2])
  
  acc_vec_rf_25 <- c(acc_vec_rf_25, accuracy)
  acc_vec_rf_25_surv0 <- c(acc_vec_rf_25_surv0, accuracy_surv0)
  acc_vec_rf_25_surv1 <- c(acc_vec_rf_25_surv1, accuracy_surv1)
  toc <- Sys.time()
  acc_vec_rf_25_time  <- acc_vec_rf_25_time +(toc-tic)
  
  
  
  tic <- Sys.time()
  model.rf <- randomForest(survived ~ .,
                           data = learning.set,
                           method = "rf", # this will use the randomForest::randomForest function
                           metric = "Accuracy", # which metric should be optimized for 
                           mtry=3,
                           # options to be passed to randomForest
                           ntree = 150,
                           nodesize=5) 
  etykietki.prog <- predict(model.rf, test.set, type='class')
  (wynik.tablica <- table(etykietki.prog,etykietki.rzecz))
  accuracy <- (sum(diag(wynik.tablica))) / n.test
  accuracy_surv0 <- wynik.tablica[1,1]/sum(wynik.tablica[,1])
  accuracy_surv1 <- wynik.tablica[2,2]/sum(wynik.tablica[,2])
  
  acc_vec_rf_150 <- c(acc_vec_rf_150, accuracy)
  acc_vec_rf_150_surv0 <- c(acc_vec_rf_150_surv0, accuracy_surv0)
  acc_vec_rf_150_surv1 <- c(acc_vec_rf_150_surv1, accuracy_surv1)
  toc <- Sys.time()
  acc_vec_rf_150_time  <- acc_vec_rf_150_time +(toc-tic)
  
  #dwa modele z coef learn='Breiman'
  tic <- Sys.time()
  model.AdaBoost <- boosting(survived~., data=learning.set, boos=TRUE, mfinal=25, coeflearn = "Breiman", control = (minsplit=2))
  
  etykietki.prog <- predict(model.AdaBoost, test.set, type='class')
  (wynik.tablica <- table(etykietki.prog$class,etykietki.rzecz))
  accuracy <- (sum(diag(wynik.tablica))) / n.test
  accuracy_surv0 <- wynik.tablica[1,1]/sum(wynik.tablica[,1])
  accuracy_surv1 <- wynik.tablica[2,2]/sum(wynik.tablica[,2])
  
  acc_vec_boos_25_Bre <- c(acc_vec_boos_25_Bre, accuracy)
  acc_vec_boos_25_Bre_surv0 <- c(acc_vec_boos_25_Bre_surv0, accuracy_surv0)
  acc_vec_boos_25_Bre_surv1 <- c(acc_vec_boos_25_Bre_surv1, accuracy_surv1)
  toc <- Sys.time()
  acc_vec_boos_25_Bre_time <- acc_vec_boos_25_Bre_time+(toc-tic)
  
  tic <- Sys.time()  
  model.AdaBoost <- boosting(survived~., data=learning.set, boos=TRUE, mfinal=150, coeflearn = "Breiman", control = (minsplit=2))
  
  etykietki.prog <- predict(model.AdaBoost, test.set, type='class')
  (wynik.tablica <- table(etykietki.prog$class,etykietki.rzecz))
  accuracy <- (sum(diag(wynik.tablica))) / n.test
  accuracy_surv0 <- wynik.tablica[1,1]/sum(wynik.tablica[,1])
  accuracy_surv1 <- wynik.tablica[2,2]/sum(wynik.tablica[,2])
  
  acc_vec_boos_150_Bre <- c(acc_vec_boos_150_Bre, accuracy)
  acc_vec_boos_150_Bre_surv0 <- c(acc_vec_boos_150_Bre_surv0, accuracy_surv0)
  acc_vec_boos_150_Bre_surv1 <- c(acc_vec_boos_150_Bre_surv1, accuracy_surv1)
  toc <- Sys.time()
  acc_vec_boos_150_Bre_time <- acc_vec_boos_150_Bre_time+(toc-tic)
  
  
  
  #dwa modele z coef learn = Freund
  tic <- Sys.time()
  model.AdaBoost <- boosting(survived~., data=learning.set, boos=TRUE, mfinal=25, coeflearn = "Freund", control = (minsplit=2))
  
  etykietki.prog <- predict(model.AdaBoost, test.set, type='class')
  (wynik.tablica <- table(etykietki.prog$class,etykietki.rzecz))
  accuracy <- (sum(diag(wynik.tablica))) / n.test
  accuracy_surv0 <- wynik.tablica[1,1]/sum(wynik.tablica[,1])
  accuracy_surv1 <- wynik.tablica[2,2]/sum(wynik.tablica[,2])
  
  acc_vec_boos_25_Fre <- c(acc_vec_boos_25_Fre, accuracy)
  acc_vec_boos_25_Fre_surv0 <- c(acc_vec_boos_25_Fre_surv0, accuracy_surv0)
  acc_vec_boos_25_Fre_surv1 <- c(acc_vec_boos_25_Fre_surv1, accuracy_surv1)
  toc <- Sys.time()
  acc_vec_boos_25_Fre_time <- acc_vec_boos_25_Fre_time+(toc-tic)
  
  
  tic <- Sys.time()
  model.AdaBoost <- boosting(survived~., data=learning.set, boos=TRUE, mfinal=150, coeflearn = "Freund", control = (minsplit=2))
  
  etykietki.prog <- predict(model.AdaBoost, test.set, type='class')
  (wynik.tablica <- table(etykietki.prog$class,etykietki.rzecz))
  accuracy <- (sum(diag(wynik.tablica))) / n.test
  accuracy_surv0 <- wynik.tablica[1,1]/sum(wynik.tablica[,1])
  accuracy_surv1 <- wynik.tablica[2,2]/sum(wynik.tablica[,2])
  
  acc_vec_boos_150_Fre <- c(acc_vec_boos_150_Fre, accuracy)
  acc_vec_boos_150_Fre_surv0 <- c(acc_vec_boos_150_Fre_surv0, accuracy_surv0)
  acc_vec_boos_150_Fre_surv1 <- c(acc_vec_boos_150_Fre_surv1, accuracy_surv1)
  
  toc <- Sys.time()
  acc_vec_boos_150_Fre_time <- acc_vec_boos_150_Fre_time+(toc-tic)
  
  
  print(i)
}


acc_df <- cbind.data.frame( c(rep("single decision tree", 100), rep("25 bagged trees", 100), rep("150 bagged trees", 100), rep("random forest 25 trees", 100), rep("random forest 150 trees", 100),
                              rep("boosting 25-Breiman",100), rep("boosting 150-Breiman",100), rep("boosting 25-Freund",100), rep("boosting 150-Freund",100)) , 
                            c(acc_vec_dec_tree, acc_vec_bagged_tree_25, acc_vec_bagged_tree_150, acc_vec_rf_25, acc_vec_rf_150, 
                              acc_vec_boos_25_Bre, acc_vec_boos_150_Bre, acc_vec_boos_25_Fre, acc_vec_boos_150_Fre),
                            c(acc_vec_dec_tree_surv0, acc_vec_bagged_tree_25_surv0, acc_vec_bagged_tree_150_surv0, acc_vec_rf_25_surv0, 
                              acc_vec_rf_150_surv0, acc_vec_boos_25_Bre_surv0, acc_vec_boos_150_Bre_surv0, acc_vec_boos_25_Fre_surv0, 
                              acc_vec_boos_150_Fre_surv0), 
                            c(acc_vec_dec_tree_surv1, acc_vec_bagged_tree_25_surv1, acc_vec_bagged_tree_150_surv1, acc_vec_rf_25_surv1, 
                              acc_vec_rf_150_surv1, acc_vec_boos_25_Bre_surv1, acc_vec_boos_150_Bre_surv1, acc_vec_boos_25_Fre_surv1, 
                              acc_vec_boos_150_Fre_surv1))
colnames(acc_df) <- c("model type", "accuracy", "specificity","sensivity")

acc_df$`model type` <- factor(acc_df$`model type`,     # reorder factor levels
                         c("single decision tree", "25 bagged trees", "150 bagged trees", "random forest 25 trees",
                           "random forest 150 trees", "boosting 25-Freund", "boosting 150-Freund", "boosting 25-Breiman",
                           "boosting 150-Breiman"))


ggplot()+ geom_boxplot(data=acc_df, aes(x=`model type`, y=accuracy))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  ggtitle("Porównanie dokładności drzewa decyzyjnego oraz metod uczenia zespołowego dla \nzbioru danych Titanic")


ggplot()+ geom_boxplot(data=acc_df, aes(x=`model type`, y=accuracy))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=11),
        plot.title = element_text(size=24, hjust=0.5, face='bold'))+
  geom_point(aes(x="single decision tree", y=mean(acc_vec_dec_tree)), size=2.5, color="red")+
  geom_point(aes(x="25 bagged trees", y=mean(acc_vec_bagged_tree_25)), size=2.5, color="red")+
  geom_point(aes(x="150 bagged trees", y=mean(acc_vec_bagged_tree_150)), size=2.5, color="red")+
  geom_point(aes(x="random forest 25 trees", y=mean(acc_vec_rf_25)), size=2.5, color="red")+
  geom_point(aes(x="random forest 150 trees", y=mean(acc_vec_rf_150)), size=2.5, color="red")+
  geom_point(aes(x="boosting 25-Freund", y=mean(acc_vec_boos_25_Fre)), size=2.5, color="red")+
  geom_point(aes(x="boosting 150-Freund", y=mean(acc_vec_boos_150_Fre)), size=2.5, color="red")+
  geom_point(aes(x="boosting 25-Breiman", y=mean(acc_vec_boos_25_Bre)), size=2.5, color="red")+
  geom_point(aes(x="boosting 150-Breiman", y=mean(acc_vec_boos_150_Bre)), size=2.5, color="red")+
  ggtitle("Dokładność")



ggplot()+ geom_boxplot(data=acc_df, aes(x=`model type`, y=specificity))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=11),
        plot.title = element_text(size=24, hjust=0.5, face='bold'))+
  geom_point(aes(x="single decision tree", y=mean(acc_vec_dec_tree_surv0)), size=2.5, color="red")+
  geom_point(aes(x="25 bagged trees", y=mean(acc_vec_bagged_tree_25_surv0)), size=2.5, color="red")+
  geom_point(aes(x="150 bagged trees", y=mean(acc_vec_bagged_tree_150_surv0)), size=2.5, color="red")+
  geom_point(aes(x="random forest 25 trees", y=mean(acc_vec_rf_25_surv0)), size=2.5, color="red")+
  geom_point(aes(x="random forest 150 trees", y=mean(acc_vec_rf_150_surv0)), size=2.5, color="red")+
  geom_point(aes(x="boosting 25-Freund", y=mean(acc_vec_boos_25_Fre_surv0)), size=2.5, color="red")+
  geom_point(aes(x="boosting 150-Freund", y=mean(acc_vec_boos_150_Fre_surv0)), size=2.5, color="red")+
  geom_point(aes(x="boosting 25-Breiman", y=mean(acc_vec_boos_25_Bre_surv0)), size=2.5, color="red")+
  geom_point(aes(x="boosting 150-Breiman", y=mean(acc_vec_boos_150_Bre_surv0)), size=2.5, color="red")+
  ggtitle("Swoistość")

ggplot()+ geom_boxplot(data=acc_df, aes(x=`model type`, y=sensivity))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1,size=11),
        plot.title = element_text(size=24, hjust=0.5, face='bold'))+
  geom_point(aes(x="single decision tree", y=mean(acc_vec_dec_tree_surv1)), size=2.5, color="red")+
  geom_point(aes(x="25 bagged trees", y=mean(acc_vec_bagged_tree_25_surv1)), size=2.5, color="red")+
  geom_point(aes(x="150 bagged trees", y=mean(acc_vec_bagged_tree_150_surv1)), size=2.5, color="red")+
  geom_point(aes(x="random forest 25 trees", y=mean(acc_vec_rf_25_surv1)), size=2.5, color="red")+
  geom_point(aes(x="random forest 150 trees", y=mean(acc_vec_rf_150_surv1)), size=2.5, color="red")+
  geom_point(aes(x="boosting 25-Freund", y=mean(acc_vec_boos_25_Fre_surv1)), size=2.5, color="red")+
  geom_point(aes(x="boosting 150-Freund", y=mean(acc_vec_boos_150_Fre_surv1)), size=2.5, color="red")+
  geom_point(aes(x="boosting 25-Breiman", y=mean(acc_vec_boos_25_Bre_surv1)), size=2.5, color="red")+
  geom_point(aes(x="boosting 150-Breiman", y=mean(acc_vec_boos_150_Bre_surv1)), size=2.5, color="red")+
  ggtitle("Czułość")

#zapis wyników
#save.image(file = "C:/Data Science/Python/Raporty/Titanic Ensemble/Wykresy Accuracy/Wyniki.RData")


#Czasy-------nieużywane
times_df <-    cbind.data.frame( c("single decision tree", "25 bagged trees", "150 bagged trees", "random forest 25 trees","random forest 150", 
                                   "boosting 25-Breiman", "boosting 150-Breiman", "boosting 25-Freund", "boosting 150-Freund"),
                                 c(acc_vec_dec_tree_time, acc_vec_bagged_tree_25_time, acc_vec_bagged_tree_150_time, acc_vec_rf_25_time, 
                                   acc_vec_rf_150_time, acc_vec_boos_25_Bre_time, acc_vec_boos_150_Bre_time, acc_vec_boos_25_Fre_time, 
                                   acc_vec_boos_150_Fre_time))
times_df[,2] <- as.numeric(times_df[,2])
colnames(times_df) <- c("model type", "evaluation time")
                                                       
ggplot()+ geom_bar(data=times_df, aes(x=`model type`, y=`evaluation time`), stat="identity")+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  # geom_point(data=acc_df,aes(x=`model type`, y=mean(accuracy) ), size=2.8, color="red")+
  ggtitle("Porównanie czasów ewaluacji metod uczenia zespołowego \ndla zbioru danych Titanic")

