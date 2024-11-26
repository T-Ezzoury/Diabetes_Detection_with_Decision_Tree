#Taha EZ-ZOURY
#Youness BOUALLOU 
#Marouane EL BISSOURI


library(rpart)
library(caret)
library(e1071)
library(ROCR)
library(party)
library(randomForest)
library(ggplot2)
library(reshape2)


rm(list=ls())
set.seed(1234)
data = read.csv("diabetes detection.csv", sep=";", header= TRUE)
head(data)
View(data)


#on transforme les valeurs de la classe en 0 et 1
data$class=ifelse(data$class == "tested_positive", 1, 0)
#set class to qualitative variable
data$class <- factor(data$class)

View(data)
#preg= nb of time the person was pregnant
#plas= plasma Glucose concentration
#pres= blood pressure
#skin= skin thickness
#insu= The serum insulin level, indicator of insulin resistance(prblm)
#mass= measure of body fat based on height and weight
#pedi= DiabetesPedigreeFunction a function that represents the likelihood of diabetes based on family history.
#the higher pedi, the higher the likelihood that an individual has a family history of diabetes

data[!complete.cases(data), ]
print(paste("somme des na values:", sum(is.na(data))))
#--> no missing values in data


# Créer des boxplots pour toutes les variables
melted_data <- melt(data)
ggplot(melted_data, aes(x = variable, y = value)) +
  geom_boxplot() +
  labs(title = "Boxplots pour toutes les variables")


#contruction d'une base de données pour construire le modèle et l'autre pour le tester
#on choisit aleatoirement 30% de data pour le test
index <- sample(1:nrow(data),round(0.70*nrow(data)))
train <- data[index,]
test <- data[-index,]

#to clean data in case there are na values (to be predicted)
clean_data= rfImpute(class ~ ., data= data, iter=6) #iter= nb de forets



#2. implementation de l'arbre
tree = rpart(class~preg+plas+pres+skin+insu+mass+pedi+age, data=train, method="class")
rpart.plot::rpart.plot(tree)
tree$cptable

#arbre élaguée par minimisation de R_alpha
min_ind <- which.min(tree$cptable[, "xerror"])
min_cp <- tree$cptable[min_ind, "CP"]
pruned_tree <- rpart::prune(tree, cp = min_cp)
rpart.plot::rpart.plot(pruned_tree)

#3. 
#prediction des donnees test + erreur de prediction
predicted <- predict(pruned_tree, test, type="class")
error1=sum(test$class != predicted)/length(predicted)
print(paste("précision :",(1-error1)*100,"%"))

# Matrice de confusion
predicted <- ordered(predicted, levels = c(1, 0))
actual<- ordered(test$class, levels = c(1, 0))
mc=table(predicted,actual, dnn=c( "Predicted","reelle"))

#matrice de confusion
mc

#sensibility
se= mc[1,1]/(mc[1,1]+mc[2,1])
se
#ou
se= sensitivity(mc)
se
print(paste("sensibilite :",(se)*100,"%"))

#specificité
sp= mc[2,2]/(mc[2,2]+mc[1,2])
sp
#ou
sp= specificity(mc)
print(paste("specificite :",(sp)*100,"%"))

#high sp ==> model is good at identifying negative outcomes
#modest se ==> model is quite good at identifying positive cases

#F-score
f_score= 2*(sp*se)/(se+sp)
print(paste("f_score :",(f_score)*100,"%"))

#ROC curve
Predprob <- predict(pruned_tree, newdata = test,type = "prob")
Predprob = as.data.frame(Predprob)
Prediction <- prediction(Predprob[2],test$class)
performance <- performance(Prediction, "tpr","fpr")
plot(performance,main = "ROC Curve",col = 2,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")


#final metric: area under curve
aucDT <- performance(Prediction, measure = "auc")
aucDT <- aucDT@y.values[[1]]; aucDT
#AUC ≈ 0.5: The model performs no better than random chance
#0.5 < AUC < 0.7: The model's performance is poor
#0.7 < AUC < 0.8: The model's performance is fair
#0.8 < AUC < 0.9: The model's performance is good

fpr = attr(performance, "x.values")[[1]]
tpr = attr(performance, "y.values")[[1]]

# point plus proche de (0,1)
closest_to_01 = function(fpr, tpr){
  n=length(fpr)
  xopt=0
  yopt=0
  distance=1
  for (i in 1:n){
    if (fpr[i]*fpr[i]+(tpr[i]-1)*(tpr[i]-1)<distance*distance){
      distance= sqrt(fpr[i]*fpr[i]+(tpr[i]-1)*(tpr[i]-1))
      xopt=fpr[i]
      yopt=tpr[i]
    }
  }
  return(c(xopt, yopt))
}
closest_to_01(fpr, tpr)



#pb3----------------------------------------
#3. traçons le LIFT et LIFT normalisé

rpp = c(0.235, 0.321, 0.494, 1.0)
se = c(0.647, 0.882, 1.0, 1.0)
lift= se/rpp; lift
plot(rpp, lift , typ='l', col= "red", main= "courbe LIFT normalisé")
plot(rpp, se , typ='l', col= "blue", main= "courbe LIFT")

#extrapolation(not interpolation) (Ouverture)
data1 = data.frame(rpp, lift)
model <- lm(lift ~ rpp, data = data1)
extrapolation_data <- data.frame(rpp = 0.1)
extrapolated_y <- predict(model, newdata = extrapolation_data)
#interpolation of rpp from se=0.76
interpolated_rpp <- approx(se, rpp, xout = 0.76)$y
#courbe coutparpersonne
cout_pp= (81*rpp)/(17*se)
plot(rpp, cout_pp, type="l", col="green", main="cout par personne")                          

#interpolation of rpp from se=14/17
inter <- approx(se, rpp, xout = 14/17)$y
plot(rpp, se , typ='l', col= "blue", main= "courbe LIFT")
points(c(0.2996023), c(se=14/17), pch = 19, col = "red")
abline(h=14/17, lty = 3)
abline(v=0.2996023, lty = 3)





#partie4----------------------------------------------

#STATQUEST
library(ggplot2)
library(cowplot)
set.seed(24)

#Trouvons le nombre d'arbre à utiliser
md <- randomForest(class ~ ., data=data, proximity=TRUE, ntree=1000)
head(md$err.rate)
oob.error.data <- data.frame(
  Trees=rep(1:nrow(md$err.rate), times=3),
  Type=rep(c("OOB", "0", "1"), each=nrow(md$err.rate)),
  Error=c(md$err.rate[,"OOB"], 
          md$err.rate[,"0"], 
          md$err.rate[,"1"]))
ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

#maintenant trouvons m minimisant l'erreur
oob.values <- vector(length=10)
for(i in 1:10) {
  temp.model <- randomForest(class ~ ., data=data, mtry=i, ntree=1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values
which(oob.values == min(oob.values))
md$proximity

#1. out of sample estimation

library(randomForest)
ranf1 <- randomForest(class ~ ., data = train, ntree = 700, mtry = 2)
print(ranf1)

#Variable importance
imp1= ranf1$importance[order(ranf1$importance[, 1], decreasing = TRUE), ]
imp1

#most important features: plas,age,mass,pedi
View(test)

#error
pr1 <- predict(ranf1, newdata = test)
mc1 <- table(pr1, test$class)
err1= 1-((mc1[1,1]+mc1[2,2])/sum(mc1))
print(paste("Précision :",(1-err1)*100,"%"))

# sensibilite et specificite
se1 = mc1[2,2]/(mc1[1,2]+mc1[2,2])
sp1 = mc1[1,1]/(mc1[1,1]+mc1[2,1])
print(paste("sensibilité :",se1*100,"%"))
print(paste("specificité :",sp1*100,"%"))

# ROC & auc
library(pROC)
Predprob1 <- predict(ranf1, newdata = test,type = "prob")
Predprob1 = as.data.frame(Predprob1)
Prediction1 <- prediction(Predprob1[2],test$class)
performance1 <- performance(Prediction1, "tpr","fpr")
plot(performance1,main = "ROC Curve",col = 2,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")
aucDT1 <- performance(Prediction1, measure = "auc")
aucDT1 <- aucDT1@y.values[[1]]
print(paste("area under curve :", aucDT1))
# point optimale
fpr1 = attr(performance1, "x.values")[[1]]
tpr1 = attr(performance1, "y.values")[[1]]
print(paste("point optimale :", closest_to_01(fpr1, tpr1)[1], closest_to_01(fpr1, tpr1)[2]))


#-----------
#2. construction d'un model just avec les features les plus importantes selon les 2 modèles précedents
select_train= train[, c("plas","age","mass","pedi","class")]
formula <- class ~ plas+age+mass+pedi
ranf2<- randomForest(formula, data = select_train, ntree = 700, 
                         mtry = 2)
select_test= test[, c("plas","age","mass","pedi","class")]
#error
pr2 <- predict(ranf2, newdata = select_test)
mc2 <- table(pr2, select_test$class)
err2= 1-((mc2[1,1]+mc2[2,2])/sum(mc2))
print(paste("Précision :",(1-err2)*100,"%"))


# sensibilite et specificite
se2 = mc2[2,2]/(mc2[1,2]+mc2[2,2])
sp2 = mc2[1,1]/(mc2[1,1]+mc2[2,1])
print(paste("sensibilité :",se2*100,"%"))
print(paste("specificité :",sp2*100,"%"))

# ROC & auc
library(pROC)
Predprob2 <- predict(ranf2, newdata = test,type = "prob")
Predprob2 = as.data.frame(Predprob2)
Prediction2 <- prediction(Predprob2[2],select_test$class)
performance2 <- performance(Prediction2, "tpr","fpr")
plot(performance2,main = "ROC Curve",col = 2,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")
aucDT2 <- performance(Prediction2, measure = "auc")
aucDT2 <- aucDT2@y.values[[1]]
print(paste("area under curve :", aucDT2))
# point optimale
fpr2 = attr(performance2, "x.values")[[1]]
tpr2 = attr(performance2, "y.values")[[1]]
print(paste("point optimale :", closest_to_01(fpr2, tpr2)[1], closest_to_01(fpr2, tpr2)[2]))


#-------------
#3. cross validation k=5
trainControl <- trainControl(method = "cv", number = 5)
ranf3 <- train(class ~ ., data = train, method = "rf", trControl = trainControl)
pr3= predict(ranf3,newdata = test)
#error
mc3= confusionMatrix(pr3, test$class)
print(mc3)
acc3 <- mc3$overall["Accuracy"]
err3 = 1-acc3
print(paste("Précision :",(1-err3)*100,"%"))

#sensibilite et specificite
sp3 = mc3$table[1, 1] / sum(mc3$table[1, ])
se3 <- mc3$table[2, 2] / sum(mc3$table[2, ])
print(paste("sensibilité :",se3*100,"%"))
print(paste("specificité :",sp3*100,"%"))

# ROC & auc
library(pROC)
Predprob3 <- predict(ranf3, newdata = test,type = "prob")
Predprob3 = as.data.frame(Predprob3)
Prediction3 <- prediction(Predprob3[2],select_test$class)
performance3 <- performance(Prediction3, "tpr","fpr")
plot(performance3,main = "ROC Curve",col = 2,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")
aucDT3 <- performance(Prediction3, measure = "auc")
aucDT3 <- aucDT3@y.values[[1]]
print(paste("area under curve :", aucDT2))
# point optimale
fpr3 = attr(performance3, "x.values")[[1]]
tpr3 = attr(performance3, "y.values")[[1]]
print(paste("point optimale :", closest_to_01(fpr3, tpr3)[1], closest_to_01(fpr3, tpr3)[2]))





#acc= 0.74 presque égale à celle des modèles (0.769 & 0.763)
#for further investigation on construit d'autres modèles en se basant sur d'autres échantillons train et test et on evalue leurs performances
#we can also get the optimal mty value and same for the ntree parameter although R optimizes them when not specified









