# Autor: Filip Żabicki (382713)

## Użyte biblioteki ----
library(ROCR)
library(ipred)
library(caTools)
library(rattle)
library(rpart)
library(caret)
library(randomForest)
library(adabag)
library(gbm)
library(pROC)
library(neuralnet)
library(nnet)
library(NeuralNetTools)

# 1. ----

# Ustawiam seed dla całego skryptu
set.seed(1337)

## Definicja przydatnych funkcji ----

# Zwraca dokładność, jako prosty stosunek dobrze przewidzianych do wszystkich 
acc <- function(y.true, y.pred) { sum(y.pred==y.true)/length(y.true) }

# Plotuje krzywą ROC
roc.function <- function(y_pred, testY) {
  pred <- prediction(as.numeric(y_pred), as.numeric(testY))
  perf.auc <- performance(pred, measure = "auc")
  auc <- round(unlist(perf.auc@y.values), 2)
  perf <- performance(pred, "tpr", "fpr")
  plot(perf, main=paste("Krzywa ROC i parametr AUC=", auc), colorize=TRUE, lwd = 3)
  abline(a = 0, b = 1, lwd = 2, lty = 2)
}

# Zapisuje koszyk drzew do pliku pdf.
trees.bag.to.pdf <- function(bag) {
  n.bags <- length(bag$mtrees)
  pdf("Bagging_Trees.pdf")
  for(i in 1:n.bags){
    tree <- bag$mtrees[[i]]
    plot(tree$btree)
    text(tree$btree, use.n=TRUE)
  }
  dev.off()
}


## Wczytanie zbioru (income.csv) ----
Income <- read.csv("income.csv", stringsAsFactors = T)
summary(Income)

split <- sample.split(Income$income, SplitRatio = 0.7)
Income_train <- subset(Income, split == TRUE)
Income_test <- subset(Income, split == FALSE)
summary(Income_test)
# Warto zauważyć że niektóre obserwacje posiadają symbol '?' zamiast informacji,
#  ale nie powinno to być problemem

prop.table(table(Income$income))
prop.table(table(Income_train$income))
prop.table(table(Income_test$income))
# Podział wyszedł prawidłowo, zbiory treningowy i testowy mają identyczny
#  stosunek ilości obserwacji należących do objaśnianych grup.
# Nie jest konieczne, aby liczba obserwacji w klasach była równa przy użyciu lasów losowych 
#  lub baggingu (<=50K to ok 76% danych).



## Algorytm bagging ----
Income_bags <- bagging(income~.,
                       Income_train,
                       coob=FALSE,
                       nbagg=80,
                       keepX=TRUE)
Income_bags
Income_bags_pred <- predict(Income_bags,
                           newdata = subset(Income_test, select = -income),
                           type="class")

# macierz pomyłek
table(Income_bags_pred, Income_test$income)
# dokładność 
bag_acc <- acc(Income_bags_pred, Income_test$income)
bag_acc
# 0.8180794 - bardzo przyzwoity wynik
# Krzywa ROC
roc.function(Income_bags_pred, Income_test$income)
# Ona również nie wygląda źle, czułość (true positive rate) rośnie szybko dla niedużych wartości
#  swoistości (false positive rate), ale od wartości czułości 0.6 dalszy wzrost \
#  odbywa się już co raz większym kosztem swoistości. AUC wynosi 0.73

# Można przykładowo wyplotować pierwsze drzewo z koszyka, 
#  ale jest ono skrajnie nieczytelne
first_bag_tree <- Income_bags$mtrees[[1]]
fancyRpartPlot(first_bag_tree$btree)
plot(first_bag_tree$btree)

# Szukanie optymalnej ilości koszyków
n.bags <- seq(10, 100, by = 10)
accuracy_values <- numeric()

# Wstępne sprawdzenie czy warto optymalizować liczbę koszyków,
#   przy użyciu pętli
# LEPIEJ NIE URUCHAMIAĆ (trwa chwilę nawet przy tylko 10 iteracjach)
for (bag in n.bags) {
  Income_bags <- bagging(income ~ .,
                         Income_train,
                         coob = FALSE,
                         nbagg = bag,
                         keepX = TRUE)
  
  Income_bags_pred <- predict(Income_bags,
                              newdata = subset(Income_test, select = -income),
                              type = "class")
  
  # Macierz pomyłek
  confusion_matrix <- table(Income_bags_pred, Income_test$income)
  
  # Dokładność 
  bag_acc <- acc(Income_bags_pred, Income_test$income)
  accuracy_values <- c(accuracy_values, bag_acc)
  
  cat(paste("n.bags:", bag, "Accuracy:", bag_acc, "\n"))
}
accuracy_values
# [1] 0.8087633 0.8136773 0.8165438 0.8177723 0.8171581 0.8165438 0.8198198 0.8165438 0.8165438 0.8182842
accuracy_values <- c(0.8087633, 0.8136773, 0.8165438, 0.8177723, 0.8171581, 0.8165438, 0.8198198, 0.8165438, 0.8165438, 0.8182842)

# Wykres zależności dokładności od ilości n.bags
plot(n.bags, accuracy_values, type = "l", xlab = "n.bags", ylab = "Accuracy", main = "Zależność dokładności od ilości n.bags")
# Jak widać optymalizacja tego parametru nieznacznie wpływa na poprawę dokładności,
#  za to jest bardzo kosztowna czasowo/obliczeniowo.
# Optymalizacja pozostałych hiperparametrów nie jest tak dogodna w tym pakiecie,
#  jak np. przy użyciu train z pakietu caret, więc pominę ją.

## Las losowy ----

p <- sqrt(length(Income))
Income_rf <- randomForest(income~.,
                          data=Income_train,
                          ntree=100,
                          mtry=as.integer(p),
                          localImp=TRUE)
Income_rf_pred <- predict(Income_rf,
                          newdata = subset(Income_test, select = -income),
                          type="class")

table(Income_rf_pred, Income_test$income)
acc.rf <- acc(Income_rf_pred, Income_test$income)
acc.rf
# [1] 0.8399877
roc.function(Income_rf_pred, Income_test$income)
# Dokładność bez dostrajania hiperparametrów jest parę pkt. procentowych  lepsza
#  niż w przypadku baggingu, krzywa ROC jest bardzo podobna. AUC = 0.76.
#  Cały algorytm działa znacznie szybciej niż bagging,
#  dlatego że tworzymy drzewa które korzystają
#  tylko z części atrybutów.

# Wykres błędu średnio kwadratowego OOB 
Income_rf$err.rate
plot(Income_rf)
Income_rf.legend <- colnames(Income_rf$err.rate)
legend(x=5,
       y=0.3,
       legend = Income_rf.legend,
       lty=c(1,2,3),
       col=c(1,2,3))
# Krzywe są "daleko" od siebie co oznacza że model nie jest najlepszy.

# Wykres istotności zmiennych
Income_rf$importance
varImpPlot(Income_rf,
           main="Wykres istotności zmiennych")
# 5 Najważniejszych zmiennych to: age, relationship, occupation, martialStatus oraz workHours


# Poszukiwanie optymalnego mtry
tuned_model_rf <- tuneRF(x = subset(Income_train, select = -income),
                         y = Income_train$income,
                         ntreeTry = 100,
                         trace = TRUE,
                         plot = TRUE)
# mtry = 3  OOB error = 17.11% 
# Searching left ...
# mtry = 2 	OOB error = 16.45% 
# 0.03846154 0.05 
# Searching right ...
# mtry = 6 	OOB error = 18.33% 
# -0.07128205 0.05 
# Jak widać zasada sqrt(liczba_zmiennych), niesprawdza się w tym przypadku.
#  Lepiej użyć mtry=

Income_rf_tuned <- randomForest(income~.,
                                data=Income_train,
                                ntree=100,
                                mtry=2,
                                localImp=TRUE)
rf_tuned_pred <- predict(Income_rf_tuned,
                         newdata = subset(Income_test, select = -income),
                         type="class")

rf_tuned_pred
table(rf_tuned_pred, Income_test$income)
acc.rf <- acc(rf_tuned_pred, Income_test$income)
acc.rf
# [1] 0.8453112
roc.function(rf_tuned_pred, Income_test$income)
# Model poradził sobie nieznacznie lepiej oceniając po dokłądności. Krzywa ROC
#  bardzo podobna do tej w modelu niedostrojonego. AUC = 0.76


## Boosting, Ada ----
Income_boo <- boosting(income~.,
                       Income_train,
                       mfinal=50)
# Zdecydowanie najwolniej tworzony model dotychczas.

# Istotnosć zmiennych
Income_boo$importance

boo_pred <- predict(Income_boo,
                    newdata=subset(Income_test, select = -income),
                    type="class")
boo_pred$confusion
boo_pred$class
table(boo_pred$class, Income_test$income)
# [1] 0.8408067
acc(boo_pred$class, Income_test$income)
roc.function(boo_pred$prob[,2], Income_test$income)
# Model poradził sobie bardzo podobnie do poprzednich oceniając po dokłądności.
#  Krzywa ROC natomiast wygląda najlepiej z wszystkich.
#  AUC również osiąga największą wartość 0.89


## Metoda gradientowa gbm ----

# Przekształcenie danych do postaci numerycznej przy użyciu dummyVars
Income_dv <- dummyVars("~ .",
                       subset(Income, select = -income),
                       fullRank = FALSE)
Income_dv

# Stworzenie ramki danych Income_d z przekształconymi danymi
Income_d <- as.data.frame(predict(Income_dv,
                                  newdata=subset(Income, select = -income)))
Income_d

# Dodanie kolumny z klasą dochodu do ramki danych
Income_d <- cbind(Income_d, Income$income)
Income_d

# Zmiana nazwy kolumny na "income" i przekształcenie wartości do postaci numerycznej (0, 1)
names(Income_d)[names(Income_d) == "Income$income"] <- "income"
Income_d$income <- as.numeric(Income_d$income) - 1
Income_d

str(Income_d)
summary(Income_d)

# Podział danych na zbiór treningowy i testowy
split <- sample.split(Income_d$income, SplitRatio = 0.7)
Income_d_train <- subset(Income_d, split == TRUE)
Income_d_train

Income_d_test <- subset(Income_d, split == FALSE)
Income_d_test

str(Income_d_train)

summary(Income_d$income)
prop.table(table(Income_d$income))
prop.table(table(Income_d_train$income))

# Budowa modelu gradientowego gbm
Income_d_gbm <- gbm(income~.,
                    distribution = "bernoulli",
                    data = Income_d_train,
                    n.trees = 500,
                    shrinkage = 0.05)
Income_d_gbm

# Podsumowanie modelu
summary(Income_d_gbm)

# Optymalna l drzew
l.drzew <- gbm.perf(Income_d_gbm,
                    plot.it = TRUE)
l.drzew

# Predykcja na zbiorze testowym
Income_d_gbm_pred <- predict(Income_d_gbm,
                             Income_d_test,
                             n.trees = l.drzew,
                             type = "response")
Income_d_gbm_pred

# Krzywa ROC
gbm.roc = roc(Income_d_test$income, Income_d_gbm_pred)
plot(gbm.roc)
coords(gbm.roc, "best")
# AUC=0.8904, wartości bliskie tych z modelu ada, wygląd krzywej prawie identyczny.

# Klasyfikacja na podstawie progu 0.75
Income_d_gbm_pred_class <- ifelse(Income_d_gbm_pred > 0.6883459, 1, 0)
Income_d_gbm_pred_class

# Tabela kontyngencji i dokładność klasyfikacji
table(Income_d_gbm_pred_class, Income_d_test$income)
acc(Income_d_gbm_pred_class, Income_d_test$income)
# [1] 0.8173628 - dokładność bez szału i zaskoczeń, parę pkt. proc. gorsza niż
#  w przypadku ada boost i lasów losowych.

summary(Income_d_gbm)
## Podsumowanie ----
# Najlepsze okazały się metody boostingiem, jednak są one najbardziej wymagające obliczeniowo.

# 2. ----

## Definicja przydatnych funkcji ----

# Normalizowanie zmiennych
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Błąd średniokwadratowy
MSE <- function(y.true, y.pred) { sum((y.true - y.pred)^2)/length(y.true) }

# Wylicza wsp. determinacji
r2 <- function(pred, actual) {
  rss = sum((actual - pred)^2) ## residual sum of squares
  tss = sum((actual - mean(actual)) ^ 2) ## total sum of squares
  result = 1 - rss / tss
  return(result)
}

## Wczytanie zbioru (voice.csv)----
voice <- read.csv("voice.csv", stringsAsFactors = T)
summary(voice)
voice$label <- as.numeric(voice$label) - 1

voicen <- as.data.frame(lapply(voice,
                               normalize))

inTrain <- createDataPartition(y=voicen$label,
                               p=0.75,
                               list=FALSE)
vn_train <- voicen[inTrain,]
vn_test <- voicen[-inTrain,]

## Model z pakietu neuralnet ----
model <- neuralnet(label~.,
                   data = vn_train)
plot(model)

wyniki1 <- compute(model, vn_test)
predicted_label <- wyniki1$net.result

MSE(predicted_label, vn_test$label)
# [1] 0.01884352
cor(predicted_label, vn_test$label)
# [1,] 0.9616644
# Bez hiperparametryzacji i przy domyślnych argumentach z jednym węzłem ukrytym
#  udało nam się uzyskać bardzo niski błąd średniokwadratowy i wysoką korelację
#  przewidzianych przez siec labeli z faktycznymi ze zbioru testowego.

r2(wyniki1$net.result, vn_test$label)*100
# [1] 92.46259
# Jest to procent wariancji zmiennej zależnej, którą model jest w stanie wyjaśnić.

model_przesadzony <- neuralnet(label~.,
                               data = vn_train,
                               hidden = c(2,3,2))
plot(model_przesadzony)
wyniki2 <- compute(model_przesadzony, vn_test)
predicted_label2 <- wyniki2$net.result

MSE(predicted_label2, vn_test$label)
# [1] 0.02203051
cor(predicted_label2, vn_test$label)
# [1,] 0.9551988
r2(predicted_label2, vn_test$label)*100
# [1] 91.1878
# Udało nam się pogorszyć wyniki zwiększając złożoność sieci.


## Model z pakietu nnet ----
model_nn <- nnet(label ~ .,
                 data = vn_train,
                 size = 1,
                 maxit=10000)
summary(model_nn)
# iter1250 value 42.572972
# final  value 42.572421 
# converged

nn_predict <- predict(model_nn, vn_test)
MSE(nn_predict, vn_test$label)
# [1] 0.02371085
cor(nn_predict, vn_test$label)
# [1,] 0.9517726
r2(nn_predict, vn_test$label)*100
# [1] 90.51566
# Wyniki zbliżone do poprzednich z modelem pakietu neuralnet.

# Dostrajanie sieci ----

control <- trainControl(method="cv",
                        number=5)
# Siatka hiperparametrów, 
siatka <- expand.grid(.decay=seq(0.00, 0.02, by=0.0025),
                      .size=seq(1, 10, by=1))
nn_model_tuned = train(label ~ ., data = vn_train,
                       method='nnet', tuneGrid=siatka,
                       trControl=control, maxit = 100)
plot(nn_model_tuned)
# Z wykresu widać że zwiększanie liczby węzłów ukrytych nieznacznie zmniejsza błędy
#  natomiast parametr decay daje znacznie gorsze wyniki gdy jest równy 0 niż w jakimkolwiek
#  innym przypadku.
nn_model_tuned$bestTune
#     size decay
# 49    9  0.01
summary(nn_model_tuned)
plotnet(nn_model_tuned , alpha=0.6)
nn_model_tuned_predict <- predict(nn_model_tuned, vn_test)
MSE(nn_model_tuned_predict , vn_test$label)
# [1] 0.01658979
cor(nn_model_tuned_predict , vn_test$label)
# [1] 0.9664799
r2(nn_model_tuned_predict, vn_test$label)*100
# [1] 93.36408
# Stosunkowo nieduże poprawienie wyników klasyfikacji odbyło się kosztem znacznego
#  skomplikowania modelu (9 węzłów ukrytych kontra 1 w poprzednim modelu).
#  Prawdopodobnie dalsze dostrajenie hiperparametrów mogłoby poprawić wynik, ale
#  złożoność modelu będzie rosła o wiele szybciej niż progres wyników,
#  dodatkowo poszukiwania optymalnej sieci pochłonełby czas. 