# ------------------------------------------------------------------------------
# 
#           Microsoft Power BI para Data Science, Versão 2.0
# 
#                         Data Science Academy
# 
#                           Mini-Projeto 3
# 
#     Prevendo a Inadimplência de Clientes com Machine Learning e Power BI
# 
# ------------------------------------------------------------------------------

# ----- Definindo a pasta de trabalho -----
setwd("C:/Users/Vinicius Chaves/Desktop/Self_study_R/DSA-PBI-R-Cap15")
getwd()

# ----- Definição do Problema -----
# Leio ao manual em PDF no Cap 15 do curso como a definição do problema

# ----- Instalando os pacotes para o projeto -----
# OBS: Os pacotes precisa ser instalados apenas uma vez
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")

# ----- Carregando os pacotes -----
library(Amelia)
library(caret)
library(ggplot2)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)

# ----- Carregando o dataset -----
# Fonte: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
dados_clientes <- read.csv("dados/dataset.csv")

# Visualizando os dados e sua estrutura
View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

# ----- Sumário das variáveis do dataset -----
# X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
# 
# X2: Gender (1 = male; 2 = female).
# 
# X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
# 
# X4: Marital status (1 = married; 2 = single; 3 = others).
# 
# X5: Age (year).
# 
# X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. 
# The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
# 
# X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
# 
# X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.
# ----- Análise Exploratória, Limpeza e Transformação -----

# Removendo a primeira coluna ID
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

# Renomeando a coluna de classe
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "inadimplente"
colnames(dados_clientes)
View(dados_clientes)

# Verificando valoresr ausentes e removendo-os do dataset
sapply(dados_clientes, function(x) sum(is.na(x)))
# ?missmap
# missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)

# Convertendo os atributos gênero, escolaridade, estado civil e idades para fatores (categorias)

# Renomeando as colunas categóricas
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)
View(dados_clientes)

# Gênero
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut
dados_clientes$Genero <- cut(dados_clientes$Genero,
                             c(0,1,2),
                             labels = c("Masculino", 
                                        "Feminino"))
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)

# Escolaridade
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
                                  c(0,1,2,3,4),
                                  labels = c("Pos-graduado", 
                                             "Graduado",
                                             "Encino Médio",
                                             "Outros"))
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)

# Estado Civil
View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil,
                                   c(-1,0,1,2,3),
                                   labels = c("Desconhecido", 
                                              "Casado",
                                              "Solteiro",
                                              "Outros"))
View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)

# Convertendo a variável para o tipo fator com faixa etária
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
hist(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade,
                                   c(0,30,50,100),
                                   labels = c("Jovem", 
                                              "Adulto",
                                              "Idoso"))
View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
View(dados_clientes)

# Convertendo as variáveis que indicam pagamentos para o tipo fator
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_1 <- as.factor(dados_clientes$PAY_1)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

# Dataset após conversões
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
dados_clientes <- na.omit(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
dim(dados_clientes)

# Alterando a variável dependente para o tipo fator
str(dados_clientes$inadimplente)
colnames(dados_clientes)
dados_clientes$inadimplente <- as.factor(dados_clientes$inadimplente)
str(dados_clientes$inadimplente)
View(dados_clientes)

# Total de inadimplentes versus não-inadimplentes
table(dados_clientes$inadimplente)

# Vejamos as proporções ente as classes
prop.table(table(dados_clientes$inadimplente))

# Plot da distribuição usando ggplot2
qplot(inadimplente, data = dados_clientes, geom = "bar") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Set seed
set.seed(12345)

# Amostragem estratificada
# Seleciona as linhas de acordo com a variável inadimplente como strata
?createDataPartition
indice <- createDataPartition(dados_clientes$inadimplente, p = 0.75, list=FALSE)
dim(indice)

# Definimos os dados de treinamento como um subconjunto do conjunto de dados original
# com numero de indices de linha (Conforme identificado acima) e todas as colunas
dados_treino <- dados_clientes[indice,]
table(dados_treino$inadimplente)

# Veja a porcentagem entre as classes
prop.table(table(dados_treino$inadimplente))

# Numero de registros no dataset de treinamento
dim(dados_treino)

# Comparamos as porcentagens entre as classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$inadimplente)),
                       prop.table(table(dados_clientes$inadimplente)))
colnames(compara_dados) <- c("Treinamento","Original")
compara_dados

# Melt data - Converte colunas em linhas
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# Plot para ver a distribuisão do treinamento vs original
ggplot(melt_compara_dados, aes(x = X1, y = value))+
  geom_bar(aes(fill=X2), stat='identity', position = 'dodge')+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Tudo que não está no dataset de treinamento está no dataset de teste, Observe o sinal - (menos)
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)

# ----- Modelo de Machine Learning -----

# Construindo a primeira versão do modelo
?randomForest
modelo_v1 <- randomForest(inadimplente ~ ., data = dados_treino)
modelo_v1

# Avaiando o modelo 1
plot(modelo_v1)

# Previsões com os dados de teste 
prevsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(prevsoes_v1, dados_teste$inadimplente, positive="1")
cm_v1

# Calculando Precision, Recall e F1-score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v1 <- prevsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Balanceamento de classes
install.packages("DMwR")
library(DMwR)
?SMOTE

# Aplicando o SMOTE - SMOTE: Synthetic Minority Over-Sampling Techinique
# http://arxiv.org/pdf/1106.1813.pdf
table(dados_treino$inadimplente)
prop.table(table(dados_treino$inadimplente))
set.seed(9560)
dados_treino_bal <- SMOTE(inadimplente ~ ., data = dados_treino)
table(dados_treino_bal$inadimplente)
prop.table(table(dados_treino_bal$inadimplente))

# Construnido a segunda versão do modelo 2
modelo_v2 <- randomForest(inadimplente ~ ., data = dados_treino_bal)
modelo_v2

# Avaiando o modelo 2
plot(modelo_v2)

# Previsões com os dados de teste
prevsoes_v2 <- predict(modelo_v2, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v2 <- caret::confusionMatrix(prevsoes_v2, dados_teste$inadimplente, positive="1")
cm_v2

# Calculando Precision, Recall e F1-score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v2 <- prevsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Importância das variáveis preditoras para as previsões
View(dados_treino_bal)
varImpPlot(modelo_v2)

# Obtendo as variáveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            Importance = round(imp_var[,'MeanDecreaseGini'],2))

# Criando o rank de variáveis baseados na importância 
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando o ggplot2 para visualizar a importância de cada variável
ggplot(rankImportance,
       aes(x = reorder(Variables,Importance),
           y = Importance,
           fill = Importance)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()

# Construindo a terceira versão do modelo preditivo apenas com as variáveis mais importântes
colnames(dados_treino_bal)
modelo_v3 <- randomForest(inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1,
                          data = dados_treino_bal)
modelo_v3

# Avaiando o modelo 3
plot(modelo_v3)

# Previsões com os dados de teste
prevsoes_v3 <- predict(modelo_v3, dados_teste)

# Confusion Matrix
cm_v3 <- caret::confusionMatrix(prevsoes_v3, dados_teste$inadimplente, positive="1")
cm_v3

# Calculando Precision, Recall e F1-score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v3 <- prevsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# ----- Salvando os modelos em disco -----
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

# ----- Carregando o modelo final -----
modelo_final <- readRDS("modelo/modelo_v3.rds")

# ----- Previsões com dados de 3 novos clientes -----
# Dados dos clientes
PAY_0 <- c(0,0,0)
PAY_2 <- c(0,0,0)
PAY_3 <- c(1,0,0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0,0,0)
BILL_AMT1 <- c(350, 420, 280)

# Concatena em um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

# Aplicando as transformações as variáveis PAY
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
novos_clientes$PAY_AMT1 <- as.integer(novos_clientes$PAY_AMT1)
novos_clientes$PAY_AMT2 <- as.integer(novos_clientes$PAY_AMT2)
novos_clientes$BILL_AMT1 <- as.integer(novos_clientes$BILL_AMT1)
str(novos_clientes)

# Previsões
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
previsoes_novos_clientes