library(tidyverse)
library(caret)
library(lightgbm)
library(sentimentr)
library(qdapRegex)
library(lubridate)

data_folder = 'D://PRORIV//AltGU//'

emp = read.csv(paste0(data_folder, 'employees.csv'), stringsAsFactors = F, encoding = 'Windows-1251')

train_issues = read.csv(paste0(data_folder, 'train_issues.csv'), stringsAsFactors = F, encoding = 'Windows-1251')
test_issues = read.csv(paste0(data_folder, 'test_issues.csv'), stringsAsFactors = F, encoding = 'Windows-1251')

train_comments = read.csv(paste0(data_folder, 'train_comments.csv'), stringsAsFactors = F, encoding = 'Windows-1251')
test_comments = read.csv(paste0(data_folder, 'test_comments.csv'), stringsAsFactors = F, encoding = 'Windows-1251')

tr_te_comm = rbind(train_comments, test_comments)
tr_te_comm$cnt_words = str_count(tr_te_comm$text, "\\W+")

test_issues$overall_worklogs = NA

train_issues$date = as.Date(train_issues$created, format = "%Y-%m-%d")
test_issues$date = as.Date(test_issues$created, format = "%Y-%m-%d")

tr_te = rbind(train_issues, test_issues)

features1 = tr_te_comm %>% group_by(issue_id) %>% summarize(cnt_comment = length(author_id),
                                                            cnt_comment_authors = length(unique(author_id)))

features2 = tr_te_comm %>% group_by(issue_id) %>% summarize(cnt_words = sum(cnt_words))

colnames(features1)[1] = 'id'
colnames(features2)[1] = 'id'

tr_te = left_join(tr_te, features1)

tr_te = tr_te %>% group_by(project_id, date) %>% mutate(cnt_issues_this_day_by_project = length(id))
tr_te = tr_te %>% group_by(assignee_id, date) %>% mutate(cnt_issues_this_day_by_assignee = length(id))
tr_te = tr_te %>% group_by(creator_id, date) %>% mutate(cnt_issues_this_day_by_creator = length(id))

tr_te$created = as.POSIXct(tr_te$created, format = "%Y-%m-%d %H:%M:%S")

tr_te = tr_te %>% arrange(created) %>% group_by(assignee_id) %>% mutate(prev_task_by_assignee = as.numeric(created - lag(created)), 
                                                                        next_task_by_assignee = as.numeric(lead(created) - created)) %>% data.frame()

tr_te = tr_te %>% arrange(created) %>% group_by(creator_id) %>% mutate(prev_task_by_creator = as.numeric(created - lag(created)), 
                                                                       next_task_by_creator = as.numeric(lead(created) - created)) %>% data.frame()

tr_te$is_assign_creator = ifelse(tr_te$assignee_id == tr_te$creator_id, 1, 0)

tr_te$type = as.numeric(as.factor(sapply(strsplit(tr_te$key, "-"), head, 1)))

colnames(emp)[1] = 'assignee_id'
tr_te = left_join(tr_te, emp[c(1,2,4,5,10:12)])
tr_te$position = as.numeric(as.factor(tr_te$position))
tr_te$hiring_type = as.numeric(as.factor(tr_te$hiring_type))

tr_te = left_join(tr_te, features2)

tr_te$sent = sentiment_by(tr_te$summary)$ave_sentiment

tr_te_comm$comm_sent = sentiment_by(tr_te_comm$text)$ave_sentiment
features3 = tr_te_comm %>% group_by(issue_id) %>% summarize(max_comm_sent = max(comm_sent),
                                                            min_comm_sent = min(comm_sent),
                                                            mean_comm_sent = mean(comm_sent))
colnames(features3)[1] = 'id'
tr_te = left_join(tr_te, features3)

extr_max_dt = function(x) {
dts = unlist(ex_date(x, pattern="@rm_date3"))
max_dt = max(dts)
return(max_dt)
}

tr_te_comm$max_date = NA
for (i in c(1:dim(tr_te_comm)[1]))
  tr_te_comm$max_date[i] = extr_max_dt(tr_te_comm$text[i])

features4 = tr_te_comm %>% group_by(issue_id) %>% summarize(max_date = max(max_date, na.rm = T)) %>% data.frame()
colnames(features4)[1] = 'id'
tr_te = left_join(tr_te, features4)
tr_te$max_date = as.Date(tr_te$max_date, format = "%Y-%m-%d")
tr_te$delta = as.numeric(tr_te$max_date - tr_te$date)

tr_te$wd = wday(tr_te$date)
tr_te$hour = hour(tr_te$created)

train = tr_te[is.na(tr_te$overall_worklogs) == F,]
test = tr_te[is.na(tr_te$overall_worklogs) == T,]

folds = c()
for (proj in unique(train$project_id))
  folds = rbind(folds, data.frame(project_id = proj, 
                                  thresh = as.numeric(quantile(as.numeric(train[train$project_id == proj,]$date), prob = 0.8))))

train$fold = 0
for (i in c(1:dim(folds)[1]))
  train$fold = ifelse(train$project_id == folds$project_id[i] & as.numeric(train$date) > folds$thresh[i], 1, train$fold)
table(train$fold)

param_lgb= list(objective = "regression_l2",
                max_bin = 256,
                learning_rate = 0.01,
                num_leaves = 31,
                bagging_fraction = 0.7,
                feature_fraction = 0.7,
                min_data = 10,
                bagging_freq = 1,
                metric = "score")

lgb_r_sq = function(preds, dtrain){
  rsq <- function (x, y) cor(x, y) ^ 2
  actual = getinfo(dtrain, "label")
  score  = rsq(preds,actual)
  return(list(name = "r_sq", value = score, higher_better = TRUE))
}

exclude = c(1:4,8,9,32,36)
colnames(train)
colnames(train[-c(exclude)])
summary(train[-c(exclude)])

fold = 1
dtrain <- lgb.Dataset(as.matrix(train[train$fold != fold & train$overall_worklogs < 180000,][-c(exclude)]),
                      label = train[train$fold != fold & train$overall_worklogs < 180000,]$overall_worklogs)
dtest = lgb.Dataset(as.matrix(train[train$fold == fold ,][-c(exclude)]),label = train[train$fold == fold ,]$overall_worklogs)
valids = list(test = dtest)
model_lgb1 = lgb.train(data=dtrain, valids = valids, params = param_lgb, nrounds=10000,
                       eval_freq = 100, eval = lgb_r_sq, early_stopping_rounds = 600)

# lgimp = lgb.importance(model_lgb1)

ITERS = 10
test_answers = 0
for (j in c(1:ITERS)) {
    print(j)
    dtrain <- lgb.Dataset(as.matrix(train[train$overall_worklogs < 180000,][-c(exclude)]),
                          label = train[train$overall_worklogs < 180000,]$overall_worklogs)
    model_lgb = lgb.train(data=dtrain, params = param_lgb, nrounds=1300, bagging_seed = 13+j, feature_fraction_seed=42+j)
    answers_iter = predict(model_lgb, as.matrix(test[c(colnames(train)[-c(exclude)])]))
    answers_iter = ifelse(answers_iter < 0, 60, answers_iter)
    test_answers = answers_iter + test_answers
}

sub8 = data.frame(id = test$id, overall_worklogs = test_answers / 10)
write.csv(sub8, paste0(data_folder, 'sub8.csv'), row.names = F, quote = F)
