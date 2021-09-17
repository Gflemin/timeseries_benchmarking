

# Setup -------------------------------------------------------------------

library(tidymodels)
library(modeltime)
library(tidyverse)
library(timetk)
library(fable)

# seed
set.seed(111)

# Modeltime code ----------------------------------------------------------


# load data from timetk and get only needed columns
data_tbl <- walmart_sales_weekly %>%
  select(id, Date, Weekly_Sales) %>%
  set_names(c("id", "date", "value")) %>% 
  group_by(id) %>% 
  arrange(date) %>%
  ungroup()

data_tbl

# showing visualization
data_tbl %>%
  group_by(id) %>%
  plot_time_series(
    date, value, .interactive = T, .facet_ncol = 2
  )

# set seed before each split to ensure exact same split
set.seed(111)

# create train/test split with 3 months assess period
splits1 <- data_tbl %>% 
  time_series_split(
    date_var = date,
    assess     = "5 months", 
    cumulative = TRUE
  )

# set seed before each split to ensure exact same split
set.seed(111)

# create train/test split with 3 months asses period. lag = 5 indicates we will create 5 lags
# saving this for later
splits2 <- data_tbl %>% 
  time_series_split(
    date_var = date,
    assess     = "5 months", 
    cumulative = TRUE,
    lag = 5
  )

# preprocess data by dropping unused IDs, adding a number of feature cols,
# removing date (bc xgboost cant handle it), remove zero variance predictors, and 
# one-hot dummy encoding of categorical vars
rec_obj1 <- recipe(value ~ ., training(splits1)) %>%
  step_mutate_at(id, fn = droplevels) %>%
  step_timeseries_signature(date) %>%
  step_rm(date) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) 


# training global xgboost - much faster than ARIMA iterating through each time series
# to get the right parameters for each (note though that ARIMA models fit faster when
# the pdq terms are specified)
wflw_xgb1 <- workflow() %>%
  add_model(
    boost_tree() %>% set_engine("xgboost")
  ) %>%
  add_recipe(rec_obj1) %>%
  fit(training(splits1)) # training rmse: 3287.501

wflw_xgb1

### RECOMMENDED MODELTIME WORKFLOW

# 1. create a modeltime table of all models you would want (using an already created
# workflow object here. in a different situation, we could specify all these wflow objects
# in the function so long as we already had a recipe and splits object good to go)
model_tbl1 <- modeltime_table(
  wflw_xgb1
)

# 2. "calibrate", aka make forecasts and get residuals/error on test set
calib_tbl1 <- model_tbl1 %>%
  modeltime_calibrate(
    new_data = testing(splits1), 
    id       = "id"
  )

# 3. measure test set accuracy from modeltime_calibrate output for the global 
# accuracy across all models (is this the mean accuracy? unsure)
calib_tbl1 %>% 
  modeltime_accuracy(acc_by_id = FALSE) %>% 
  table_modeltime_accuracy(.interactive = FALSE) # test rmse 4994.88

# per id model performance
calib_tbl1 %>% 
  modeltime_accuracy(acc_by_id = TRUE) %>% 
  mutate(all_rmse = sum(rmse),
         mean_rmse = mean(rmse)) # sum test rmse over all ts: 31,619
                    # mean test rmse: 4,517 

# 4. forecast test data

calib_tbl1 %>%
  modeltime_forecast(
    new_data    = testing(splits1),
    actual_data = data_tbl,
    conf_by_id  = TRUE
  ) %>%
  group_by(id) %>%
  plot_modeltime_forecast(
    .facet_ncol  = 3,
    .interactive = FALSE
  )

# 5. refit and forecast the future
refit_tbl1 <- calib_tbl1 %>%
  modeltime_refit(data = data_tbl)

## get future data
future_tbl <- data_tbl %>%
  group_by(id) %>%
  future_frame(.length_out = 52, .bind_data = FALSE)

future_tbl

## generate future predictions

refit_tbl1 %>%
  modeltime_forecast(
    new_data    = future_tbl,
    actual_data = data_tbl, 
    conf_by_id  = TRUE
  ) %>% 
  group_by(id) %>%
  plot_modeltime_forecast(
    .interactive = F,
    .facet_ncol  = 2
  )



# Modeltime 2: Lags and na_omit -------------------------------------------


# preprocess data by dropping unused IDs, adding a number of feature cols,
# removing date (bc xgboost cant handle it), remove zero variance predictors, and 
# one-hot dummy encoding of categorical vars
rec_obj2 <- recipe(value ~ ., training(splits2)) %>%
  step_mutate_at(id, fn = droplevels) %>%
  step_timeseries_signature(date) %>%
  step_rm(date) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) #%>% 
  # step_lag(value, lag = 1:5) 

# training global xgboost - much faster than ARIMA iterating through each time series
# to get the right parameters for each (note though that ARIMA models fit faster when
# the pdq terms are specified)
wflw_xgb2 <- workflow() %>%
  add_model(
    boost_tree() %>% set_engine("xgboost")
  ) %>%
  add_recipe(rec_obj2) %>%
  fit(training(splits2)) # training rmse: 3287.501 

# QUESTION: is this actually fitting the lags? have lag = 5 on splits but unsure if that
# is actually resulting in any changes to the data that the model is being fit on
wflw_xgb2

### RECOMMENDED MODELTIME WORKFLOW

# 1. create a modeltime table of all models you would want (using an already created
# workflow object here. in a different situation, we could specify all these wflow objects
# in the function so long as we already had a recipe and splits object good to go)
model_tbl2 <- modeltime_table(
  wflw_xgb2
)

# 2. "calibrate", aka make forecasts and get residuals/error on test set
calib_tbl2 <- model_tbl2 %>%
  modeltime_calibrate(
    new_data = testing(splits2), 
    id       = "id",
    quiet = FALSE
  )

# 3. measure test set accuracy from modeltime_calibrate output for the global 
# accuracy across all models (is this the mean accuracy? unsure)
calib_tbl2 %>% 
  modeltime_accuracy(acc_by_id = FALSE) %>% 
  table_modeltime_accuracy(.interactive = FALSE) # test rmse 4586.61


# per id model performance
calib_tbl2 %>% 
  modeltime_accuracy(acc_by_id = TRUE) %>% 
  mutate(all_rmse = sum(rmse),
         mean_rmse = mean(rmse)) # sum test rmse over all ts: 29,317
                                 # mean test rmse over all ts: 4,188

# Fable comparison --------------------------------------------------------


# pull out testing data from splits and filter because we would otherwise get some overlap
testing_filtered = testing(splits2) %>% 
  anti_join(training(splits2), by = c("date" = "date"))

# get test data
test_y = testing_filtered %>% 
  pull(value)

# get forecasting horizon
n_dates = testing_filtered %>% 
  distinct(date) %>% 
  nrow()

# get the training data from splits and make into tsibble for fable
training_tsibble = training(splits2) %>% 
  tsibble::tsibble(key = "id", index = "date") 

# fit fable model
fit_fable = training_tsibble %>% 
  model(arima_auto = ARIMA(value))

# forecast
fit_fable %>% 
  forecast(h = n_dates) %>% 
  left_join(testing_filtered, by = c("id" = "id", "date" = "date")) %>% 
  as_tibble() %>% 
  select(-value.x) %>% 
  group_by(id) %>% 
  summarize(rmse = sqrt(((sum(.mean - value.y))^2)/(nrow(.)))) %>% 
  mutate(all_rmse = sum(rmse),
         mean_rmse = mean(rmse)) # sum test rmse over all ts: 43,486
                                 # mean test rmse over all ts: 6,212


